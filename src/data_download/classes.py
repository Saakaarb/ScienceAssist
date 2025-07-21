import feedparser
import csv
from sentence_transformers import SentenceTransformer
from src.lib.LLM.helper_functions import create_LLM_instance
from pathlib import Path
import os
from urllib.parse import quote
import numpy as np
import requests
import shutil

# Class to download correct data from arXiv based on user request
# 1. Accept a user query on a topic
# 2. Generate N semantically similar queries. These plus user query will be used to fetch the data from arXiv.
# 3. Download the metadata of papers from these queries.
# 4. Use the titles and abstracts of the papers metadata to only keep those that have high semantical similarity to AT LEAST one of the user queries.
# 5. Sort the filtered papers by the best similarity scores of the titles and abstracts to the user queries. If the downlaod limit is hit the most relevant papers are selected.
# 6. Download the PDFs of the filtered papers.
# 7. Save the metadata to a CSV file


class DataDownloader:
    def __init__(self, downloads_dir: Path, num_docs_check: int, num_docs_download: int, metadata_filename: Path, API_key_string: str = None, LLM_model_name: str = "gpt-4.1", LLM_vendor_name: str = "openai", embedding_model_name: str = "all-MiniLM-L6-v2", cutoff_score: float = 0.5) -> None:
        
        self.downloads_dir = downloads_dir
        self.num_docs_check = num_docs_check
        self.num_docs_download = num_docs_download
        self.metadata_file = metadata_filename
        self.LLM_model_name = LLM_model_name
        self.LLM_vendor_name = LLM_vendor_name
        self.embedding_model_name = embedding_model_name
        self.cutoff_score=cutoff_score # cutoff score for similarity scores
        self.config_file_path=Path("config")/Path("data_download_config.yaml")
        self.instr_filename=Path("src/lib/LLM/LLM_instr_files/semantically_similar_queries_instr.txt")
        if API_key_string is None:
            raise ValueError("API key string is required")
        
        self.api_key_string=API_key_string
        self.reference_file_paths=[]

    def fetch_arxiv_metadata(self,query, num_docs_check):
        """
        Fetch titles and abstracts from arXiv based on query.
        
        Args:
            query (str): Search query for arXiv
            num_docs_download (int): Number of papers to fetch
            
        Returns:
            list: List of dictionaries containing paper metadata (title, abstract, arxiv_link, doi)
        """
        # URL for arXiv API - this returns XML with titles, abstracts, and metadata
        # Properly encode the query to handle spaces and special characters
        encoded_query = quote(query)
        url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={num_docs_check}&sortBy=relevance"
        feed = feedparser.parse(url)

        if len(feed.entries) == 0:
            raise ValueError("No papers found; check your query to make sure it is coherent and will return results on arXiv. Also check your internet connection.")

        print(f"Number of papers found: {len(feed.entries)}")

        papers_metadata = []
        for idx, entry in enumerate(feed.entries, start=1):
            title = entry.title.replace("\n", " ").strip()
            abstract = entry.summary.replace("\n", " ").strip()
            arxiv_link = entry.id
            doi = entry.get('arxiv_doi', 'N/A')  # DOI might not be present
            
            # Extract additional metadata if available
            authors = [author.name for author in entry.authors] if hasattr(entry, 'authors') else []
            published_date = entry.published if hasattr(entry, 'published') else 'N/A'
            
            paper_info = {
                'title': title,
                'abstract': abstract,
                'arxiv_link': arxiv_link,
                'doi': doi,
                'authors': authors,
                'published_date': published_date,
                'index': idx
            }
            
            papers_metadata.append(paper_info)
            print(f"Paper {idx}: {title}")

        return papers_metadata

    def save_config_copy(self) -> None:
        """
        Save a copy of the configuration file to the model output directory.
        
        This function copies the model creation configuration file to the model output
        directory for reproducibility and documentation purposes.
        
        Args:
            None (uses self.config_file_path and self.model_output_folder from class instance)
            
        Returns:
            None (saves config file to model output directory)
            
        Note:
            If no config file path is provided, this function does nothing.
        """
        if self.config_file_path and self.config_file_path.exists():
            config_copy_path = self.downloads_dir / Path("data_download_config.yaml")
            shutil.copy2(self.config_file_path, config_copy_path)
            print(f"Configuration file saved to: {config_copy_path}")
        elif self.config_file_path:
            print(f"Warning: Configuration file not found at {self.config_file_path}")
        else:
            print("No configuration file path provided, skipping config copy")

    def generate_semantically_similar_queries(self,user_query: str) -> list[str]:
        """
        Generate semantically similar queries to the user query using an LLM.
        
        This function uses a pre-trained language model to expand the user's original query
        into multiple semantically similar queries. This helps to broaden the search scope
        and capture papers that might be relevant but use different terminology.
        
        Args:
            user_query (str): The original search query provided by the user
            
        Returns:
            list[str]: List of semantically similar queries generated by the LLM
            
        Note:
            The function uses the instruction file specified in self.instr_filename to guide
            the LLM in generating appropriate semantically similar queries. The LLM is
            configured with the API key and model settings from the class instance.
        """
        
        model=create_LLM_instance(self.api_key_string,self.reference_file_paths,self.LLM_model_name,self.LLM_vendor_name)
        result=model.query_model(user_query,self.instr_filename)

        result_list=result.split(",")
        
        print(f"Generated {len(result_list)} semantically similar queries:{result_list}")
        return result_list

    def compute_cosine_similarity(self, query_embeddings: np.ndarray, original_query_embedding: np.ndarray) -> np.ndarray:
        """
        Efficiently compute cosine similarity between query embeddings and title embeddings.
        
        Args:
            query_embeddings (np.ndarray): All embeddings [queries + titles] with shape (num_queries + num_titles, embedding_dim)
            num_original_queries (int): Number of original query embeddings
            
        Returns:
            np.ndarray: Cosine similarity matrix with shape (num_original_queries, num_titles)
        """
        # Extract query embeddings (first num_original_queries rows)
        query_emb = original_query_embedding # Shape: (num_original_queries, embedding_dim)
        
        # Extract title embeddings (remaining rows after queries)
        title_emb = query_embeddings  # Shape: (num_titles, embedding_dim)
        
        # Normalize embeddings for efficient cosine similarity computation
        query_norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
        title_norm = np.linalg.norm(title_emb, axis=1, keepdims=True)
        
        # Avoid division by zero
        query_norm = np.where(query_norm == 0, 1, query_norm)
        title_norm = np.where(title_norm == 0, 1, title_norm)
        
        # Normalize embeddings
        query_emb_normalized = query_emb / query_norm
        title_emb_normalized = title_emb / title_norm
        
        # Compute cosine similarity using matrix multiplication
        # cosine_sim = (A_normalized) @ (B_normalized).T
        similarity_matrix = np.dot(query_emb_normalized, title_emb_normalized.T)
        
        return similarity_matrix

    def get_high_similarity_indices(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        Efficiently find indices of titles where at least one query has similarity score above cutoff.
        
        Args:
            similarity_matrix (np.ndarray): Similarity matrix with shape (num_original_queries, num_titles)
            
        Returns:
            np.ndarray: Indices of titles that meet the cutoff criteria
        """
        # Check which titles have at least one query similarity above cutoff
        # Use np.any() along axis=0 to check if any query (row) has score > cutoff for each title (column)
        high_similarity_mask = np.any(similarity_matrix > self.cutoff_score, axis=0)
        
        # Get indices where the mask is True
        high_similarity_indices = np.where(high_similarity_mask)[0]
        
        return high_similarity_indices

    def filter_titles_and_abstracts(self,papers_metadata: list[dict], queries: list[str]) -> np.ndarray:
        """
        Filter papers by semantic similarity of titles and abstracts to user queries.
        
        This function uses an embedding model to compute semantic similarity between paper titles/abstracts
        and the user queries (both original and semantically similar queries). It filters papers to only
        include those that have high semantic similarity to at least one of the queries, then sorts them
        by relevance score.
        
        Args:
            papers_metadata (list[dict]): List of paper metadata dictionaries, each containing 'title' and 'abstract' keys
            queries (list[str]): List of search queries (original user query + semantically similar queries)
            
        Returns:
            np.ndarray: Array of indices corresponding to papers that meet the similarity criteria,
                       sorted by relevance score (most relevant first)
                       
        Process:
        1. Encode all queries, titles, and abstracts using the embedding model
        2. Compute cosine similarity between queries and titles
        3. Compute cosine similarity between queries and abstracts  
        4. Find papers where either title OR abstract has similarity above cutoff score
        5. Average title and abstract similarity scores for each paper
        6. Sort papers by maximum similarity score across all queries
        7. Return sorted indices of papers that meet criteria
        """

        # create embedding model instance
        embedding_model= SentenceTransformer(self.embedding_model_name)

        num_original_queries=len(queries)

        title_query_list=[]#queries.copy()
        abstract_query_list=[]#queries.copy()

        for paper_data in papers_metadata:
            title_query_list.append(paper_data['title'])
            abstract_query_list.append(paper_data['abstract'])

        original_query_embedding=embedding_model.encode(queries).astype('float32')
        title_query_embedding=embedding_model.encode(title_query_list).astype('float32')
        abstract_query_embedding=embedding_model.encode(abstract_query_list).astype('float32')

        # Compute cosine similarity for titles
        title_similarity_matrix = self.compute_cosine_similarity(
            title_query_embedding,
            original_query_embedding,
        )
        
        # Compute cosine similarity for abstracts
        abstract_similarity_matrix = self.compute_cosine_similarity(
            abstract_query_embedding,
            original_query_embedding,
        )
        
        print(f"Title similarity matrix shape: {title_similarity_matrix.shape}")
        print(f"Abstract similarity matrix shape: {abstract_similarity_matrix.shape}")
        
        # Example: Get similarity scores for first query to all titles
        if title_similarity_matrix.shape[0] > 0 and title_similarity_matrix.shape[1] > 0:
            first_query_scores = title_similarity_matrix[0, :]
            print(f"Similarity scores for first query: {first_query_scores[:5]}...")  # Show first 5 scores
        

        high_similarity_indices_title=self.get_high_similarity_indices(title_similarity_matrix)
        print(f"Number of titles with high title similarity: {len(high_similarity_indices_title)}")
        #print(f"Indices of titles with high similarity: {high_similarity_indices}")

        high_similarity_indices_abstract=self.get_high_similarity_indices(abstract_similarity_matrix)
        print(f"Number of abstracts with high abstract similarity: {len(high_similarity_indices_abstract)}")
        #print(f"Indices of abstracts with high similarity: {high_similarity_indices}")

        common_indices=np.array(list(set(high_similarity_indices_title).union(set(high_similarity_indices_abstract))))
        
        
        common_title_matrix=title_similarity_matrix[:,common_indices]
        common_abstract_matrix=abstract_similarity_matrix[:,common_indices]

        # average the similarity scores of the titles and abstracts
        common_matrix=(common_title_matrix+common_abstract_matrix)/2

        # sort the papers by the average similarity scores
        sorted_indices=np.argsort(np.max(common_matrix, axis=0), axis=0)[::-1]

        # return the indices of the sorted papers
        return_indices=common_indices[sorted_indices]

        return return_indices

    def download_pdf_for_paper(self, paper_metadata: dict, downloads_dir: Path, paper_index: int) -> bool:
        """
        Download PDF for a single paper using its arXiv link.
        
        Args:
            paper_metadata (dict): Paper metadata containing arxiv_link
            downloads_dir (str): Directory to save PDFs
            paper_index (int): Index for naming the PDF file
            
        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            # Convert arXiv abs link to PDF link
            pdf_url = paper_metadata['arxiv_link'].replace('abs', 'pdf') + ".pdf"
            title = paper_metadata['title']
            
            # Create filename
            file_name = f"paper_{paper_index}.pdf"
            file_path = os.path.join(downloads_dir, file_name)
            
            print(f"Downloading: {file_name} -> {title}")
            
            # Download PDF
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()  # Raise exception for bad status codes
            
            # Save PDF
            with open(file_path, 'wb') as f:
                f.write(response.content)
                
            print(f"Successfully downloaded: {file_name}")
            return True
            
        except Exception as e:
            print(f"Failed to download {title}: {e}")
            return False

    def download_pdf_data(self,user_query: str) -> list[dict]:
        """
        Main pipeline function to download relevant PDF papers based on user query.
        
        This function orchestrates the complete data download workflow: generating semantically similar
        queries, fetching paper metadata from arXiv, filtering papers by semantic similarity, downloading
        PDFs, and saving metadata to CSV. It implements a sophisticated retrieval system that combines
        query expansion with semantic filtering to find the most relevant papers.
        
        Args:
            user_query (str): The original search query provided by the user
            
        Returns:
            list[dict]: List of paper metadata dictionaries for all filtered papers, including
                       downloaded PDF filenames and all original metadata (title, abstract, DOI, etc.)
                       
        Workflow:
        1. Generate semantically similar queries using LLM
        2. Fetch metadata for papers from arXiv using all queries (original + similar)
        3. Filter papers by semantic similarity of titles/abstracts to queries
        4. Limit downloads to user-specified number (num_docs_download)
        5. Download PDFs for filtered papers with error handling
        6. Save comprehensive metadata to CSV file
        7. Return metadata for all processed papers
        
        Files Created:
        - PDF files: paper_1.pdf, paper_2.pdf, etc. in downloads_dir
        - CSV file: metadata.csv with all paper information and filenames
        """
        
        # generate semantically similar queries to the user query
        semantically_similar_queries=self.generate_semantically_similar_queries(user_query)

        semantically_similar_queries.insert(0,user_query)

        all_papers_metadata=[]
        for query in semantically_similar_queries:
            papers_metadata = self.fetch_arxiv_metadata(query, self.num_docs_check)
            all_papers_metadata.extend(papers_metadata)

        print(f"{len(all_papers_metadata)} initial titles are selected for processing, narrowing down papers..")
        
        # filter papers by similarity scores of titles and abstracts to queries
        # the return indices are sorted by similarity scores to have the most relevant papers first
        filtered_indices = self.filter_titles_and_abstracts(all_papers_metadata,semantically_similar_queries)

        if len(filtered_indices) > self.num_docs_download:
            print(f"Number of papers to download is greater than the user-set download limit. Downloading {self.num_docs_download} most relevant papers.")
            filtered_indices=filtered_indices[:self.num_docs_download]
        else:
            print(f"Number of papers to download is less than the user-set download limit. Downloading {len(filtered_indices)} papers.")

        # Download PDFs for filtered papers and create metadata list in one loop
        print(f"\nDownloading PDFs for {len(filtered_indices)} filtered papers...")
        successful_downloads = 0
        filtered_papers_metadata = []
        
        for idx, index in enumerate(filtered_indices, start=1):
            paper = all_papers_metadata[index]
            
            # Download PDF
            if self.download_pdf_for_paper(paper, self.downloads_dir, idx):
                successful_downloads += 1
                # Add filename to paper metadata
                paper['filename'] = f"paper_{idx}.pdf"
            
            # Add paper to filtered metadata (regardless of download success)
            filtered_papers_metadata.append(paper)
        
        print(f"Successfully downloaded {successful_downloads} PDFs to {self.downloads_dir}")

        # Save metadata to CSV
        with open(self.metadata_file, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Index", "Title", "Abstract", "DOI", "ArXiv Link", "Authors", "Published Date", "Filename"])
            for paper in filtered_papers_metadata:
                writer.writerow([
                    paper['index'],
                    paper['title'],
                    paper['abstract'],
                    paper['doi'],
                    paper['arxiv_link'],
                    ', '.join(paper['authors']),
                    paper['published_date'],
                    paper.get('filename', 'N/A')  # Use get() to handle cases where download failed
                ])

        print(f"\nMetadata saved to {self.metadata_file}")
        self.save_config_copy()
        return filtered_papers_metadata
