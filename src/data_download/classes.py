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
    def __init__(self, downloads_dir, num_docs_check,num_docs_download, metadata_filename,LLM_model_name="gpt-4.1",LLM_vendor_name="openai",embedding_model_name="all-MiniLM-L6-v2"):
        
        self.downloads_dir = downloads_dir
        self.num_docs_check = num_docs_check
        self.num_docs_download = num_docs_download
        self.metadata_file = metadata_filename
        self.LLM_model_name = LLM_model_name
        self.LLM_vendor_name = LLM_vendor_name
        self.embedding_model_name = embedding_model_name
        self.cutoff_score=0.5

        self.instr_filename=Path("src/lib/LLM/LLM_instr_files/semantically_similar_queries_instr.txt")
        self.api_key_string="OPENAI_API_KEY" 
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


    def generate_semantically_similar_queries(self,user_query):
        """
        Generate semantically similar queries to the user query.
        """
        
        model=create_LLM_instance(self.api_key_string,self.reference_file_paths,self.LLM_model_name,self.LLM_vendor_name)
        result=model.query_model(user_query,self.instr_filename)

        result_list=result.split(",")
        print(result_list)
        print(f"Generated {len(result_list)} semantically similar queries:{result_list}")
        return result_list

    def compute_cosine_similarity(self, query_embeddings, original_query_embedding):
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

    def get_high_similarity_indices(self, similarity_matrix):
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

    def filter_titles_and_abstracts(self,papers_metadata, queries):
        """
        Use an embedding model to filter the titles and abstracts of the papers metadata to only include those
        that are semantically similar to the user query.

        Steps:
        1. compute cosine similarity scores of titles with queries(user-specified and semantically similar)
        2. compute cosine similarity scores of abstracts with queries (user-specified and semantically similar)
        3. sort the scores of titles and abstracts for each query
        4. select the top num_docs_download papers for each query
        5. return the metadata of the selected papers
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

        print(title_query_embedding.shape)
        print(abstract_query_embedding.shape)

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

        common_indices=np.array(list(set(high_similarity_indices_title).intersection(set(high_similarity_indices_abstract))))
        
        
        common_title_matrix=title_similarity_matrix[:,common_indices]
        common_abstract_matrix=abstract_similarity_matrix[:,common_indices]

        # average the similarity scores of the titles and abstracts
        common_matrix=(common_title_matrix+common_abstract_matrix)/2

        # sort the papers by the average similarity scores
        sorted_indices=np.argsort(np.max(common_matrix, axis=0), axis=0)[::-1]

        # return the indices of the sorted papers
        return_indices=common_indices[sorted_indices]

        print(return_indices.shape)

        return return_indices

    def download_pdf_for_paper(self, paper_metadata, downloads_dir, paper_index):
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

    def download_pdf_data(self,user_query):
        """

        
        """
        # Create downloads directory if it doesn't exist
        if os.path.isdir(self.downloads_dir):
            shutil.rmtree(self.downloads_dir)
        os.makedirs(self.downloads_dir, exist_ok=True)
        
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


        filtered_papers_metadata=[]
        for index in filtered_indices:
            filtered_papers_metadata.append(all_papers_metadata[index])

        # Download PDFs for filtered papers
        print(f"\nDownloading PDFs for {len(filtered_papers_metadata)} filtered papers...")
        successful_downloads = 0
        

        
        for idx, paper in enumerate(filtered_papers_metadata, start=1):
            if self.download_pdf_for_paper(paper, self.downloads_dir, idx):
                successful_downloads += 1
                
            # Limit downloads to num_docs_download
            #if successful_downloads >= self.num_docs_download:
            #    print(f"Reached download limit of {self.num_docs_download} papers")
            #    break
        
        print(f"Successfully downloaded {successful_downloads} PDFs to {self.downloads_dir}")

        # Save metadata to CSV
        with open(self.metadata_file, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Index", "Title", "Abstract", "DOI", "ArXiv Link", "Authors", "Published Date"])
            for paper in filtered_papers_metadata:
                writer.writerow([
                    paper['index'],
                    paper['title'],
                    paper['abstract'],
                    paper['doi'],
                    paper['arxiv_link'],
                    ', '.join(paper['authors']),
                    paper['published_date']
                ])

        print(f"\nMetadata saved to {self.metadata_file}")
        return filtered_papers_metadata
