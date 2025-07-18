from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from datasets import load_from_disk
from src.lib.LLM.helper_functions import create_LLM_instance
import numpy as np

class ModelInference:
    def __init__(self, model_output_folder: Path, model_name: str, path_to_dataset: Path, LLM_vendor_name: str, LLM_model_name: str, LLM_instr_filename: Path, API_key_string: str = None) -> None:
        self.model_output_folder=model_output_folder
        self.model_name=model_name
        self.path_to_dataset=path_to_dataset
        self.embedding_model_path=model_output_folder/Path("embedding_model/")
        self.faiss_index_path=model_output_folder/Path("faiss_index.idx")
        if API_key_string is None:
            raise ValueError("API key string is required")
        self.API_key_string=API_key_string
        self.reference_file_paths=[]
        self.LLM_vendor_name=LLM_vendor_name
        self.LLM_model_name=LLM_model_name
        self.LLM_instr_filename=LLM_instr_filename
        self.load_model()
        self.load_dataset()

    def load_model(self) -> None:
        """
        Load the trained embedding model and FAISS index from disk.
        
        This function loads the sentence transformer model and FAISS vector index
        that were created by the model creation pipeline, making them available
        for inference operations.
        
        Args:
            None (uses self.embedding_model_path and self.faiss_index_path from class instance)
            
        Returns:
            None (sets self.model and self.index attributes)
        """
        self.model= SentenceTransformer(str(self.embedding_model_path))
        self.index = faiss.read_index(str(self.faiss_index_path))
    
    def load_dataset(self) -> None:
        """
        Load the processed dataset from disk.
        
        This function loads the HuggingFace Dataset that was created by the data extraction
        pipeline, which contains the processed text chunks and metadata needed for
        retrieval-augmented generation.
        
        Args:
            None (uses self.path_to_dataset from class instance)
            
        Returns:
            None (sets self.dataset attribute)
        """
        self.dataset = load_from_disk(self.path_to_dataset)

    def get_context_from_dataset(self, indices: np.ndarray) -> list[dict]:
        """
        Retrieve text chunks and metadata from the dataset based on FAISS search indices.
        
        This function takes the indices returned by FAISS similarity search and
        retrieves the corresponding text chunks and metadata from the dataset.
        The returned data will be used as context for the LLM.
        
        Args:
            indices (np.ndarray): Array of indices returned by FAISS search
            
        Returns:
            list[dict]: List of dictionaries containing text chunks and metadata
                       (text, Title, page_nos, ArXiv Link, etc.)
        """
        returned_texts_with_metadata=[]
        for index in np.squeeze(indices):
            returned_texts_with_metadata.append(self.dataset[int(index)])
        return returned_texts_with_metadata

    def prepare_question_for_llm(self, question: str, returned_texts_with_metadata: list[dict]) -> str:
        """
        Format the user question with retrieved context for LLM processing.
        
        This function creates a structured prompt that combines the user's question
        with the retrieved context from the dataset. The formatted prompt includes
        the question followed by numbered context sections, each containing the
        text chunk, title, page numbers, and arXiv link for proper citation.
        
        Args:
            question (str): The user's original question
            returned_texts_with_metadata (list[dict]): List of retrieved text chunks with metadata
            
        Returns:
            str: Formatted question with context for LLM processing
            
        Format:
            Question: [user_question]
            
            Context 1:
            [text_chunk]
            Title: [paper_title]
            Page Nos: [page_numbers]
            Arxiv Link: [arxiv_link]
            
            Context 2:
            ...
        """
        question_with_context=f"Question:\n {question}\n\n"

        for idx,text_with_metadata in enumerate(returned_texts_with_metadata):
            question_with_context+=f"Context {idx+1}:\n {text_with_metadata['text']}\n"
            question_with_context+=f"Title: {text_with_metadata['Title']}\n"
            question_with_context+=f"Page Nos: {text_with_metadata['page_nos']}\n"
            question_with_context+=f"Arxiv Link: {text_with_metadata['ArXiv Link']}\n"
            question_with_context+="\n\n"

        return question_with_context

    def query_llm(self, formatted_question: str) -> str:
        """
        Send formatted question to LLM and retrieve response.
        
        This function creates an LLM instance with the configured vendor and model,
        then sends the formatted question (with context) to the LLM using the
        specified instruction file for retrieval-augmented generation.
        
        Args:
            formatted_question (str): Question formatted with retrieved context
            
        Returns:
            str: LLM response based on the provided context
            
        Note:
            The LLM is configured with the instruction file specified in
            self.LLM_instr_filename to ensure proper RAG behavior and citation format.
        """
        llm_model=create_LLM_instance(self.API_key_string,self.reference_file_paths,self.LLM_model_name,self.LLM_vendor_name)

        reply = llm_model.query_model(formatted_question,self.LLM_instr_filename)

        return reply


    def query_model(self, question: str, k: int = 5) -> str:
        """
        Main inference function to answer user questions using retrieval-augmented generation.
        
        This function orchestrates the complete RAG pipeline: encoding the user question,
        performing similarity search to retrieve relevant context, formatting the question
        with context, and generating an answer using the LLM.
        
        Args:
            question (str): The user's question to answer
            k (int): Number of most similar documents to retrieve (default: 5)
            
        Returns:
            str: LLM-generated answer based on retrieved context
            
        Workflow:
        1. Encode the user question using the sentence transformer model
        2. Perform similarity search using FAISS index to find k most similar documents
        3. Retrieve the corresponding text chunks and metadata from the dataset
        4. Format the question with retrieved context and metadata
        5. Send formatted question to LLM for answer generation
        6. Return the LLM response
        
        Note:
            The function implements a complete RAG pipeline that ensures answers are
            based on the provided scientific literature context with proper citations.
        """
        print("Thinking...")
        query_embedding = self.model.encode_query([question]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        
        returned_texts_with_metadata=self.get_context_from_dataset(indices)

        formatted_question=self.prepare_question_for_llm(question,returned_texts_with_metadata)
        reply=self.query_llm(formatted_question)
        return reply



        

        