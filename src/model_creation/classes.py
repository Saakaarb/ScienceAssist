from datasets import load_from_disk, Dataset
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import shutil
import os

class ModelCreator: 
    def __init__(self, processed_data_location: Path, model_output_folder: Path, embedding_model_name: str) -> None:
        self.processed_data_location=processed_data_location
        self.model_output_folder=model_output_folder
        self.embedding_model_name=embedding_model_name
        self.faiss_index_path=model_output_folder/Path("faiss_index.idx")
        self.embedding_model_path=model_output_folder/Path("embedding_model")
        

    def load_dataset(self) -> Dataset:
        """
        Load the processed dataset from disk.
        
        This function loads the HuggingFace Dataset that was created by the data extraction
        pipeline, which contains the processed text chunks and metadata.
        
        Args:
            None (uses self.processed_data_location from class instance)
            
        Returns:
            Dataset: HuggingFace Dataset object containing processed text chunks and metadata
        """
        dataset = load_from_disk(self.processed_data_location)
        return dataset

    def get_texts_from_dataset(self, dataset: Dataset) -> list[str]:
        """
        Extract all text chunks from the processed dataset.
        
        This function iterates through the dataset and extracts the 'text' field from each
        example, creating a list of all text chunks that will be used for embedding generation.
        
        Args:
            dataset (Dataset): HuggingFace Dataset object containing processed text chunks
            
        Returns:
            list[str]: List of all text chunks from the dataset
        """
        all_text=[]
        for example in dataset:
            all_text.append(example["text"])
        return all_text

    def create_model(self) -> None:
        """
        Main pipeline function to create embedding model and vector database.
        
        This function orchestrates the complete model creation workflow: loading the processed
        dataset, extracting text chunks, generating embeddings using a sentence transformer
        model, creating a FAISS vector index for efficient similarity search, and saving
        both the embedding model and index to disk.
        
        Args:
            None (uses class instance variables for configuration)
            
        Returns:
            None (saves model and index to self.model_output_folder)
            
        Workflow:
        1. Load processed dataset from disk
        2. Extract all text chunks from the dataset
        3. Load sentence transformer model from remote repository
        4. Generate embeddings for all text chunks
        5. Create FAISS index with L2 distance metric
        6. Add embeddings to the index
        7. Save FAISS index to disk
        8. Save embedding model to disk
        
        Files Created:
        - faiss_index.idx: FAISS vector index for similarity search
        - embedding_model/: Directory containing the saved sentence transformer model
        """
        print("Loading dataset...")
        dataset=self.load_dataset()

        all_text=self.get_texts_from_dataset(dataset)

        print(f"Loading model named {self.embedding_model_name} from remote...")
        model = SentenceTransformer(self.embedding_model_name) 
        

        # creating chunk embeddings
        print("Creating chunk embeddings, this could take a while...")
        embeddings = model.encode(all_text, convert_to_tensor=True) 
    
        # create vector database

        embedding_dim = embeddings.shape[1]
        
        index = faiss.IndexFlatL2(embedding_dim)

        print("Creating indexing model..")
        index.add(np.array(embeddings).astype('float32'))

        # save the index model
        faiss.write_index(index, str(self.faiss_index_path))

        # save the embedding model
        model.save(str(self.embedding_model_path))