from datasets import load_from_disk
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import shutil
import os

class ModelCreator: 
    def __init__(self,processed_data_location,processed_dataset_name,model_output_folder,model_name,embedding_model_name):
        self.processed_data_location=processed_data_location
        self.processed_dataset_name=processed_dataset_name
        self.model_output_folder=model_output_folder
        self.model_name=model_name
        self.embedding_model_name=embedding_model_name
        self.faiss_index_path=model_output_folder/Path("faiss_index.idx")
        self.embedding_model_path=model_output_folder/Path("embedding_model")
        

    def load_dataset(self):
        dataset = load_from_disk(self.processed_data_location/self.processed_dataset_name)
        return dataset

    def get_texts_from_dataset(self,dataset):
        all_text=[]
        for example in dataset:
            all_text.append(example["text"])
        return all_text

    def create_model(self):
        if os.path.isdir(self.model_output_folder):
            shutil.rmtree(self.model_output_folder)
        os.makedirs(self.model_output_folder, exist_ok=True)

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