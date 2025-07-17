from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from datasets import load_from_disk
from src.lib.LLM.helper_functions import create_LLM_instance
import numpy as np

class ModelInference:
    def __init__(self,model_output_folder,model_name,path_to_dataset,LLM_vendor_name,LLM_model_name,LLM_instr_filename,API_key_string=None):
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

    def load_model(self):
        self.model= SentenceTransformer(str(self.embedding_model_path))
        self.index = faiss.read_index(str(self.faiss_index_path))
    
    def load_dataset(self):
        self.dataset = load_from_disk(self.path_to_dataset)

    def get_context_from_dataset(self,indices):
        returned_texts_with_metadata=[]
        for index in np.squeeze(indices):
            returned_texts_with_metadata.append(self.dataset[int(index)])
        return returned_texts_with_metadata

    def prepare_question_for_llm(self,question,returned_texts_with_metadata):
        question_with_context=f"Question:\n {question}\n\n"

        for idx,text_with_metadata in enumerate(returned_texts_with_metadata):
            question_with_context+=f"Context {idx+1}:\n {text_with_metadata['text']}\n"
            question_with_context+=f"Title: {text_with_metadata['Title']}\n"
            question_with_context+=f"Page Nos: {text_with_metadata['page_nos']}\n"
            question_with_context+=f"Arxiv Link: {text_with_metadata['ArXiv Link']}\n"
            question_with_context+="\n\n"

        return question_with_context

    def query_llm(self,formatted_question):

        llm_model=create_LLM_instance(self.API_key_string,self.reference_file_paths,self.LLM_model_name,self.LLM_vendor_name)

        reply = llm_model.query_model(formatted_question,self.LLM_instr_filename)

        return reply


    def query_model(self,question,k=5):
        print("Thinking...")
        query_embedding = self.model.encode_query([question]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        
        returned_texts_with_metadata=self.get_context_from_dataset(indices)

        formatted_question=self.prepare_question_for_llm(question,returned_texts_with_metadata)
        reply=self.query_llm(formatted_question)
        return reply



        

        