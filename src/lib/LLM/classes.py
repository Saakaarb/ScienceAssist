import os
from openai import OpenAI
from tqdm import tqdm

class LLMBase():

    def __init__(self,name,vendor,model_name,API_key_string):

        self.name=name # model name
        # Vendor and model info
        #---------------------------
        self.vendor=vendor # model vendor
        self.model_name=model_name # model name
        self.API_key_string=API_key_string
        self.vector_store_id=None
        # Get API key from environment
        self.set_API_key(API_key_string)
        #---------------------------

        self.init_client()


    def set_API_key(self,API_key_string=None):
        
        
        self.API_key=os.getenv(API_key_string)
        if not self.API_key:
            raise ValueError(f"API key not found in environment variable {API_key_string}") 

    def init_client(self):

        return self._init_client()
    
    def create_vector_database(self,text_list):

        return self._create_vector_database(text_list)

    def query_vector_store(self,query):

        return self._query_vector_store(query)

    def delete_vector_store(self):

        return self._delete_vector_store()

    def query_model(self,query,instr_filename):

        return self._query_model(query,instr_filename)


class OpenAI_model(LLMBase):
   
    # Sets API key and initializes client
    def __init__(self, api_key_string, reference_file_paths=[],model_name="gpt-4.1"):
        super().__init__("OpenAI_model", "openai", model_name, api_key_string)
        self.reference_file_paths = reference_file_paths
        
        if not self.API_key:
            raise ValueError(f"API key not found in environment variable {api_key_string}") 

    def _init_client(self):
        self.client = OpenAI(api_key=self.API_key)

    def _set_instructions(self,instr_filename):
        
        with open(instr_filename,"r",encoding="utf=8") as f:

            model_instr=f.read()
        return model_instr
    
    def _create_vector_database(self,text_list):

        self._create_vector_store()

        self._upload_text_to_vector_store(text_list)

    # vector store commands
    # need to create a new vector store and delete it after use
    def _create_vector_store(self):

        # store created vector store path
        vector_store = self.client.vector_stores.create(
        name="knowledge_base"
            )
        self.vector_store_id=vector_store.id

        # just in case vector store is not deleted
        self.client.vector_stores.update(
        vector_store_id=self.vector_store_id,
        expires_after={
            "anchor": "last_active_at",
            "days": 1
        }
        )

    def _upload_text_to_vector_store(self,text_list):

        # Upload each text chunk individually
        import tempfile
        import os
        
        file_ids = []
        
        for i, text in tqdm(enumerate(text_list),total=len(text_list),desc="Uploading text to vector store"):
            # Create a temporary file for each chunk
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                temp_file.write(text)
                temp_file_path = temp_file.name
            
            # Upload the file to OpenAI
            with open(temp_file_path, 'rb') as file:
                uploaded_file = self.client.files.create(
                    file=file,
                    purpose='assistants'
                )
                file_ids.append(uploaded_file.id)
            
            # Clean up temporary file
            os.unlink(temp_file_path)
        
        # Add all files to vector store
        self.client.vector_stores.file_batches.create(
            vector_store_id=self.vector_store_id,
            file_ids=file_ids
        )
    
    def _query_vector_store(self,query):
        
        response=self.client.vector_stores.search(
            vector_store_id=self.vector_store_id,
            query=query
        )
        return response

    def _delete_vector_store(self):

        self.client.vector_stores.delete(vector_store_id=self.vector_store_id)


    def _query_model(self,query,instr_filename):

        model_instr=self._set_instructions(instr_filename)

        # Prepare input content
        input_content = [{"type":"input_text", "text":query}]
        

        response= self.client.responses.create(
                    model=self.model_name,
                    instructions=model_instr,
                    input= [{
                            "role":"user",
                            "content": input_content,
                    }],
                    
                    )

        return response.output_text