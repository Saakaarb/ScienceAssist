import os
from openai import OpenAI

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
        
        print("Note: Set the env key as an environment variable")
        self.API_key=os.getenv(API_key_string)
        if not self.API_key:
            raise ValueError(f"API key not found in environment variable {API_key_string}") 

    def init_client(self):

        return self._init_client()
    
    def create_vector_database(self):

        return self._create_vector_database()

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


    def _upload_text_to_vector_store(self,text_list):

        self.client.vector_stores.files.create(
            vector_store_id=self.vector_store_id,
            file=text_list
        )

    def _delete_vector_store(self):

        self.client.vector_stores.delete(vector_store_id=self.vector_store_id)


    def _query_model(self,query,instr_filename):

        model_instr=self._set_instructions(instr_filename)

        # Prepare input content
        input_content = [{"type":"input_text", "text":query}]
        
        # Add vector store if provided
        if self.vector_store_id is not None:
            
            input_content.append({
                "type": "vector_store",
                "vector_store_id": self.vector_store_id
            })

        response= self.client.responses.create(
                    model=self.model_name,
                    instructions=model_instr,
                    input= [{
                            "role":"user",
                            "content": input_content,
                    }],
                    
                    )

        return response.output_text