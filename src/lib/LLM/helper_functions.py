from src.lib.LLM.classes import OpenAI_model

def create_LLM_instance(api_key_string,reference_file_paths,model_name,vendor_name=None):

    if vendor_name is None:
        print("WARNING: No vendor name provided. Defaulting to OpenAI")
        vendor_name="openai"

    if vendor_name=="openai":
        return OpenAI_model(api_key_string, reference_file_paths, model_name)
    else:
        raise ValueError(f"Vendor {vendor_name} currently not supported")