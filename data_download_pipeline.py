from src.data_download.classes import DataDownloader
from src.utils.config_loader import load_pipeline_config, get_config_value
import yaml
from pathlib import Path
from dotenv import load_dotenv
import mlflow
import logging
import argparse
import os
from src.utils.logging_functions import setup_logging
from src.utils.mlflow_functions import setup_mlflow




def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ScienceAssist Data Download Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--exp_name',
        type=str,
        required=True,
        help='Name of the MLflow experiment'
    )
    
    parser.add_argument(
        '--user_query',
        type=str,
        required=True,
        help='Search query for arXiv papers'
    )
    
    parser.add_argument(
        '--raw_dataset_name',
        type=str,
        required=True,
        help='Name of the raw dataset to use for model creation'
    )

    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
# Load environment variables
    load_dotenv()

    # Load configuration
    config = load_pipeline_config('data_download')
    
    main_config = load_pipeline_config('main')

    data_dir = Path(get_config_value(main_config, 'paths.data_dir', 'data'))
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    # Setup logging
    setup_logging(config,pipeline_name='data_download')
    logger = logging.getLogger(__name__)
    
    
    
    # Extract parameters from config
    user_query = str(args.user_query)  # Use command line argument instead of config
    exp_name = str(args.exp_name)
    raw_dataset_name = str(args.raw_dataset_name)

    print(f"User query: {user_query}")
    print(f"Experiment name: {exp_name}")
    print(f"Raw dataset name: {raw_dataset_name}")

    if not os.path.isdir(data_dir /Path(exp_name)):
        os.mkdir(data_dir /Path(exp_name))

    num_docs_check = get_config_value(config, 'query.num_docs_check', 100)
    num_docs_download = get_config_value(config, 'query.num_docs_download', 100)

    raw_data_dirname = Path(get_config_value(config, 'paths.downloads_dir', 'raw_data'))
    metadata_filename = Path(get_config_value(config, 'paths.metadata_filename', 'metadata.csv'))
    
    raw_data_dir_path = data_dir /Path(exp_name) /raw_data_dirname

    if not os.path.isdir(raw_data_dir_path):
        os.mkdir(raw_data_dir_path)

    raw_data_path_for_exp = raw_data_dir_path /Path(raw_dataset_name)

    if os.path.isdir(raw_data_path_for_exp):
        raise ValueError(f"Raw data path {raw_data_path_for_exp} already exists and is not going to be overwritten. Please ensure you want to overwrite it and delete it to proceed.")

    else:   
        os.mkdir(raw_data_path_for_exp)

    metadata_path = raw_data_path_for_exp /metadata_filename

    # Model parameters
    LLM_model_name = get_config_value(config, 'model.LLM_model_name', 'gpt-4.1')
    LLM_vendor_name = get_config_value(config, 'model.LLM_vendor_name', 'openai')
    embedding_model_name = get_config_value(config, 'model.embedding_model_name', 'all-MiniLM-L6-v2')
    cutoff_score = get_config_value(config, 'model.cutoff_score', 0.5)
    
    # API configuration
    API_key_string = get_config_value(config, 'api.API_key_string', 'OPENAI_API_KEY')
    
    logger.info(f"Starting data download pipeline with query: {user_query}")
    logger.info(f"Experiment name: {args.exp_name}")
    
    # Setup MLflow with custom experiment name
    setup_mlflow(config, args.exp_name)
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            'user_query': user_query,
            'exp_name': args.exp_name,
            'num_docs_check': num_docs_check,
            'num_docs_download': num_docs_download,
            'LLM_model_name': LLM_model_name,
            'embedding_model_name': embedding_model_name,
            'cutoff_score': cutoff_score,
            'raw_dataset_name': raw_dataset_name
        })
        
        try:
            # Create DataDownloader instance
            datadownloader = DataDownloader(
                downloads_dir=raw_data_path_for_exp,
                num_docs_check=num_docs_check,
                num_docs_download=num_docs_download,
                metadata_filename=metadata_path,
                LLM_model_name=LLM_model_name,
                LLM_vendor_name=LLM_vendor_name,
                embedding_model_name=embedding_model_name,
                API_key_string=API_key_string,
                cutoff_score=cutoff_score
            )
            
            
            # Download data
            result = datadownloader.download_pdf_data(user_query)
            
            # Log metrics (you'll need to modify DataDownloader to return metrics)
            mlflow.log_metric("papers_downloaded", len(result) if result else 0)
            mlflow.log_metric("download_success_rate", 1.0 if result else 0.0)
            
            # Log artifacts
            #if get_config_value(config, 'mlflow.log_artifacts', True):
            #    mlflow.log_artifact(metadata_filename)
            
            logger.info("Data download pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Error in data download pipeline: {e}")
            mlflow.log_metric("download_success_rate", 0.0)
            raise 