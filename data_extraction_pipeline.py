from pathlib import Path
from src.data_extraction.classes import DataExtractor
from src.utils.config_loader import load_pipeline_config, get_config_value
import mlflow
import logging
import argparse
import os
def setup_logging(config):
    """Setup logging based on configuration."""
    log_config = get_config_value(config, 'logging', {})
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_config.get('file', 'logs/data_extraction.log')),
            logging.StreamHandler()
        ]
    )

def setup_mlflow(config, exp_name):
    """Setup MLflow tracking based on configuration."""
    mlflow_config = get_config_value(config, 'mlflow', {})
    mlflow.set_tracking_uri(mlflow_config.get('tracking_uri', 'sqlite:///mlflow.db'))
    mlflow.set_experiment(exp_name)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ScienceAssist Data Extraction Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--exp_name',
        type=str,
        required=True,
        help='Name of the MLflow experiment (must match data download experiment)'
    )
    
    parser.add_argument(
        '--processed_dataset_name',
        type=str,
        required=True,
        help='Name for the processed dataset'
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
    
    # Load configurations
    config = load_pipeline_config('data_extraction')
    main_config = load_pipeline_config('main')
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Get base data directory from main config
    data_dir = Path(get_config_value(main_config, 'paths.data_dir', 'data'))
    if not os.path.isdir(data_dir):
        raise ValueError(f"Data directory {data_dir} does not exist and must exist to proceed. Did you download the dataset using the data_download_pipeline?")

    exp_name = args.exp_name
    processed_dataset_name = args.processed_dataset_name
    raw_dataset_name = args.raw_dataset_name
    # Paths based on experiment name
    raw_data_dirname = Path(get_config_value(config, 'paths.raw_dataset_location', 'raw_data'))
    processed_data_dirname = Path(get_config_value(config, 'paths.processed_data_location', 'processed_data'))

    raw_data_path_for_exp = raw_data_dirname /Path(raw_dataset_name)

    # Construct full paths and check if they exist
    #------------------------------------------------
    if not os.path.isdir(raw_data_path_for_exp):
        raise ValueError(f"Raw data path {raw_data_path_for_exp} does not exist and must exist to proceed. Did you download the dataset using the data_download_pipeline?")

    processed_data_path = data_dir / exp_name / processed_data_dirname
    if not os.path.isdir(processed_data_path):
        os.mkdir(processed_data_path)

    processed_dataset_path_for_exp = processed_data_path / Path(processed_dataset_name)  

    if os.path.isdir(processed_dataset_path_for_exp):
        raise ValueError(f"Processed dataset path {processed_dataset_path_for_exp} already exists and is not going to be overwritten. Please ensure you want to overwrite it and delete it to proceed.")
 
    
    # Text processing parameters
    max_characters = get_config_value(config, 'text_processing.max_characters', 500)
    new_after_n_chars = get_config_value(config, 'text_processing.new_after_n_chars', 400)
    min_characters = get_config_value(config, 'text_processing.min_characters', 100)
    timeout_seconds = get_config_value(config, 'text_processing.timeout_seconds', 180)
    
    logger.info(f"Starting data extraction pipeline for experiment: {exp_name}")
    logger.info(f"Processed dataset name: {processed_dataset_name}")
    logger.info(f"Raw data path: {raw_data_path_for_exp}")
    logger.info(f"Processed data path: {processed_dataset_path_for_exp}")
    
    # Setup MLflow
    setup_mlflow(config, exp_name)
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            'exp_name': exp_name,
            'processed_dataset_name': processed_dataset_name,
            'max_characters': max_characters,
            'new_after_n_chars': new_after_n_chars,
            'min_characters': min_characters,
            'timeout_seconds': timeout_seconds,
            'raw_data_path': str(raw_data_path_for_exp),
            'processed_data_path': str(processed_dataset_path_for_exp)
        })
        
        try:
            # Create DataExtractor instance
            data_extractor = DataExtractor(
                processed_data_location=processed_dataset_path_for_exp,
                raw_dataset_location=raw_data_path_for_exp,
                max_characters=max_characters,
                new_after_n_chars=new_after_n_chars,
                config=config
            )
            
            # Set timeout
            data_extractor.timeout_seconds = timeout_seconds
            data_extractor.min_characters = min_characters
            
            # Extract and save data
            data_extractor.extract_and_save_data()
            
            # Log metrics (you'll need to modify DataExtractor to return metrics)
            # For now, we'll log basic file existence metrics
            metadata_file = raw_data_path_for_exp / "metadata.csv"
            if metadata_file.exists():
                mlflow.log_metric("metadata_file_exists", 1.0)
            else:
                mlflow.log_metric("metadata_file_exists", 0.0)
       
            
            # Log artifacts
            if get_config_value(config, 'mlflow.log_artifacts', True):
                if metadata_file.exists():
                    mlflow.log_artifact(str(metadata_file))
                
            
            logger.info("Data extraction pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Error in data extraction pipeline: {e}")
            mlflow.log_metric("extraction_success", 0.0)
            raise 