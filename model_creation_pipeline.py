from pathlib import Path
from src.model_creation.classes import ModelCreator
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
            logging.FileHandler(log_config.get('file', 'logs/model_creation.log')),
            logging.StreamHandler()
        ]
    )

def setup_mlflow(config, exp_name):
    """Setup MLflow tracking based on configuration."""
    mlflow_config = get_config_value(config, 'mlflow', {})
    mlflow.set_tracking_uri(mlflow_config.get('tracking_uri', 'http://localhost:5000'))
    mlflow.set_experiment(exp_name)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ScienceAssist Model Creation Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--exp_name',
        type=str,
        required=True,
        help='Name of the MLflow experiment (must match previous pipeline experiments)'
    )
    
    parser.add_argument(
        '--processed_dataset_name',
        type=str,
        required=True,
        help='Name of the processed dataset to use for model creation'
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Name for the created model'
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configurations
    config = load_pipeline_config('model_creation')
    main_config = load_pipeline_config('main')
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Get base data directory from main config
    data_dir = Path(get_config_value(main_config, 'paths.data_dir', 'data'))
    models_dir = Path(get_config_value(main_config, 'paths.models_dir', 'models'))
    
    # Extract parameters from config and arguments
    exp_name = args.exp_name
    processed_dataset_name = args.processed_dataset_name
    model_name = args.model_name
    
    # Paths based on experiment name
    processed_data_dirname = Path(get_config_value(config, 'paths.processed_data_location', 'processed_data'))
    
    # Construct full paths
    processed_data_path_for_exp = data_dir / exp_name / processed_data_dirname/Path(processed_dataset_name)
    if not os.path.isdir(processed_data_path_for_exp):
        raise ValueError(f"Processed data path {processed_data_path_for_exp} does not exist and must exist to proceed. Did you run the data_extraction_pipeline?")
    model_output_path_for_exp = models_dir / Path(exp_name) / Path(model_name)
    
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)
    if not os.path.isdir(models_dir/Path(exp_name)):
        os.mkdir(models_dir/Path(exp_name))
    

    if os.path.isdir(model_output_path_for_exp):
        raise ValueError(f"Model output path {model_output_path_for_exp} already exists and is not going to be overwritten. Please ensure you want to overwrite it and delete it to proceed.")
    
    else:
        os.mkdir(model_output_path_for_exp)
    # Model configuration parameters
    embedding_model_name = get_config_value(config, 'model.embedding_model_name', 'sentence-transformers/multi-qa-MiniLM-L6-dot-v1')
    fine_tune_model = get_config_value(config, 'model.fine_tune_model', False)
    embedding_model_path = model_output_path_for_exp / Path(get_config_value(config, 'model.embedding_model_path', 'embedding_model/'))
    faiss_index_path = model_output_path_for_exp / Path(get_config_value(config, 'model.faiss_index_path', 'faiss_index.idx'))
    
    # Vector database parameters
    index_type = get_config_value(config, 'vector_db.index_type', 'IndexFlatL2')
    normalize_embeddings = get_config_value(config, 'vector_db.normalize_embeddings', True)
    use_gpu = get_config_value(config, 'vector_db.use_gpu', False)
    
    logger.info(f"Starting model creation pipeline for experiment: {exp_name}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Processed dataset path: {processed_data_path_for_exp}")
    logger.info(f"Model output path: {model_output_path_for_exp}")
    
    # Setup MLflow
    setup_mlflow(config, exp_name)
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            'exp_name': exp_name,
            'processed_dataset_name': processed_dataset_name,
            'model_name': model_name,
            'embedding_model_name': embedding_model_name,
            'fine_tune_model': fine_tune_model,
            'index_type': index_type,
            'normalize_embeddings': normalize_embeddings,
            'use_gpu': use_gpu,
            'processed_dataset_name': str(processed_dataset_name),
            'model_name': str(model_name)
        })
        
        try:
            # Create ModelCreator instance
            model_creator = ModelCreator(
                processed_data_location=processed_data_path_for_exp,
                model_output_folder=model_output_path_for_exp,
                embedding_model_name=embedding_model_name
            )
            
            # Create model
            model_creator.create_model()
            
            # Log metrics (you'll need to modify ModelCreator to return metrics)
            # For now, we'll log basic file existence metrics
            '''
            if embedding_model_path.exists():
                mlflow.log_metric("embedding_model_created", 1.0)
            else:
                mlflow.log_metric("embedding_model_created", 0.0)
            
            if faiss_index_path.exists():
                mlflow.log_metric("faiss_index_created", 1.0)
            else:
                mlflow.log_metric("faiss_index_created", 0.0)
            
            # Log artifacts
            if get_config_value(config, 'mlflow.log_artifacts', True):
                if embedding_model_path.exists():
                    mlflow.log_artifact(str(embedding_model_path))
                if faiss_index_path.exists():
                    mlflow.log_artifact(str(faiss_index_path))
            
            logger.info("Model creation pipeline completed successfully")
            '''

        except Exception as e:
            logger.error(f"Error in model creation pipeline: {e}")
            mlflow.log_metric("model_creation_success", 0.0)
            raise 