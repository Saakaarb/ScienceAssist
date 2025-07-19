from pathlib import Path
from src.model_inference.classes import ModelInference
from src.utils.config_loader import load_pipeline_config, get_config_value
import mlflow
import logging
import argparse
import os
from dotenv import load_dotenv
from src.utils.logging_functions import setup_logging
from src.utils.mlflow_functions import setup_mlflow


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ScienceAssist Model Inference Pipeline - Interactive Q&A',
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
        help='Name of the processed dataset to use for inference'
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Name of the model to use for inference'
    )
    
    return parser.parse_args()

def interactive_qa_session(model_inference, config, exp_name, model_name):
    """Run an interactive Q&A session."""
    logger = logging.getLogger(__name__)
    
    # Get inference parameters from config
    num_docs_to_retrieve = get_config_value(config, 'retrieval.num_docs_to_retrieve', 5)
    temperature = get_config_value(config, 'llm.temperature', 0.1)
    max_tokens = get_config_value(config, 'llm.max_tokens', 1000)
    
    print("\n" + "="*60)
    print("🤖 ScienceAssist Interactive Q&A Session")
    print("="*60)
    print(f"📊 Experiment: {exp_name}")
    print(f"🧠 Model: {model_name}")
    print(f"📚 Documents to retrieve: {num_docs_to_retrieve}")
    print(f"🌡️  Temperature: {temperature}")
    print("="*60)
    print("💡 Type your questions below. Type 'quit', 'exit', or 'q' to end the session.")
    print("="*60 + "\n")
    
    question_count = 0
    
    while True:
        try:
            # Get user question
            question = input("\n❓ Your question: ").strip()
            
            # Check for exit commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using ScienceAssist! Goodbye!")
                break
            
            if not question:
                print("⚠️  Please enter a question.")
                continue
            
            question_count += 1
            print(f"\n🔍 Processing question #{question_count}...")
            
            # Track this interaction in MLflow
            with mlflow.start_run(nested=True):
                # Log question parameters
                mlflow.log_params({
                    'question_number': question_count,
                    'question': question,
                    'num_docs_to_retrieve': num_docs_to_retrieve,
                    'temperature': temperature,
                    'max_tokens': max_tokens
                })
                
                # Get answer from model
                start_time = mlflow.start_run(nested=True)
                reply = model_inference.query_model(question, k=num_docs_to_retrieve)
                
                # Log metrics
                mlflow.log_metric("question_number", question_count)
                mlflow.log_metric("response_length", len(reply) if reply else 0)
                
                # Display answer
                print("\n" + "="*60)
                print("🤖 Answer:")
                print("="*60)
                print(reply)
                print("="*60)
                
                # Automatically continue to next question
                print("\n💡 Ask your next question or type 'quit', 'exit', or 'q' to end the session.")
                    
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit.")
    
    return question_count

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    load_dotenv()
    # Load configurations
    config = load_pipeline_config('model_inference')
    main_config = load_pipeline_config('main')
    
    # Setup logging
    setup_logging(config,pipeline_name='model_inference')
    logger = logging.getLogger(__name__)
    
    # Get base directories from main config
    data_dir = Path(get_config_value(main_config, 'paths.data_dir', 'data'))
    models_dir = Path(get_config_value(main_config, 'paths.models_dir', 'models'))
    
    # Extract parameters from config and arguments
    exp_name = args.exp_name
    processed_dataset_name = args.processed_dataset_name
    model_name = args.model_name
    
    # Construct paths
    processed_data_dirname = Path(get_config_value(config, 'paths.processed_data_location', 'processed_data'))
    model_output_dirname = Path(get_config_value(config, 'paths.model_output_folder', 'model'))
    
    # Verify paths exist
    processed_data_path_for_exp = data_dir / Path(exp_name) / processed_data_dirname / Path(processed_dataset_name)
    model_output_path_for_exp = models_dir / Path(exp_name) / Path(model_name)
    
    if not os.path.isdir(processed_data_path_for_exp):
        raise ValueError(f"Processed data path {processed_data_path_for_exp} does not exist. Did you run the data_extraction_pipeline?")
    
    if not os.path.isdir(model_output_path_for_exp):
        raise ValueError(f"Model path {model_output_path_for_exp} does not exist. Did you run the model_creation_pipeline?")
    
    # LLM configuration
    LLM_vendor_name = get_config_value(config, 'llm.LLM_vendor_name', 'openai')
    LLM_model_name = get_config_value(config, 'llm.LLM_model_name', 'gpt-4.1')
    API_key_string = get_config_value(config, 'api.API_key_string', 'OPENAI_API_KEY')
    LLM_instr_filename = Path(get_config_value(config, 'paths.LLM_instr_filename', 'src/lib/LLM/LLM_instr_files/RAG_instr.txt'))
    
    logger.info(f"Starting model inference pipeline for experiment: {exp_name}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Processed dataset path: {processed_data_path_for_exp}")
    logger.info(f"Model path: {model_output_path_for_exp}")
    
    # Setup MLflow
    
        
    try:
        # Create ModelInference instance
        model_inference = ModelInference(
            model_output_folder=model_output_path_for_exp,
            model_name=model_name,
            path_to_dataset=processed_data_path_for_exp,
            LLM_vendor_name=LLM_vendor_name,
            LLM_model_name=LLM_model_name,
            LLM_instr_filename=LLM_instr_filename,
            API_key_string=API_key_string
        )
        
        # Start interactive Q&A session
        total_questions = interactive_qa_session(model_inference, config, exp_name, model_name)
        
        
        logger.info(f"Model inference pipeline completed successfully. Total questions: {total_questions}")
        
    except Exception as e:
        logger.error(f"Error in model inference pipeline: {e}")
        
        raise 