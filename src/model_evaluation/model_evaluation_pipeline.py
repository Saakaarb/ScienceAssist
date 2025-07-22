from dotenv import load_dotenv
from pathlib import Path
import logging
from src.model_evaluation.classes import ModelEvaluation
from src.utils.config_loader import get_config_value

import os
from src.utils.config_loader import load_pipeline_config
import mlflow


def main_model_evaluation_pipeline(exp_name:str, processed_dataset_name:str, model_name:str):

    load_dotenv()

    if not os.path.isdir("evaluations"):
        os.mkdir("evaluations")
    
    config=load_pipeline_config("model_evaluation")

    # Setup MLflow experiment tracking
    experiment_name = get_config_value(config, "mlflow.experiment_name", "science_assist_model_evaluation")
    tracking_uri = get_config_value(config, "mlflow.tracking_uri", "file:./mlruns")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set experiment
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"evaluation_{exp_name}_{processed_dataset_name}_{model_name}"):
        
        # Log experiment parameters
        mlflow.log_param("experiment_name", exp_name)
        mlflow.log_param("processed_dataset_name", processed_dataset_name)
        mlflow.log_param("model_name", model_name)
        
        # Log pipeline configuration
        mlflow.log_param("pipeline_name", get_config_value(config, "pipeline.name", "model_evaluation_pipeline"))
        mlflow.log_param("pipeline_version", get_config_value(config, "pipeline.version", "1.0.0"))
        
        # Log model configuration
        mlflow.log_param("llm_model_name", get_config_value(config, "model.LLM_model_name", "gpt-4.1"))
        mlflow.log_param("llm_vendor_name", get_config_value(config, "model.LLM_vendor_name", "openai"))
        
        # Log dataset generation parameters
        mlflow.log_param("num_top_topics", get_config_value(config, "dataset_generation.num_top_topics_for_questions", 5))
        mlflow.log_param("num_random_topics", get_config_value(config, "dataset_generation.num_random_topics_for_questions", 5))
        mlflow.log_param("num_questions_per_topic", get_config_value(config, "dataset_generation.num_questions_per_topic", 2))
        mlflow.log_param("topic_confidence_threshold", get_config_value(config, "dataset_generation.topic_confidence_threshold", 0.5))

        model_evaluation=ModelEvaluation(exp_name, processed_dataset_name, model_name,config)

        avg_score, max_score, min_score, dataset_config, model_config, inference_config=model_evaluation.evaluate_model()

        # get critical experiment configuration from config files
        
        # info about text processing
        max_characters=get_config_value(dataset_config, "text_processing.max_characters")
        new_after_n_chars=get_config_value(dataset_config, "text_processing.new_after_n_chars")
        min_characters=get_config_value(dataset_config, "text_processing.min_characters")
        timeout_seconds=get_config_value(dataset_config, "text_processing.timeout_seconds")

        # info about model creation
        embedding_model_name=get_config_value(model_config, "model.embedding_model_name")
        cross_encoder_model_name=get_config_value(model_config, "model.cross_encoder_model_name")

        # inference information
        num_chunks_to_retrieve=get_config_value(inference_config, "retrieval.num_chunks_to_retrieve")
        num_cross_encoder_results=get_config_value(inference_config, "retrieval.num_cross_encoder_results")
        similarity_threshold=get_config_value(inference_config, "retrieval.similarity_threshold")

        # Log dataset processing parameters
        mlflow.log_param("max_characters", max_characters)
        mlflow.log_param("new_after_n_chars", new_after_n_chars)
        mlflow.log_param("min_characters", min_characters)
        mlflow.log_param("timeout_seconds", timeout_seconds)

        # Log model creation parameters
        mlflow.log_param("embedding_model_name", embedding_model_name)
        mlflow.log_param("cross_encoder_model_name", cross_encoder_model_name)

        # Log inference parameters
        mlflow.log_param("num_chunks_to_retrieve", num_chunks_to_retrieve)
        mlflow.log_param("num_cross_encoder_results", num_cross_encoder_results)
        mlflow.log_param("similarity_threshold", similarity_threshold)

        # Log evaluation metrics
        mlflow.log_metric("avg_score", avg_score)
        mlflow.log_metric("max_score", max_score)
        mlflow.log_metric("min_score", min_score)
        
        # Log additional derived metrics
        score_range = max_score - min_score
        mlflow.log_metric("score_range", score_range)
        
        # Log artifacts
        if get_config_value(config, "mlflow.log_artifacts", True):
            # Log evaluation dataset
            if model_evaluation.eval_dataset_path.exists():
                mlflow.log_artifact(str(model_evaluation.eval_dataset_path), "evaluation_dataset")
            
            # Log evaluation results
            if model_evaluation.eval_results_path.exists():
                mlflow.log_artifact(str(model_evaluation.eval_results_path), "evaluation_results")
            
            # Log configuration files
            mlflow.log_artifact("config/model_evaluation_config.yaml", "eval_config")
            mlflow.log_artifact("config/model_inference_config.yaml", "inference_config")
        # Log tags for better organization
        mlflow.set_tag("experiment_type", "model_evaluation")
        mlflow.set_tag("dataset", processed_dataset_name)
        mlflow.set_tag("model", model_name)
        mlflow.set_tag("embedding_model", embedding_model_name)
        mlflow.set_tag("cross_encoder_model", cross_encoder_model_name)
        
        print(f"✅ MLflow experiment tracking completed for {experiment_name}")
        print(f"   - Average Score: {avg_score}")
        print(f"   - Score Range: {min_score} - {max_score}")
        print(f"   - Run ID: {mlflow.active_run().info.run_id}")
        
       