from dotenv import load_dotenv
from pathlib import Path
import logging
from src.model_evaluation.classes import ModelEvaluation

import os
from src.utils.config_loader import load_pipeline_config

def main_model_evaluation_pipeline(exp_name:str, processed_dataset_name:str, model_name:str):

    load_dotenv()

    if not os.path.isdir("evaluations"):
        os.mkdir("evaluations")
    
    config=load_pipeline_config("model_evaluation")

    model_evaluation=ModelEvaluation(exp_name, processed_dataset_name, model_name,config)

    model_evaluation.evaluate_model()



    