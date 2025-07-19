#!/usr/bin/env python3
"""
ScienceAssist Pipeline Runner
A Python wrapper that can run the full pipeline or individual components.
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path
from typing import Dict, Any
from src.data_download.data_download_pipeline import main_download_pipeline
from src.data_extraction.data_extraction_pipeline import main_extraction_pipeline
from src.model_creation.model_creation_pipeline import main_model_creation_pipeline
from src.model_inference.model_inference_pipeline import main_inference_pipeline

def load_config(config_file: str = "pipeline_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_file}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML file {config_file}: {e}")
        sys.exit(1)


def run_data_download(exp_name: str, user_query: str, raw_dataset_name: str) -> bool:
    """Run data download pipeline independently."""
    print("🔄 Step 1/3: Data Download Pipeline")
    print("-----------------------------------")
    print(f"📥 Downloading PDFs for query: '{user_query}'")
    
    
    try:
        main_download_pipeline(user_query, exp_name, raw_dataset_name)
        print("✅ Data Download Pipeline completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Data Download Pipeline failed with exit code {e.returncode}")
        return False


def run_data_extraction(exp_name: str, raw_dataset_name: str, processed_dataset_name: str) -> bool:
    """Run data extraction pipeline independently."""
    print("🔄 Step 2/3: Data Extraction Pipeline")
    print("-------------------------------------")
    print("📄 Processing PDFs into text chunks")
    
    try:
        main_extraction_pipeline(exp_name, processed_dataset_name, raw_dataset_name)
        print("✅ Data Extraction Pipeline completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Data Extraction Pipeline failed with exit code {e.returncode}")
        return False


def run_model_creation(exp_name: str, processed_dataset_name: str, model_name: str) -> bool:
    """Run model creation pipeline independently."""
    print("🔄 Step 3/3: Model Creation Pipeline")
    print("-----------------------------------")
    print("🤖 Creating embedding model and vector database")
    
    try:
        main_model_creation_pipeline(exp_name, processed_dataset_name, model_name)
        print("✅ Model Creation Pipeline completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Model Creation Pipeline failed with exit code {e.returncode}")
        return False


def run_model_inference(exp_name: str, processed_dataset_name: str, model_name: str) -> bool:
    """Run model inference pipeline independently."""
    print("🔄 Model Inference Pipeline")
    print("----------------------------")
    print("🤖 Starting interactive Q&A session")
    
    try:
        main_inference_pipeline(exp_name, processed_dataset_name, model_name)
        print("✅ Model Inference Pipeline completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Model Inference Pipeline failed with exit code {e.returncode}")
        return False


def run_full_pipeline(config_file: str = "pipeline_config.yaml") -> bool:
    """Run the complete pipeline using YAML config."""
    print("🚀 ScienceAssist End-to-End Pipeline")
    print("======================================")
    
    # Load configuration
    config = load_config(config_file)
    
    # Extract parameters
    exp_name = config['experiment']['exp_name']
    user_query = config['experiment']['user_query']
    raw_dataset_name = config['datasets']['raw_dataset_name']
    processed_dataset_name = config['datasets']['processed_dataset_name']
    model_name = config['model']['model_name']
    
    # Display configuration
    print(f"📋 Configuration:")
    print(f"  • Experiment Name: {exp_name}")
    print(f"  • User Query: {user_query}")
    print(f"  • Raw Dataset Name: {raw_dataset_name}")
    print(f"  • Processed Dataset Name: {processed_dataset_name}")
    print(f"  • Model Name: {model_name}")
    print(f"  • Config File: {config_file}")
    print()
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: No virtual environment detected. Make sure you're in the correct environment.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("❌ Pipeline cancelled")
            return False
    else:
        print("✅ Virtual environment detected")
    
    print()
    print("📋 Pipeline Steps:")
    print("  1. Data Download Pipeline")
    print("  2. Data Extraction Pipeline")
    print("  3. Model Creation Pipeline")
    print()
    
    # Run all three pipelines
    success = True
    
    # Step 1: Data Download
    if not run_data_download(exp_name, user_query, raw_dataset_name):
        success = False
    else:
        print()
    
    # Step 2: Data Extraction
    if success and not run_data_extraction(exp_name, raw_dataset_name, processed_dataset_name):
        success = False
    else:
        print()
    
    # Step 3: Model Creation
    if success and not run_model_creation(exp_name, processed_dataset_name, model_name):
        success = False
    else:
        print()
    
    if success:
        print("🎉 Pipeline completed successfully!")
        print("======================================")
        print("✅ All three pipelines completed without errors")
        print()
        print("📁 Generated Files:")
        print(f"  • Raw data: data/{exp_name}/raw_data/{raw_dataset_name}/")
        print(f"  • Processed data: data/{exp_name}/processed_data/{processed_dataset_name}/")
        print(f"  • Model: models/{exp_name}/{model_name}/")
        print()
        print("🔍 Next Steps:")
        print(f"  • Run inference: python3 run_pipeline.py --component inference --exp_name '{exp_name}' --processed_dataset_name '{processed_dataset_name}' --model_name '{model_name}'")
        print("  • Check MLflow UI for experiment tracking")
        print()
        print("📝 Configuration:")
        print(f"  • Config file: {config_file}")
        print("  • To modify parameters, edit the YAML file and run this script again")
        print()
    else:
        print("❌ Pipeline failed. Check the error messages above.")
    
    return success


def show_usage():
    """Show usage information."""
    print("ScienceAssist Pipeline Runner")
    print("=============================")
    print()
    print("Usage:")
    print("  python3 run_pipeline.py --component <component> [--config <config_file>]")
    print()
    print("Components:")
    print("  --component full      - Run complete pipeline (default)")
    print("  --component download  - Run only data download")
    print("  --component extract   - Run only data extraction")
    print("  --component create    - Run only model creation")
    print("  --component inference - Run only model inference (interactive Q&A)")
    print()
    print("Options:")
    print("  --config <file>       - Path to config file (default: pipeline_config.yaml)")
    print("  --help, -h            - Show this help message")
    print()
    print("Examples:")
    print("  python3 run_pipeline.py                    # Run full pipeline")
    print("  python3 run_pipeline.py --component download  # Run only download")
    print("  python3 run_pipeline.py --component extract   # Run only extraction")
    print("  python3 run_pipeline.py --component create    # Run only model creation")
    print("  python3 run_pipeline.py --component inference # Run only inference")
    print("  python3 run_pipeline.py --config my_config.yaml  # Use custom config")
    print()
    print("For individual components, you can also run the pipeline scripts directly:")
    print("  python3 src/data_download/data_download_pipeline.py --help")
    print("  python3 src/data_extraction/data_extraction_pipeline.py --help")
    print("  python3 src/model_creation/model_creation_pipeline.py --help")
    print("  python3 src/model_inference/model_inference_pipeline.py --help")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="ScienceAssist Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_pipeline.py                    # Run full pipeline
  python3 run_pipeline.py --component download  # Run only download
  python3 run_pipeline.py --component extract   # Run only extraction
  python3 run_pipeline.py --component create    # Run only model creation
  python3 run_pipeline.py --component inference # Run only inference
        """
    )
    
    parser.add_argument(
        "--component", 
        choices=["download", "extract", "create", "inference", "full"], 
        default="full",
        help="Pipeline component to run (default: full)"
    )
    
    parser.add_argument(
        "--config", 
        default="pipeline_config.yaml", 
        help="Path to config file (default: pipeline_config.yaml)"
    )
    
    args = parser.parse_args()
    
    if args.component == "full":
        success = run_full_pipeline(args.config)
        sys.exit(0 if success else 1)
    else:
        # For individual components, load config and run specific component
        config = load_config(args.config)
        
        exp_name = config['experiment']['exp_name']
        user_query = config['experiment']['user_query']
        raw_dataset_name = config['datasets']['raw_dataset_name']
        processed_dataset_name = config['datasets']['processed_dataset_name']
        model_name = config['model']['model_name']
        
        print(f"🚀 Running {args.component} component")
        print(f"📋 Config file: {args.config}")
        print()
        
        success = False
        
        if args.component == "download":
            success = run_data_download(exp_name, user_query, raw_dataset_name)
        elif args.component == "extract":
            success = run_data_extraction(exp_name, raw_dataset_name, processed_dataset_name)
        elif args.component == "create":
            success = run_model_creation(exp_name, processed_dataset_name, model_name)
        elif args.component == "inference":
            success = run_model_inference(exp_name, processed_dataset_name, model_name)
        
        if success:
            print(f"✅ {args.component} component completed successfully")
        else:
            print(f"❌ {args.component} component failed")
        
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 