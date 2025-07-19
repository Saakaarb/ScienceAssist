# ScienceAssist 🚀

An intelligent LLM-based research assistant that accelerates your research on technical topics by leveraging the latest research from arXiv. ScienceAssist automatically downloads, processes, and creates a knowledge base from scientific papers, enabling you to ask questions and get AI-powered answers based on the most recent research.

## 🎯 Use Cases

- **Research Acceleration**: Quickly understand new research areas
- **Literature Reviews**: Automated paper analysis and summarization
- **Technical Q&A**: Get answers based on the latest research
- **Knowledge Discovery**: Explore connections between research papers
- **Academic Writing**: Generate insights for papers and presentations


## 🌟 Features

- **📥 Automated Data Download**: Downloads relevant PDFs from arXiv based on your search queries
- **📄 Intelligent Text Extraction**: Processes PDFs into searchable text chunks with metadata
- **🧠 Vector Database Creation**: Builds FAISS vector databases for efficient document retrieval
- **🤖 Interactive Q&A**: Ask questions and get AI-powered answers based on the research papers
- **⚙️ Configurable Pipeline**: Easy-to-use YAML configuration for all parameters

## 🏗️ Architecture

ScienceAssist consists of four main pipeline components:

1. **Data Download Pipeline**: Searches arXiv and downloads relevant PDFs using a smart query method
2. **Data Extraction Pipeline**: Extracts and processes text from PDFs based on user set-able parameters
3. **Model Creation Pipeline**: Creates embedding models and vector databases
4. **Model Inference Pipeline**: Provides interactive Q&A capabilities

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Virtual environment (recommended)

### Installation

1. **Clone the repository**:
   ```bash
   git clone git@github.com:Saakaarb/ScienceAssist.git
   cd ScienceAssist
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root with your OpenAI API key:
   ```bash
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

### Configuration and Usage

Suppose you want to create an interactive model which is an expert on the latest research in "quantum machine learning". 
To do this, you need to provide a dir name for:
1. Raw PDF data (eg: raw_dataset_name: "quantum_ml_raw")
2. Processed data (eg: processed_dataset_name: "quantum_ml_processed")
3. Created Model using this data (eg: model_name:"quantum_ml_model")

Edit `pipeline_config.yaml` to customize your research parameters:

```yaml
# Experiment configuration
experiment:
  exp_name: "quantum_computing"  # Name of the MLflow experiment
  user_query: "quantum machine learning"  # Search query for arXiv papers

# Dataset names
datasets:
  raw_dataset_name: "quantum_ml_raw"  # Name for the raw dataset
  processed_dataset_name: "quantum_ml_processed"  # Name for the processed dataset

# Model configuration
model:
  model_name: "quantum_ml_model"  # Name for the created model
```

## 📖 Usage

### Full Pipeline for Model Creation (Recommended)

Run the complete pipeline from data download to interactive Q&A:

```bash
python3 run_pipeline.py
```

This will:
1. Download PDFs from arXiv based on your query, and other queries semantically similar to it
2. Extract and process the text
3. Create embedding models and vector databases
4. Start an interactive Q&A session


### Running the Created Model

Once the model is created, you can run the model using:

```bash
python3 run_pipeline.py --component inference
```
This will read the same pipeline_config.yaml file used during model creation. 
NOTE: If any of the names in this config file are set incorrectly during inference, the code will error out

### Individual Components of Model Creation

You can also run specific pipeline components as needed:

```bash
# Download papers from arXiv
python3 run_pipeline.py --component download

# Extract text from downloaded PDFs
python3 run_pipeline.py --component extract

# Create embedding models and vector databases
python3 run_pipeline.py --component create

# Start interactive Q&A session
python3 run_pipeline.py --component inference
```
This is useful if you would like to run a single part of the pipeline several times with different settings,
to compare performance of different settings of the pipeline configuration in the final model.

Each sub-part of the pipeline has its own configuration file in the `config/` directory:
- `data_download_config.yaml`
- `data_extraction_config.yaml`
- `model_creation_config.yaml`
- `model_inference_config.yaml`

### Custom Configuration

Use a different configuration file:

```bash
python3 run_pipeline.py --config my_custom_config.yaml
```

## 📁 Project Structure

```
ScienceAssist/
├── run_pipeline.py                    # Main pipeline runner
├── pipeline_config.yaml               # Configuration file
├── requirements.txt                   # Python dependencies
├── .env                              # Environment variables (create this)
├── src/
│   ├── data_download/                # Data download pipeline
│   ├── data_extraction/              # Data extraction pipeline
│   ├── model_creation/               # Model creation pipeline
│   ├── model_inference/              # Model inference pipeline
│   └── utils/                        # Utility functions
├── config/                           # Pipeline configurations
├── data/                             # Downloaded and processed data
├── models/                           # Created models and vector databases
└── logs/                             # Pipeline logs
```

## 📊 Example Workflow

1. **Configure your research topic**:
   ```yaml
   experiment:
     exp_name: "machine_learning_research"
     user_query: "deep learning transformer models"
   ```

2. **Run the full pipeline for model creation**:
   ```bash
   python3 run_pipeline.py
   ```

3. **Ask questions interactively**:
   ```
   🤖 ScienceAssist Interactive Q&A Session
   ============================================
   📊 Experiment: machine_learning_research
   🧠 Model: ml_transformer_model
   📚 Documents to retrieve: 5
   🌡️  Temperature: 0.1
   ============================================
   💡 Type your questions below. Type 'quit', 'exit', or 'q' to end the session.
   ============================================

   ❓ Your question: What are the latest developments in transformer architecture?
   ```

Customize embedding models, LLM settings, and retrieval parameters in the respective config files.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


