# Model Evaluation Pipeline

## Overview

The Model Evaluation Pipeline is a comprehensive system for evaluating the performance of Retrieval-Augmented Generation (RAG) models. It generates evaluation datasets, compares model outputs against ground truth, and provides detailed metrics for model assessment.

**NOTE**: This pipeline automatically generates intelligent evaluation questions to analyze the RAG pipeline on. While convenient, it is always best to have a few handcrafted examples as well to ensure correctness. Since this pipeline is expected to be used for expert-level analysis of different topics, generation of hand-crafted examples is left to the user. Further, since the pipeline used an LLM to evaluate the answers from RAG, there is some randomness in the generated evaluations. One way around this is to run an ensemble of evaluations for the same evaluation dataset and consider mean scores.


## Workflow Steps

### 1. Dataset Loading and Topic Analysis
- **Input**: Processed dataset from data extraction pipeline
- **Process**: 
  - Loads the processed dataset containing text chunks
  - Performs topic modeling using BERTopic to identify key themes
  - Selects top topics and random topics based on configuration
- **Output**: Topic clusters and associated text chunks

### 2. Evaluation Dataset Generation
- **Input**: Topic clusters and text chunks
- **Process**:
  - Uses LLM to generate questions for each topic using keywords
  - Creates OpenAI vector store with processed chunks
  - Retrieves relevant context for each question using vector similarity
  - Generates "ground truth" answers using LLM with retrieved context
- **Output**: HuggingFace dataset with questions, context, and ground truth answers

### 3. Model Inference and Answer Generation
- **Input**: Evaluation dataset with questions
- **Process**:
  - Initializes ModelInference class with trained embedding model and FAISS index
  - For each question:
    - Performs embedding-based retrieval to get relevant chunks
    - Uses cross-encoder to rerank retrieved chunks
    - Generates answer using LLM with reranked context
- **Output**: Dataset with model-generated answers

### 4. Answer Comparison and Scoring
- **Input**: Ground truth answers and model-generated answers
- **Process**:
  - Formats question-answer pairs for LLM judge
  - Uses specialized judge instructions to evaluate correctness
  - Scores each answer based on missing technical points (0-5 scale)
  - Calculates aggregate metrics (average, min, max scores)
- **Output**: Evaluation metrics and scores

## Key Components

### ModelEvaluation Class
The main class that orchestrates the entire evaluation process.

**Key Methods:**
- `load_dataset()`: Loads and processes the dataset
- `generate_eval_data()`: Creates evaluation dataset with questions and ground truth
- `query_created_model()`: Uses trained RAG model to generate answers
- `compare_answers()`: Compares model answers with ground truth
- `evaluate_model()`: Main orchestration method

### Configuration Management
Uses multiple configuration files:
- **Model Evaluation Config**: Evaluation-specific parameters
- **Dataset Config**: Text processing and dataset parameters  
- **Model Config**: Model creation parameters
- **Inference Config**: Retrieval and inference parameters

### MLflow Integration
Comprehensive experiment tracking including:
- **Parameters**: All configuration parameters
- **Metrics**: Evaluation scores and derived metrics
- **Artifacts**: Evaluation datasets and results
- **Tags**: Experiment organization and filtering

## Configuration Parameters

### Dataset Generation
```yaml
dataset_generation:
  num_top_topics_for_questions: 5      # Top topics to use
  num_random_topics_for_questions: 5   # Random topics to use
  num_questions_per_topic: 2           # Questions per topic
  topic_confidence_threshold: 0.5      # Topic confidence threshold
```
It is **highly** recommended that the product of num_top_topics_for_questions * num_questions_per_topic does not cross 10.
This enables 

### Model Configuration
```yaml
model:
  LLM_model_name: "gpt-4.1"           # LLM for question generation and judging
  LLM_vendor_name: "openai"           # LLM vendor
```

### MLflow Tracking
```yaml
mlflow:
  experiment_name: "science_assist_model_evaluation"
  tracking_uri: "file:./mlruns"
  log_artifacts: true
```

## File Structure

```
src/model_evaluation/
├── README.md                           # This file
├── classes.py                          # Main evaluation class
├── model_evaluation_pipeline.py        # Pipeline orchestration
└── __init__.py

evaluations/
├── {exp_name}/
│   └── {processed_dataset_name}/
│       ├── eval_dataset/               # Generated evaluation dataset
│       └── eval_results/               # Model-generated answers
```


## Evaluation Metrics

### Scoring System
- **Score 5**: Perfect - All key technical points present
- **Score 4**: Good - Missing 1-2 minor technical details  
- **Score 3**: Fair - Missing 3-4 important technical points
- **Score 2**: Poor - Missing 5-6 key technical points
- **Score 1**: Very Poor - Missing 7+ key technical points
- **Score 0**: Incorrect - Completely wrong or missing most points

### Reported Metrics
- **Average Score**: Mean correctness across all questions
- **Maximum Score**: Best performing question
- **Minimum Score**: Worst performing question  
- **Score Range**: Spread of performance (max - min)

## Dependencies

### Core Dependencies
- `sentence_transformers`: For topic modeling and embeddings
- `faiss`: For vector similarity search
- `datasets`: For dataset management
- `mlflow`: For experiment tracking
- `openai`: For LLM interactions

### Internal Dependencies
- `src.model_inference.classes.ModelInference`: RAG model inference
- `src.lib.LLM.classes`: LLM interaction utilities
- `src.utils.config_loader`: Configuration management

## Output Files

### Evaluation Dataset (`eval_dataset/`)
HuggingFace dataset containing:
- `question`: Generated evaluation questions
- `context`: Retrieved context chunks for each question (using topic analysis, not our RAG pipeline)
- `answer`: Ground truth answers generated by LLM

### Evaluation Results (`eval_results/`)
HuggingFace dataset containing:
- `question`: Evaluation questions
- `answer`: Model-generated answers

### MLflow Artifacts
- Configuration files
- Evaluation datasets
- Evaluation results
- Logged metrics and parameters

## Troubleshooting

### Common Issues

1. **Missing Evaluation Dataset**
   - Ensure the data extraction pipeline has been run
   - Check that the experiment and dataset names match

2. **LLM API Errors**
   - Verify OpenAI API key is set in environment
   - Check API rate limits and quotas

3. **Model Loading Errors**
   - Ensure model creation pipeline has been run
   - Verify model paths and file existence

4. **Memory Issues**
   - Reduce `chunks_for_openai_limits` in configuration
   - Use smaller batch sizes for processing

### Debug Mode
Enable detailed logging by setting:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Best Practices

1. **Dataset Quality**: Ensure high-quality processed data for better evaluation
2. **Question Diversity**: Use varied topics and question types
3. **Configuration Tuning**: Adjust parameters based on dataset characteristics
4. **Regular Evaluation**: Run evaluations after model updates
5. **Metric Tracking**: Use MLflow to track performance over different settings

## Future Enhancements

- **Human Evaluation**: Integration with human annotators
- **Advanced Metrics**: BLEU, ROUGE, and other NLP metrics
- **Comparative Analysis**: Side-by-side model comparisons
- **Automated Reporting**: Generate evaluation reports automatically 
