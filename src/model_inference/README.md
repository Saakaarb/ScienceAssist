# Model Inference Module 🤖

The Model Inference module is the fourth and final component of the ScienceAssist pipeline, responsible for providing interactive Q&A capabilities using retrieval-augmented generation (RAG) with cross-encoder reranking. This module implements a sophisticated two-stage retrieval system that combines the speed of bi-encoders with the accuracy of cross-encoders.

## 🧠 Algorithm Overview

The model inference process follows a sophisticated RAG pipeline designed to provide accurate, context-aware answers to user questions:

### **Stage 1: Model Loading**
- **Input**: Saved models from model creation pipeline
- **Process**: Loads embedding model, cross-encoder model, and FAISS index
- **Output**: Initialized models ready for inference
- **Purpose**: Set up all components for RAG pipeline

### **Stage 2: Question Encoding**
- **Input**: User question
- **Process**: Encodes question using sentence transformer model
- **Output**: Question embedding vector
- **Purpose**: Prepare question for similarity search

### **Stage 3: First-Stage Retrieval (Bi-Encoder)**
- **Input**: Question embedding and FAISS index
- **Process**: 
  1. Performs similarity search using FAISS
  2. Retrieves top-k candidate documents
  3. Extracts text chunks and metadata
- **Output**: Candidate documents with metadata
- **Purpose**: Fast retrieval of relevant candidates

### **Stage 4: Second-Stage Reranking (Cross-Encoder)**
- **Input**: User question and candidate documents
- **Process**:
  1. Computes relevance scores using cross-encoder
  2. Reranks candidates by relevance score
  3. Selects top-n most relevant documents
- **Output**: Reranked and filtered documents
- **Purpose**: Improve retrieval accuracy through precise reranking

### **Stage 5: Context Preparation**
- **Input**: Reranked documents and user question
- **Process**: 
  1. Formats question with retrieved context
  2. Includes metadata (title, page numbers, arXiv links)
  3. Structures prompt for LLM
- **Output**: Formatted question with context
- **Purpose**: Prepare comprehensive context for LLM

### **Stage 6: LLM Answer Generation**
- **Input**: Formatted question with context
- **Process**: 
  1. Sends prompt to configured LLM
  2. Uses RAG instruction file for proper formatting
  3. Generates answer based on provided context
- **Output**: LLM-generated answer with citations
- **Purpose**: Generate accurate, context-aware responses

## 🔧 Configuration

The model inference module is configured through `config/model_inference_config.yaml`:

### **Retrieval Parameters**
```yaml
retrieval:
  num_docs_to_retrieve: 100        # Number of candidates from bi-encoder
  num_cross_encoder_results: 5     # Number of final results after reranking
```

### **LLM Configuration**
```yaml
llm:
  LLM_vendor_name: "openai"        # LLM provider
  LLM_model_name: "gpt-4.1"        # LLM model
  temperature: 0.1                 # Response creativity
  max_tokens: 1000                 # Maximum response length
```

### **API Configuration**
```yaml
api:
  API_key_string: "OPENAI_API_KEY" # Environment variable name
```

### **MLflow Tracking**
```yaml
mlflow:
  experiment_name: "science_assist_model_inference"
  tracking_uri: "file:./mlruns"
  log_artifacts: true
  log_metrics:
    - "question_number"
    - "response_length"
    - "retrieval_time"
    - "reranking_time"
    - "llm_response_time"
```

## 📊 Key Parameters Explained

### **`num_docs_to_retrieve`**
- **Purpose**: Number of candidate documents retrieved by bi-encoder
- **Impact**: 
  - Higher values: More candidates, potentially better coverage
  - Lower values: Faster processing, fewer candidates
- **Recommendation**: 50-200 depending on dataset size

### **`num_cross_encoder_results`**
- **Purpose**: Number of final documents after cross-encoder reranking
- **Impact**: 
  - Higher values: More context, potentially better answers
  - Lower values: Faster LLM processing, more focused context
- **Recommendation**: 3-10 for optimal balance

### **`temperature`**
- **Purpose**: Controls randomness in LLM responses
- **Range**: 0.0 to 1.0
- **Impact**: 
  - Lower (0.0-0.3): More deterministic, factual responses
  - Higher (0.7-1.0): More creative, varied responses
- **Recommendation**: 0.1 for research Q&A

### **Why Cross-Encoder Reranking?**

Traditional bi-encoder retrieval can miss relevant documents due to:
- **Vocabulary mismatch**: Different terms for same concepts
- **Semantic complexity**: Complex relationships not captured by embeddings
- **Context sensitivity**: Importance depends on specific query context

### **Benefits of Two-Stage Retrieval**

- **Speed**: Bi-encoder provides fast initial retrieval
- **Accuracy**: Cross-encoder provides precise reranking
- **Scalability**: Works efficiently on large document collections
- **Quality**: Significantly improves final answer quality

## 📈 Performance Considerations

### **Computational Complexity**
- **Bi-Encoder Retrieval**: O(log n) - FAISS similarity search
- **Cross-Encoder Reranking**: O(k) - Linear with candidate count
- **LLM Generation**: O(1) - Single API call
- **Total**: O(log n + k) - Efficient for large datasets

### **Memory Usage**
- **Model Loading**: ~200-400MB (embedding + cross-encoder models)
- **Dataset Loading**: ~100MB-1GB depending on size
- **FAISS Index**: ~4MB per 1000 vectors
- **Total**: ~500MB-2GB for typical setup

## 🎯 Optimization Tips

### **For Speed**
- Reduce `num_docs_to_retrieve` to 50-100
- Reduce `num_cross_encoder_results` to 3-5
- Use faster cross-encoder models
- Enable GPU acceleration if available

### **For Quality**
- Increase `num_docs_to_retrieve` to 200+
- Increase `num_cross_encoder_results` to 8-10
- Use higher-quality cross-encoder models
- Adjust temperature for response creativity


 