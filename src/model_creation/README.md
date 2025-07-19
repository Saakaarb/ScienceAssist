# Model Creation Module 🤖

The Model Creation module is the third component of the ScienceAssist pipeline, responsible for transforming processed text chunks into searchable vector embeddings and creating efficient retrieval systems with advanced reranking capabilities. This module implements state-of-the-art embedding generation, cross-encoder models, and vector database creation for RAG (Retrieval-Augmented Generation) systems.

## 🧠 Algorithm Overview

The model creation process follows a sophisticated pipeline designed to create high-quality, searchable vector representations of research content:

### **Stage 1: Dataset Loading**
- **Input**: Processed HuggingFace Dataset from data extraction pipeline
- **Process**: Loads dataset containing text chunks and metadata
- **Output**: Dataset object with all processed text chunks
- **Purpose**: Prepare text data for embedding generation

### **Stage 2: Text Extraction**
- **Input**: Loaded dataset object
- **Process**: Extracts all text chunks from dataset examples
- **Output**: List of text strings for embedding generation
- **Purpose**: Prepare clean text for vectorization

### **Stage 3: Model Initialization**
- **Input**: Embedding model name from configuration
- **Process**: Loads pre-trained sentence transformer model from HuggingFace Hub
- **Output**: Initialized embedding model ready for inference
- **Purpose**: Set up state-of-the-art embedding generation

### **Stage 4: Embedding Generation**
- **Input**: Text chunks and embedding model
- **Process**: 
  1. Generates embeddings for all text chunks
  2. Converts to tensor format for efficiency
  3. Handles batching for large datasets
- **Output**: High-dimensional vector embeddings
- **Purpose**: Create semantic representations of text

### **Stage 5: Vector Database Creation**
- **Input**: Generated embeddings
- **Process**:
  1. Creates FAISS index with L2 distance metric
  2. Adds all embeddings to the index
  3. Optimizes index for similarity search
- **Output**: FAISS vector index for efficient retrieval
- **Purpose**: Enable fast similarity search over embeddings

### **Stage 6: Cross-Encoder Model Creation**
- **Input**: Cross-encoder model name from configuration
- **Process**: Loads pre-trained cross-encoder model for reranking
- **Output**: Cross-encoder model ready for inference
- **Purpose**: Enable advanced reranking for improved retrieval accuracy

### **Stage 7: Model Persistence**
- **Input**: Embedding model, cross-encoder model, and FAISS index
- **Process**: Saves all components to disk in specified locations
- **Output**: Persistent model files for inference
- **Purpose**: Enable model reuse and deployment

## 🔧 Configuration

The model creation module is configured through `config/model_creation_config.yaml`:

### **Input Data Paths**
```yaml
paths:
  processed_data_location: "processed_data"
  processed_dataset_name: "topology_optimization_dataset"
  model_output_folder: "model"
  model_name: "topology_optimization_chatmodel"
```

### **Model Configuration**
```yaml
model:
  embedding_model_name: "sentence-transformers/multi-qa-MiniLM-L6-dot-v1"
  cross_encoder_model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

### **Vector Database Parameters**
```yaml
vector_db:
  index_type: "IndexFlatL2"        # FAISS index type
  normalize_embeddings: true       # Normalize embeddings for cosine similarity
```

### **Performance Settings**
```yaml
performance:
  use_gpu: false                   # Enable GPU acceleration
  num_threads: 4                   # Number of CPU threads
  batch_size_inference: 64         # Batch size for inference
```

## 📊 Key Parameters Explained

### **`embedding_model_name`**
- **Purpose**: Pre-trained sentence transformer model for embedding generation
- **Options**:
  - `"sentence-transformers/multi-qa-MiniLM-L6-dot-v1"`: Optimized for Q&A (recommended)
  - `"sentence-transformers/all-MiniLM-L6-v2"`: General purpose, fast
  - `"sentence-transformers/all-mpnet-base-v2"`: Higher quality, slower
- **Impact**: Determines embedding quality and speed
- **Recommendation**: `"multi-qa-MiniLM-L6-dot-v1"` for RAG systems

### **`cross_encoder_model_name`**
- **Purpose**: Pre-trained cross-encoder model for reranking retrieved documents
- **Options**:
  - `"cross-encoder/ms-marco-MiniLM-L-6-v2"`: Fast and effective (recommended)
  - `"cross-encoder/ms-marco-MiniLM-L-12-v2"`: Higher quality, slower
  - `"cross-encoder/ms-marco-TinyBERT-L-6"`: Very fast, smaller model
- **Impact**: Determines reranking accuracy and inference speed
- **Recommendation**: `"cross-encoder/ms-marco-MiniLM-L-6-v2"` for balanced performance

## 🔍 Model Types Explained

### **Sentence Transformers (Bi-Encoders)**
- **Architecture**: BERT-based models fine-tuned for sentence similarity
- **Output**: 384-768 dimensional vectors
- **Training**: Contrastive learning on sentence pairs
- **Use Case**: First-stage retrieval and semantic similarity

### **Cross-Encoders**
- **Architecture**: BERT-based models that jointly encode query-document pairs
- **Output**: Single relevance score for query-document pair
- **Training**: Supervised learning on relevance labels
- **Use Case**: Second-stage reranking for improved accuracy

### **Two-Stage Retrieval Pipeline**
1. **First Stage (Bi-Encoder)**: Fast retrieval of candidate documents using FAISS
2. **Second Stage (Cross-Encoder)**: Precise reranking of candidates for final selection
3. **Benefits**: Combines speed of bi-encoders with accuracy of cross-encoders




