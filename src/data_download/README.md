# Data Download Module 📥

The Data Download module is the first component of the ScienceAssist pipeline, responsible for intelligently retrieving relevant research papers from arXiv based on user queries. This module implements a sophisticated multi-stage filtering algorithm that combines query expansion with semantic similarity to ensure high-quality paper selection.

## 🧠 Algorithm Overview

The data download process follows a sophisticated 7-stage algorithm designed to maximize relevance while minimizing noise:

### **Stage 1: Query Expansion**
- **Input**: User's original search query
- **Process**: Uses an LLM to generate semantically similar queries
- **Output**: List of expanded queries that capture different aspects and terminology
- **Purpose**: Broaden search scope to capture papers using different terminology

### **Stage 2: Metadata Retrieval**
- **Input**: All queries (original + semantically similar)
- **Process**: Fetches paper metadata from arXiv API for each query
- **Output**: Comprehensive metadata including title, abstract, DOI, authors, publication date
- **Purpose**: Gather candidate papers from arXiv

### **Stage 3: Semantic Filtering**
- **Input**: All retrieved papers and their metadata
- **Process**: 
  1. Encode queries, titles, and abstracts using sentence transformers
  2. Compute cosine similarity between queries and titles
  3. Compute cosine similarity between queries and abstracts
  4. Filter papers where either title OR abstract has similarity above cutoff score
- **Output**: Indices of papers meeting similarity criteria
- **Purpose**: Remove irrelevant papers using semantic understanding

### **Stage 4: Relevance Ranking**
- **Input**: Filtered paper indices
- **Process**:
  1. Average title and abstract similarity scores for each paper
  2. Sort papers by maximum similarity score across all queries
- **Output**: Ranked list of papers by relevance
- **Purpose**: Prioritize most relevant papers

### **Stage 5: Download Limiting**
- **Input**: Ranked paper list
- **Process**: Limit downloads to user-specified number (most relevant first)
- **Output**: Final list of papers to download
- **Purpose**: Control resource usage and focus on top results

### **Stage 6: PDF Download**
- **Input**: Final paper list
- **Process**: Download PDFs from arXiv with error handling
- **Output**: PDF files saved locally
- **Purpose**: Obtain full paper content for processing

### **Stage 7: Metadata Storage**
- **Input**: Downloaded papers and metadata
- **Process**: Save comprehensive metadata to CSV file
- **Output**: `metadata.csv` with all paper information
- **Purpose**: Maintain searchable record of downloaded papers

## 🔧 Configuration

The data download module is configured through `config/data_download_config.yaml`:

### **Query Parameters**
```yaml
query:
  num_docs_check: 100      # Papers to check per query
  num_docs_download: 100   # Maximum papers to download
```

### **Model Parameters**
```yaml
model:
  LLM_model_name: "gpt-4.1"                    # LLM for query expansion
  LLM_vendor_name: "openai"                    # LLM provider
  embedding_model_name: "all-MiniLM-L6-v2"     # Embedding model for similarity
  cutoff_score: 0.5                            # Similarity threshold (-1 to 1)
```

### **API Configuration**
```yaml
api:
  API_key_string: "OPENAI_API_KEY"  # Environment variable name
```

## 📊 Key Parameters Explained

### **`num_docs_check`**
- **Purpose**: Number of papers to retrieve from arXiv for each query
- **Impact**: Higher values = more candidates, but slower processing
- **Recommendation**: 50-200 depending on query specificity

### **`num_docs_download`**
- **Purpose**: Maximum number of papers to actually download
- **Impact**: Controls final dataset size and resource usage
- **Recommendation**: 20-100 depending on research scope

### **`cutoff_score`**
- **Purpose**: Minimum similarity score for paper inclusion
- **Range**: -1 to 1 (cosine similarity)
- **Impact**: 
  - Higher (0.7-0.9): Very strict, high relevance
  - Medium (0.5-0.7): Balanced relevance/coverage
  - Lower (0.3-0.5): Broader coverage, may include less relevant papers
- **Recommendation**: 0.5 for most use cases

### **`embedding_model_name`**
- **Purpose**: HuggingFace Model for computing semantic similarity
- **Options**: 
  - `"all-MiniLM-L6-v2"`: Fast, good performance
  - `"all-mpnet-base-v2"`: Higher quality, slower
  - `"multi-qa-MiniLM-L6-dot-v1"`: Optimized for Q&A
- **Recommendation**: `"all-MiniLM-L6-v2"` for speed/quality balance

## 📈 Performance Considerations

### **Computational Complexity**
- **Query Expansion**: O(1) - Single LLM call
- **Metadata Retrieval**: O(n_queries × num_docs_check) - API calls
- **Semantic Filtering**: O(n_papers × embedding_dim) - Matrix operations
- **PDF Download**: O(n_downloads) - Network operations

### **Memory Usage**
- **Embedding Storage**: ~4MB per 1000 papers (float32 embeddings)
- **Metadata Storage**: ~1MB per 1000 papers (text data)
- **PDF Storage**: Variable (typically 1-10MB per paper)

### **Network Usage**
- **arXiv API**: ~1KB per paper metadata
- **PDF Downloads**: 1-10MB per paper
- **Total**: ~100MB-1GB for typical dataset

## 🎯 Optimization Tips

### **For Speed**
- Reduce `num_docs_check` to 50-100
- Use faster embedding model (`all-MiniLM-L6-v2`)
- Increase `cutoff_score` to 0.7+ for stricter filtering

### **For Quality**
- Increase `num_docs_check` to 200+
- Use higher-quality embedding model (`all-mpnet-base-v2`)
- Lower `cutoff_score` to 0.3-0.5 for broader coverage

### **For Resource Efficiency**
- Set `num_docs_download` based on available storage
- Monitor network usage for large datasets


