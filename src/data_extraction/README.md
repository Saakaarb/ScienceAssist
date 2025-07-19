# Data Extraction Module 📄

The Data Extraction module is the second component of the ScienceAssist pipeline, responsible for transforming raw PDF papers into structured, searchable text chunks. This module implements intelligent text extraction, cleaning, and chunking algorithms to create high-quality datasets for downstream processing.

## 🧠 Algorithm Overview

The data extraction process follows a sophisticated multi-stage pipeline designed to maximize text quality while maintaining semantic coherence:

### **Stage 1: Metadata Loading**
- **Input**: Raw dataset metadata from data download pipeline
- **Process**: Loads `metadata.csv` and filters out failed downloads
- **Output**: Clean metadata DataFrame and list of PDF filenames
- **Purpose**: Prepare for systematic PDF processing

### **Stage 2: PDF Element Extraction**
- **Input**: Individual PDF files from raw dataset
- **Process**: Uses `unstructured` library to extract text elements with structural information
- **Output**: List of text elements with metadata (page numbers, titles, etc.)
- **Purpose**: Preserve document structure for intelligent chunking

### **Stage 3: Intelligent Chunking**
- **Input**: Extracted PDF elements
- **Process**: 
  1. Chunks elements by title boundaries
  2. Respects character limits (`max_characters`, `new_after_n_chars`)
  3. Maintains semantic coherence across chunks
- **Output**: Structured text chunks with metadata
- **Purpose**: Create manageable, semantically meaningful text units

### **Stage 4: Text Cleaning**
- **Input**: Raw text chunks
- **Process**: Comprehensive regex-based cleaning:
  - Remove LaTeX math expressions
  - Remove figure/table captions
  - Remove references sections
  - Clean formatting artifacts
  - Normalize text structure
- **Output**: Clean, normalized text chunks
- **Purpose**: Improve text quality for downstream processing

### **Stage 5: Quality Filtering**
- **Input**: Cleaned text chunks
- **Process**: Filter chunks below minimum character threshold
- **Output**: High-quality chunks meeting size requirements
- **Purpose**: Remove noise and ensure chunk quality

### **Stage 6: Dataset Assembly**
- **Input**: Filtered chunks and metadata
- **Process**: Combine text with paper metadata and page information
- **Output**: HuggingFace Dataset with structured format
- **Purpose**: Create searchable, metadata-rich dataset

### **Stage 7: Dataset Persistence**
- **Input**: Assembled dataset
- **Process**: Save to disk in HuggingFace format
- **Output**: Persistent dataset for model creation
- **Purpose**: Enable efficient dataset loading and sharing

## 🔧 Configuration

The data extraction module is configured through `config/data_extraction_config.yaml`:

### **Text Processing Parameters**
```yaml
text_processing:
  max_characters: 5000      # Maximum characters per chunk
  new_after_n_chars: 4000   # Start new chunk after this many characters
  min_characters: 100       # Minimum characters for valid chunk- chunks smaller than this are skipped
  timeout_seconds: 180      # PDF processing timeout- if processing takes longer the PDF is skipped
```

### **Text Cleaning Parameters**
```yaml
cleaning:
  remove_math: true         # Remove LaTeX math expressions
  remove_brackets: true     # Remove content in brackets
  remove_figures: true      # Remove figure/table captions
  remove_references: true   # Remove references section
  convert_to_lowercase: true # Convert text to lowercase
  remove_non_ascii: true    # Remove non-ASCII characters
```

### **Performance Settings**
```yaml
performance:
  parallel_processing: false # Enable parallel PDF processing
  max_workers: 4            # Number of parallel workers
  chunk_size: 1000          # Process chunks in batches
```

## 📊 Key Parameters Explained

### **`max_characters`**
- **Purpose**: Maximum size of each text chunk
- **Impact**: 
  - Smaller chunks: Better for specific queries, more chunks
  - Larger chunks: Better context, fewer chunks
- **Recommendation**: 3000-5000 for research papers

### **`new_after_n_chars`**
- **Purpose**: Character threshold for starting new chunks
- **Impact**: Controls chunk overlap and context preservation
- **Recommendation**: 80-90% of `max_characters`

### **`min_characters`**
- **Purpose**: Minimum size for valid chunks
- **Impact**: Filters out noise and incomplete sentences
- **Recommendation**: 100-200 characters

### **`timeout_seconds`**
- **Purpose**: Maximum time to process each PDF
- **Impact**: Prevents hanging on problematic PDFs
- **Recommendation**: 180-300 seconds depending on PDF complexity

## 🎯 Optimization Tips

### **For Speed**
- Reduce `max_characters` to 2000-3000
- Increase `timeout_seconds` for complex PDFs

### **For Quality**
- Increase `max_characters` to 5000-8000
- Lower `new_after_n_chars` for more overlap
- Increase `min_characters` to 200-300
- Adjust cleaning parameters for specific content types

### **For Memory Efficiency**
- Process PDFs sequentially (disable parallel)
- Use smaller chunk sizes
- Monitor memory usage during processing
- Clean up temporary files

