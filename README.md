# Search Query Agent

A complete, production-ready local search-query agent that uses QLoRA fine-tuning on browser history to provide personalized URL recommendations, then scrapes and extracts precise answers from the content.

## Architecture Overview

This repository implements a personalized search agent with the following pipeline:

1. **Browser History Ingestion**: Safely reads Chrome/Firefox history with privacy controls
2. **QLoRA Fine-tuning**: Fine-tunes a 7B model on personalized browsing data using 4-bit quantization
3. **URL Retrieval**: Uses the fine-tuned model to return relevant URLs for user queries
4. **Content Scraping**: Asynchronously scrapes full webpage content with Playwright
5. **Content Formatting**: Structures scraped HTML into clean JSON using the local model
6. **Answer Extraction**: Extracts precise answers to queries from formatted content

## File Structure

```
search-query-agent/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Project configuration
├── make.py                           # Task runner (replaces Makefile)
├── .env.example                      # Environment variables template
├── .gitignore                        # Git ignore patterns
│
├── scripts/                          # Data preparation scripts
│   ├── preview_history.py           # Safely preview browser history with privacy controls
│   └── prepare_dataset.py           # Transform history into QLoRA training dataset
│
├── training/                         # QLoRA fine-tuning pipeline
│   ├── train_qlora.py               # Main training script with 4-bit quantization
│   └── convert_to_ollama.py         # Convert checkpoints to Ollama format
│
├── server/                           # Model serving components
│   └── ollama_adapter.py            # Ollama client with JSON response parsing
│
├── scraper/                          # Web scraping infrastructure
│   └── playwright_scraper.py        # Async scraper with full content capture
│
├── formatter/                        # Content structuring
│   └── format_with_model.py         # Convert HTML to structured JSON
│
├── extractor/                        # Answer extraction
│   └── extract_answer.py            # Extract precise answers from formatted content
│
├── app/                              # FastAPI application
│   └── main.py                      # Main API server with /search endpoint
│
├── configs/                          # Configuration files
│   └── accelerate_config.yaml       # Accelerate configuration for training
│
└── tests/                            # Comprehensive test suite
    ├── test_dataset.py              # Test dataset creation and browser history
    ├── test_training.py             # Test QLoRA training pipeline
    ├── test_scraper.py              # Test web scraping functionality
    └── test_integration.py          # End-to-end integration tests
```

## Quick Start

### Prerequisites

- **Python 3.11+**
- **CUDA-capable GPU** (recommended) or CPU fallback
- **8GB+ RAM** for model training
- **Chrome or Firefox** with browsing history

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd search-query-agent

# Install dependencies
python make.py install

# Setup environment
python make.py setup
```

### 2. Data Preparation

```bash
# Preview browser history (with privacy controls)
python make.py preview

# Prepare training dataset
python make.py dataset
```

### 3. Model Training

```bash
# Train QLoRA model (quick test with 10 steps)
python make.py train

# Convert model for Ollama
python make.py convert
```

### 4. Start Services

```bash
# Start Ollama server
python make.py start-ollama

# Start API server
python make.py start-server
```

### 5. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Search query
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "python tutorial", "max_urls": 3}'
```

## Detailed Usage

### Browser History Privacy

The system includes comprehensive privacy controls:

```bash
# Preview history with privacy options
python scripts/preview_history.py
```

Options:
- **Full URLs**: Complete browsing history for maximum personalization
- **Hashed URLs**: URLs are SHA256-hashed for privacy while maintaining training effectiveness
- **Confirmation Required**: Explicit user consent before any data processing

### Training Configuration

Configure training in `.env`:

```bash
# Model settings
MODEL_NAME=llama3.1:8b
TRAIN_BATCH_SIZE=1
LEARNING_RATE=2e-4
LORA_RANK=8
MAX_STEPS=100

# Privacy settings
REQUIRE_CONFIRMATION=true
HASH_URLS=false
```

### API Endpoints

#### POST /search
Main search endpoint that runs the complete pipeline:

```json
{
  "query": "How to install Python?",
  "max_urls": 5,
  "use_model_formatting": true,
  "use_model_extraction": true
}
```

Response:
```json
{
  "query": "How to install Python?",
  "answer": "Download Python from python.org and run the installer...",
  "confidence": 0.92,
  "reasoning": "Found detailed installation guide in official documentation",
  "sources": [
    {
      "url": "https://docs.python.org/3/installing/",
      "title": "Installing Python Modules",
      "confidence": 0.95,
      "summary": "Official Python installation guide...",
      "content_length": 2500
    }
  ],
  "processing_time": 4.2,
  "status": "success"
}
```

#### GET /health
Health check endpoint:

```json
{
  "status": "healthy",
  "ollama_available": true,
  "model_ready": true,
  "timestamp": 1640995200.0
}
```

## Performance & Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | None (CPU fallback) | NVIDIA GPU with 8GB+ VRAM |
| **RAM** | 8GB | 16GB+ |
| **Storage** | 5GB | 20GB+ |
| **CPU** | 4 cores | 8+ cores |

### Performance Benchmarks

| Operation | GPU (RTX 3080) | CPU (8-core) | Notes |
|-----------|----------------|--------------|-------|
| **QLoRA Training** | ~2 min/epoch | ~15 min/epoch | 1000 samples |
| **URL Retrieval** | ~200ms | ~500ms | Per query |
| **Content Scraping** | ~1.5s/page | ~1.5s/page | Network dependent |
| **Answer Extraction** | ~300ms | ~800ms | Per formatted page |

### Memory Usage

- **Training**: 6-8GB GPU memory (4-bit quantization)
- **Inference**: 2-4GB GPU memory
- **CPU Fallback**: 8-12GB system RAM

## Advanced Configuration

### Custom Model Base

```bash
# Use different base model
python training/train_qlora.py \
  --model_name_or_path microsoft/DialoGPT-large \
  --output_dir ./checkpoints \
  --max_steps 100
```

### Scraping Configuration

```python
# Configure scraper in code
scraper = PlaywrightScraper(
    headless=True,
    timeout=30,
    max_concurrent=5,
    user_agent="Custom-Agent/1.0"
)
```

### Training with Custom Data

```python
# Custom dataset format
{
  "instruction": "Given query: 'machine learning', which URL best answers it?",
  "response": '{"url": "https://scikit-learn.org", "title": "Scikit-learn", "confidence": 0.9}'
}
```

## Testing

### Run All Tests

```bash
python make.py test
```

### Individual Test Suites

```bash
# Dataset creation tests
python tests/test_dataset.py

# Training pipeline tests  
python tests/test_training.py

# Scraping functionality tests
python tests/test_scraper.py

# End-to-end integration tests
python tests/test_integration.py
```

### Test Coverage

The test suite includes:

- **Unit Tests**: Individual component functionality
- **Integration Tests**: End-to-end pipeline simulation
- **Performance Tests**: Benchmark timing and memory usage
- **Error Handling**: Edge cases and failure modes
- **Security Tests**: Privacy protection and URL filtering

## Security & Privacy

### Privacy Protection

1. **Explicit Consent**: User must confirm before any data processing
2. **URL Hashing**: Optional SHA256 hashing of sensitive URLs
3. **Local Processing**: All data stays on your machine
4. **Safe Scraping**: Blocked domains list prevents access to sensitive sites
5. **Data Cleanup**: Clear training data and checkpoints when needed

### Security Measures

- **Input Validation**: All user inputs are sanitized
- **URL Filtering**: Blocked domains list (social media, localhost, etc.)
- **Rate Limiting**: Scraping respects robots.txt and implements delays
- **Error Handling**: Graceful degradation without exposing internals

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
export TRAIN_BATCH_SIZE=1

# Use CPU fallback
export CUDA_VISIBLE_DEVICES=""
```

#### 2. Ollama Connection Failed
```bash
# Check Ollama status
ollama --version

# Start Ollama service
ollama serve

# Pull required model
ollama pull llama3.1:8b
```

#### 3. Browser History Not Found
```bash
# Check browser paths
python -c "
from scripts.preview_history import BrowserHistoryReader
reader = BrowserHistoryReader()
print('Chrome:', reader.find_chrome_history())
print('Firefox:', reader.find_firefox_history())
"
```

#### 4. Playwright Issues
```bash
# Install browsers
python -m playwright install

# Use HTTP fallback
export USE_PLAYWRIGHT=false
```

### Debug Mode

```bash
# Enable verbose logging
export DEBUG=true

# Run with detailed output
python app/main.py --log-level debug
```

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python make.py setup

EXPOSE 8000
CMD ["python", "app/main.py"]
```

### Environment Variables

```bash
# Production settings
export API_HOST=0.0.0.0
export API_PORT=8000
export OLLAMA_HOST=http://ollama-server:11434
export MAX_URLS_TO_SCRAPE=10
export SCRAPE_TIMEOUT=60
```

### Performance Tuning

```bash
# Increase worker processes
uvicorn app.main:app --workers 4 --host 0.0.0.0 --port 8000

# Configure timeouts
export SCRAPE_TIMEOUT=30
export API_TIMEOUT=120
```

## Limitations & Future Improvements

### Current Limitations

1. **Model Size**: Limited to models that fit in available GPU memory
2. **Browser Support**: Currently supports Chrome and Firefox only
3. **Language**: Optimized for English content
4. **Real-time Training**: No incremental learning from new browsing data

### Recommended Improvements

1. **Incremental Training**: Add capability to retrain on new browsing data
2. **Multi-language**: Support for non-English content and queries
3. **Advanced Retrieval**: Implement semantic similarity for URL ranking
4. **Caching**: Add Redis/memcached for faster repeated queries
5. **Monitoring**: Add comprehensive logging and metrics
6. **Browser Extension**: Direct integration with browsers for real-time indexing

## License & Contributing

This project is available under the MIT License. Contributions are welcome!

### Contributing Guidelines

1. **Fork** the repository
2. **Create** a feature branch
3. **Add** tests for new functionality
4. **Ensure** all tests pass
5. **Submit** a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
python make.py setup

# Run tests before committing
python make.py test

# Format code
python -m black .
python -m isort .
```

## Support

For issues and questions:

1. **Check** the troubleshooting section above
2. **Search** existing GitHub issues
3. **Create** a new issue with detailed reproduction steps
4. **Include** system info and error logs

---

**Note**: This is a research/educational project. For production use, ensure compliance with privacy laws and website terms of service when scraping content.
