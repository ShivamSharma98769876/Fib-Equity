# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection for data fetching

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd fib-equity
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Configuration (Optional)

```bash
# Copy environment template
cp config/env_example.txt .env

# Edit .env file with your API keys (optional)
# Yahoo Finance works without API keys
```

### 5. Run the Application

```bash
# Method 1: Using the run script
python run_app.py

# Method 2: Direct streamlit command
streamlit run src/ui/main.py
```

### 6. Access the Dashboard

Open your browser and go to: http://localhost:8501

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'streamlit'**
   ```bash
   pip install streamlit
   ```

2. **ImportError: No module named 'yfinance'**
   ```bash
   pip install yfinance
   ```

3. **Permission denied on Windows**
   - Run Command Prompt as Administrator
   - Or use `python -m pip install` instead of `pip install`

4. **Port 8501 already in use**
   ```bash
   streamlit run src/ui/main.py --server.port 8502
   ```

### Dependencies Installation

If you encounter issues with specific packages:

```bash
# Install TA-Lib (may require additional system dependencies)
pip install TA-Lib

# Install finta
pip install finta

# Install all requirements
pip install -r requirements.txt --upgrade
```

### System Dependencies (for TA-Lib)

**Windows:**
- Download TA-Lib from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
- Install the appropriate wheel file

**macOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install libta-lib-dev
pip install TA-Lib
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# API Configuration
YAHOO_FINANCE_ENABLED=true
KITE_API_KEY=your_kite_api_key_here
KITE_ACCESS_TOKEN=your_kite_access_token_here

# Analysis Configuration
LOOKBACK_PERIOD=30
REFRESH_INTERVAL=30

# Data Configuration
DATA_INTERVAL=15m
DATA_PERIOD=5d
CACHE_DURATION=300
```

### Stock List Files

Place your stock list files in the `samples/` directory:

- **Text format**: One symbol per line
- **CSV format**: With 'Symbol' column

Example:
```
RELIANCE.NS
TCS.NS
HDFCBANK.NS
```

## Development Setup

### For Developers

1. **Install development dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install pytest black flake8
   ```

2. **Run tests:**
   ```bash
   pytest tests/
   ```

3. **Code formatting:**
   ```bash
   black src/
   flake8 src/
   ```

### IDE Setup

**VS Code:**
- Install Python extension
- Set Python interpreter to virtual environment
- Configure linting with flake8

**PyCharm:**
- Open project directory
- Set Python interpreter to virtual environment
- Configure code style with black

## Docker Setup (Optional)

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "src/ui/main.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

### Docker Commands

```bash
# Build image
docker build -t fib-equity-screener .

# Run container
docker run -p 8501:8501 fib-equity-screener
```

## Performance Optimization

### For Large Stock Lists

1. **Increase cache duration:**
   ```python
   config.data.cache_duration = 600  # 10 minutes
   ```

2. **Reduce lookback period:**
   ```python
   config.analysis.lookback_period = 20
   ```

3. **Use fewer Fibonacci levels:**
   ```python
   config.analysis.fibonacci_levels = [0.5, 0.618]
   ```

## Support

For issues and questions:
- Check the troubleshooting section above
- Review the documentation in `docs/`
- Create an issue in the repository
