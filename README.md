# Swing Trade Stock Screener with Fibonacci Analysis

A Python-based system for identifying swing trade candidates using Fibonacci retracement levels on Nifty 50 stocks with 15-minute chart analysis.

## Features

- **Real-time Stock Scanning**: Scans Nifty 50 stocks using 15-minute OHLCV data
- **Swing Point Detection**: Identifies second last swing high and low points
- **Fibonacci Analysis**: Calculates retracement levels (1.618, 0.786, 0.5)
- **Eligibility Filtering**: Highlights stocks between 0.5 and 0.618 Fibonacci levels
- **Interactive Dashboard**: Streamlit-based user interface with real-time updates
- **Export Functionality**: Export results to CSV format

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd fib-equity
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up configuration:
```bash
cp config/env_example.txt .env
# Edit .env file with your API keys (optional)
```

4. Run the application:
```bash
streamlit run src/ui/main.py
```

## Usage

### Basic Usage

1. **Load Stock List**: Upload a .txt or .csv file with stock symbols, or use the default Nifty 50 list
2. **Configure Parameters**: Set lookback period, Fibonacci levels, and refresh interval
3. **Scan Stocks**: Click "Scan Stocks" to analyze all symbols
4. **View Results**: Review the dashboard table with highlighted eligible candidates
5. **Export Results**: Download filtered results as CSV

### Configuration

Edit `config/config.py` or set environment variables:

- `LOOKBACK_PERIOD`: Number of bars to analyze (default: 30)
- `REFRESH_INTERVAL`: Dashboard refresh interval in seconds (default: 30)
- `DATA_INTERVAL`: Data interval (default: 15m)
- `DATA_PERIOD`: Data period (default: 5d)

## Project Structure

```
fib-equity/
├── src/
│   ├── data/           # Data fetching and processing
│   ├── analysis/       # Swing detection and Fibonacci analysis
│   ├── ui/             # Streamlit dashboard
│   └── utils/          # Utility functions
├── config/             # Configuration files
├── tests/              # Test modules
├── docs/               # Documentation
├── samples/            # Sample data files
└── requirements.txt    # Python dependencies
```

## API Integration

The system supports multiple data sources:

- **Yahoo Finance** (default, no API key required)
- **Kite Connect** (requires API key and access token)
- **Upstox** (requires API key and access token)

## Algorithm Details

### Swing Point Detection

1. **Swing High**: Local maximum with higher close than immediate neighbors
2. **Swing Low**: Local minimum with lower close than immediate neighbors
3. **Second Last Points**: Identifies the second last swing high and low within the lookback period

### Fibonacci Analysis

1. **Uptrend**: Fibonacci retracement from swing low to swing high
2. **Downtrend**: Fibonacci retracement from swing high to swing low
3. **Levels**: Calculates 1.618, 0.786, and 0.5 retracement levels

### Eligibility Criteria

A stock is considered eligible for swing trading if:
- Current price is between 0.5 and 0.618 Fibonacci levels
- Sufficient data points are available for analysis
- Trend direction is clearly identified

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Create an issue in the repository
- Check the documentation in the `docs/` folder
- Review the troubleshooting guide

## Roadmap

- [ ] Multi-timeframe analysis
- [ ] Chart visualization with Fibonacci overlays
- [ ] Backtesting module
- [ ] Alert notifications
- [ ] Performance optimization
