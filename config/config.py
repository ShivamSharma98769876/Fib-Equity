"""
Configuration settings for the Swing Trade Stock Screener
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class APIConfig:
    """API configuration settings"""
    yahoo_finance_enabled: bool = True   # Enabled by default for reliable data
    kite_connect_enabled: bool = False   # Disabled by default (requires valid API keys)
    upstox_enabled: bool = False
    
    # API Keys (to be set via environment variables)
    kite_api_key: Optional[str] = "14vxzgvarfrfxs5k"
    kite_access_token: Optional[str] = "XZz9Z3Sv7xiDxZ53uvweHYQrB8O3yR7d"
    upstox_api_key: Optional[str] = None
    upstox_access_token: Optional[str] = None

@dataclass
class AnalysisConfig:
    """Analysis configuration settings"""
    # Swing point detection
    lookback_period: int = 20  # Number of bars to look back (increased for better detection)
    swing_minimum_bars: int = 3  # Minimum bars between swing points (increased for better filtering)
    
    # Fibonacci levels
    fibonacci_levels: List[float] = None
    
    # Eligibility criteria
    eligibility_min_level: float = 0.5
    eligibility_max_level: float = 0.618
    
    def __post_init__(self):
        if self.fibonacci_levels is None:
            self.fibonacci_levels = [1.618, 0.786, 0.618, 0.5, 0.382, 0.236]

@dataclass
class UIConfig:
    """UI configuration settings"""
    # Dashboard settings
    refresh_interval: int = 30  # seconds
    max_stocks_display: int = 100
    
    # Table columns
    table_columns: List[str] = None
    
    def __post_init__(self):
        if self.table_columns is None:
            self.table_columns = [
                "Symbol", "Swing Low", "Swing High", 
                "1.618 Level", "0.786 Level", "0.5 Level", 
                "Current Price", "Trend", "Eligibility"
            ]

@dataclass
class MarketConfig:
    """Market hours configuration settings"""
    # Market hours (24-hour format)
    market_start_hour: int = 9
    market_start_minute: int = 15
    market_end_hour: int = 15
    market_end_minute: int = 30
    
    # Market days (0=Monday, 6=Sunday)
    market_days: List[int] = None
    
    def __post_init__(self):
        if self.market_days is None:
            self.market_days = [0, 1, 2, 3, 4]  # Monday to Friday

@dataclass
class DataConfig:
    """Data configuration settings"""
    # Data fetching
    data_interval: str = "15m"  # 15-minute intervals
    data_period: str = "30d"  # Last 30 days (increased for more data points)
    cache_duration: int = 300  # 5 minutes cache
    
    # File paths
    default_stock_list: str = "samples/nifty50.txt"
    export_path: str = "exports"
    
    # Data validation
    min_data_points: int = 20
    max_retries: int = 3

class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.api = APIConfig()
        self.analysis = AnalysisConfig()
        self.ui = UIConfig()
        self.data = DataConfig()
        self.market = MarketConfig()
        
        # Load environment variables
        self._load_env_vars()
    
    def _load_env_vars(self):
        """Load configuration from environment variables"""
        # API Keys - only override if environment variables are set
        if os.getenv("KITE_API_KEY"):
            self.api.kite_api_key = os.getenv("KITE_API_KEY")
        if os.getenv("KITE_ACCESS_TOKEN"):
            self.api.kite_access_token = os.getenv("KITE_ACCESS_TOKEN")
        if os.getenv("UPSTOX_API_KEY"):
            self.api.upstox_api_key = os.getenv("UPSTOX_API_KEY")
        if os.getenv("UPSTOX_ACCESS_TOKEN"):
            self.api.upstox_access_token = os.getenv("UPSTOX_ACCESS_TOKEN")
        
        # Analysis settings
        if os.getenv("LOOKBACK_PERIOD"):
            self.analysis.lookback_period = int(os.getenv("LOOKBACK_PERIOD"))
        
        if os.getenv("REFRESH_INTERVAL"):
            self.ui.refresh_interval = int(os.getenv("REFRESH_INTERVAL"))
    
    def get_stock_symbols(self, file_path: str = None) -> List[str]:
        """Load stock symbols from file"""
        if file_path is None:
            file_path = self.data.default_stock_list
        
        symbols = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    symbol = line.strip()
                    if symbol and not symbol.startswith('#'):
                        symbols.append(symbol)
        except FileNotFoundError:
            print(f"Warning: Stock list file {file_path} not found")
        
        return symbols

# Global configuration instance
config = Config()
