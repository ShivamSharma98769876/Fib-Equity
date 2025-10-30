"""
Main data fetching module that coordinates data retrieval and processing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path

from .api_client import APIManager, create_api_manager
from .stock_loader import StockLoader
from ..utils.cache import DataCache

logger = logging.getLogger(__name__)

class DataFetcher:
    """Main class for fetching and processing stock data"""
    
    def __init__(self, config):
        self.config = config
        self.api_manager = create_api_manager(config)
        self.stock_loader = StockLoader()
        self.cache = DataCache(config.data.cache_duration)
    
    def fetch_stock_data(self, symbol: str, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single stock
        
        Args:
            symbol: Stock symbol
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first
        if use_cache:
            cached_data = self.cache.get(symbol)
            if cached_data is not None:
                logger.info(f"Using cached data for {symbol}")
                return cached_data
        
        # Fetch fresh data
        try:
            data = self.api_manager.fetch_data(
                symbol, 
                self.config.data.data_interval, 
                self.config.data.data_period
            )
            
            if data.empty:
                logger.warning(f"No data available for {symbol}")
                return pd.DataFrame()
            
            # Validate data
            data = self._validate_data(data, symbol)
            
            # Cache the data
            if use_cache:
                self.cache.set(symbol, data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_multiple_stocks(self, symbols: List[str], use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks
        
        Args:
            symbols: List of stock symbols
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        for symbol in symbols:
            try:
                data = self.fetch_stock_data(symbol, use_cache)
                if not data.empty:
                    results[symbol] = data
                else:
                    logger.warning(f"No data available for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        
        return results
    
    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current prices for multiple stocks
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to current prices
        """
        prices = {}
        
        for symbol in symbols:
            try:
                price = self.api_manager.get_current_price(symbol)
                if price > 0:
                    prices[symbol] = price
                else:
                    logger.warning(f"Invalid price for {symbol}: {price}")
            except Exception as e:
                logger.error(f"Error getting price for {symbol}: {e}")
        
        return prices
    
    def fetch_vix_data(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch VIX data for volatility-adjusted swing detection
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with VIX data
        """
        # Check cache first
        if use_cache:
            cached_data = self.cache.get('VIX')
            if cached_data is not None:
                logger.info("Using cached VIX data")
                return cached_data
        
        # Fetch fresh VIX data
        try:
            vix_data = self.api_manager.fetch_data(
                'VIX', 
                self.config.data.data_interval, 
                self.config.data.data_period
            )
            
            if not vix_data.empty:
                # Cache the data
                self.cache.set('VIX', vix_data)
                logger.info(f"Fetched VIX data: {len(vix_data)} records")
                return vix_data
            else:
                logger.warning("No VIX data available")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching VIX data: {e}")
            return pd.DataFrame()
    
    def load_and_fetch_stocks(self, file_path: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load stock list from file and fetch data for all stocks
        
        Args:
            file_path: Path to stock list file (uses default if None)
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        # Load stock symbols
        if file_path is None:
            file_path = self.config.data.default_stock_list
        
        try:
            symbols = self.stock_loader.load_stocks(file_path)
            symbols = self.stock_loader.validate_symbols(symbols)
            
            logger.info(f"Loaded {len(symbols)} symbols from {file_path}")
            
            # Fetch data for all symbols
            return self.fetch_multiple_stocks(symbols)
            
        except Exception as e:
            logger.error(f"Error loading stock list from {file_path}: {e}")
            return {}
    
    def _validate_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validate and clean stock data
        
        Args:
            data: Raw stock data
            symbol: Stock symbol for logging
            
        Returns:
            Cleaned DataFrame
        """
        if data.empty:
            return data
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns for {symbol}: {missing_columns}")
            return pd.DataFrame()
        
        # Remove rows with invalid data
        initial_rows = len(data)
        
        # Remove rows with NaN values in critical columns
        data = data.dropna(subset=required_columns)
        
        # Remove rows with zero or negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            data = data[data[col] > 0]
        
        # Remove rows with negative volume
        data = data[data['volume'] >= 0]
        
        # Ensure high >= low and high >= open, close
        data = data[
            (data['high'] >= data['low']) &
            (data['high'] >= data['open']) &
            (data['high'] >= data['close'])
        ]
        
        # Ensure low <= open, close
        data = data[
            (data['low'] <= data['open']) &
            (data['low'] <= data['close'])
        ]
        
        final_rows = len(data)
        if final_rows < initial_rows:
            logger.info(f"Cleaned {symbol}: {initial_rows} -> {final_rows} rows")
        
        # Check minimum data points
        if len(data) < self.config.data.min_data_points:
            logger.warning(f"Insufficient data points for {symbol}: {len(data)} < {self.config.data.min_data_points}")
            return pd.DataFrame()
        
        return data
    
    def get_latest_data(self, symbol: str, lookback_bars: int = None) -> pd.DataFrame:
        """
        Get latest data for a symbol with specified lookback period
        
        Args:
            symbol: Stock symbol
            lookback_bars: Number of bars to look back (uses config default if None)
            
        Returns:
            DataFrame with latest data
        """
        if lookback_bars is None:
            lookback_bars = self.config.analysis.lookback_period
        
        data = self.fetch_stock_data(symbol)
        
        if data.empty:
            return data
        
        # Get the latest N bars
        return data.tail(lookback_bars).copy()
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return self.cache.get_stats()

# Convenience functions
def fetch_stock_data(symbol: str, config, use_cache: bool = True) -> pd.DataFrame:
    """Convenience function to fetch stock data"""
    fetcher = DataFetcher(config)
    return fetcher.fetch_stock_data(symbol, use_cache)

def load_and_analyze_stocks(file_path: str, config) -> Dict[str, pd.DataFrame]:
    """Convenience function to load and fetch stock data"""
    fetcher = DataFetcher(config)
    return fetcher.load_and_fetch_stocks(file_path)
