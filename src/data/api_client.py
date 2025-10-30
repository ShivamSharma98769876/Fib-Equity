"""
API client module for fetching stock data from various sources
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseAPIClient(ABC):
    """Abstract base class for API clients"""
    
    @abstractmethod
    def fetch_data(self, symbol: str, interval: str = "15m", period: str = "5d") -> pd.DataFrame:
        """Fetch OHLCV data for a symbol"""
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        pass

class YahooFinanceClient(BaseAPIClient):
    """Yahoo Finance API client"""
    
    def __init__(self):
        self.name = "Yahoo Finance"
        self.rate_limit_delay = 0.1  # 100ms delay between requests
    
    def fetch_data(self, symbol: str, interval: str = "15m", period: str = "5d") -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance
        
        Args:
            symbol: Stock symbol
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d)
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Add rate limiting
            time.sleep(self.rate_limit_delay)
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Ensure proper column names
            data.columns = data.columns.str.lower()
            
            # Add symbol column
            data['symbol'] = symbol
            
            # Reset index to make datetime a column
            data = data.reset_index()
            
            logger.info(f"Fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            time.sleep(self.rate_limit_delay)
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Try different price fields
            price_fields = ['currentPrice', 'regularMarketPrice', 'previousClose']
            
            for field in price_fields:
                if field in info and info[field] is not None:
                    return float(info[field])
            
            # Fallback: get latest close price from history
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return float(data['Close'].iloc[-1])
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return 0.0

class KiteConnectClient(BaseAPIClient):
    """Kite Connect API client for real-time data fetching"""
    
    def __init__(self, api_key: str, access_token: str):
        self.name = "Kite Connect"
        self.api_key = api_key
        self.access_token = access_token
        
        try:
            from kiteconnect import KiteConnect
            self.kite = KiteConnect(api_key=api_key)
            self.kite.set_access_token(access_token)
            logger.info("Kite Connect client initialized successfully")
        except ImportError:
            logger.error("KiteConnect library not installed. Install with: pip install kiteconnect")
            self.kite = None
        except Exception as e:
            logger.error(f"Error initializing Kite Connect: {e}")
            self.kite = None
    
    def _convert_interval(self, interval: str) -> str:
        """Convert standard interval to Kite Connect format"""
        interval_map = {
            "1m": "minute",
            "5m": "5minute", 
            "15m": "15minute",
            "30m": "30minute",
            "1h": "60minute",
            "1d": "day"
        }
        return interval_map.get(interval, "15minute")
    
    def _convert_period_to_days(self, period: str) -> int:
        """Convert period string to number of days"""
        period_map = {
            "1d": 1,
            "5d": 5,
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825
        }
        return period_map.get(period, 5)
    
    def _get_instrument_token(self, symbol: str) -> str:
        """Get instrument token for symbol"""
        try:
            # Remove .NS suffix if present
            clean_symbol = symbol.replace('.NS', '')
            
            # Get instruments list
            instruments = self.kite.instruments("NSE")
            
            # Find the instrument
            for instrument in instruments:
                if instrument['tradingsymbol'] == clean_symbol:
                    return str(instrument['instrument_token'])
            
            logger.warning(f"Instrument not found for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting instrument token for {symbol}: {e}")
            return None
    
    def fetch_data(self, symbol: str, interval: str = "15m", period: str = "5d") -> pd.DataFrame:
        """
        Fetch OHLCV data from Kite Connect
        
        Args:
            symbol: Stock symbol
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d)
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y)
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.kite:
            logger.error("Kite Connect not initialized")
            return pd.DataFrame()
        
        try:
            # Get instrument token
            instrument_token = self._get_instrument_token(symbol)
            if not instrument_token:
                return pd.DataFrame()
            
            # Convert parameters
            kite_interval = self._convert_interval(interval)
            days = self._convert_period_to_days(period)
            
            # Calculate date range
            from datetime import datetime, timedelta
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            # Fetch historical data
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=kite_interval
            )
            
            if not data:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Rename columns to lowercase
            df.columns = df.columns.str.lower()
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Reset index to make datetime a column
            df = df.reset_index()
            
            logger.info(f"Fetched {len(df)} records for {symbol} from Kite Connect")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        if not self.kite:
            logger.error("Kite Connect not initialized")
            return 0.0
        
        try:
            # Get instrument token
            instrument_token = self._get_instrument_token(symbol)
            if not instrument_token:
                return 0.0
            
            # Get LTP (Last Traded Price)
            ltp_data = self.kite.ltp([instrument_token])
            
            if instrument_token in ltp_data:
                return float(ltp_data[instrument_token]['last_price'])
            else:
                logger.warning(f"No LTP data for {symbol}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return 0.0
    
    def get_quote(self, symbol: str) -> dict:
        """Get full quote data for a symbol"""
        if not self.kite:
            logger.error("Kite Connect not initialized")
            return {}
        
        try:
            # Get instrument token
            instrument_token = self._get_instrument_token(symbol)
            if not instrument_token:
                return {}
            
            # Get quote
            quote_data = self.kite.quote([instrument_token])
            
            if instrument_token in quote_data:
                return quote_data[instrument_token]
            else:
                logger.warning(f"No quote data for {symbol}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return {}
    
    def get_multiple_quotes(self, symbols: List[str]) -> dict:
        """Get quotes for multiple symbols"""
        if not self.kite:
            logger.error("Kite Connect not initialized")
            return {}
        
        try:
            # Get instrument tokens
            instrument_tokens = []
            symbol_map = {}
            
            for symbol in symbols:
                token = self._get_instrument_token(symbol)
                if token:
                    instrument_tokens.append(token)
                    symbol_map[token] = symbol
            
            if not instrument_tokens:
                return {}
            
            # Get quotes
            quotes = self.kite.quote(instrument_tokens)
            
            # Map back to symbols
            result = {}
            for token, quote_data in quotes.items():
                if token in symbol_map:
                    result[symbol_map[token]] = quote_data
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting multiple quotes: {e}")
            return {}

class UpstoxClient(BaseAPIClient):
    """Upstox API client (placeholder for future implementation)"""
    
    def __init__(self, api_key: str, access_token: str):
        self.name = "Upstox"
        self.api_key = api_key
        self.access_token = access_token
        # Note: Actual Upstox implementation would go here
    
    def fetch_data(self, symbol: str, interval: str = "15m", period: str = "5d") -> pd.DataFrame:
        """Placeholder for Upstox implementation"""
        logger.warning("Upstox client not implemented yet")
        return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> float:
        """Placeholder for Upstox implementation"""
        logger.warning("Upstox client not implemented yet")
        return 0.0

class APIManager:
    """Manages multiple API clients and provides unified interface"""
    
    def __init__(self, config):
        self.config = config
        self.clients = {}
        self._setup_clients()
    
    def _setup_clients(self):
        """Setup available API clients based on configuration"""
        
        # Yahoo Finance (always available)
        if self.config.api.yahoo_finance_enabled:
            self.clients['yahoo'] = YahooFinanceClient()
        
        # Kite Connect
        if (self.config.api.kite_connect_enabled and 
            self.config.api.kite_api_key and 
            self.config.api.kite_access_token):
            self.clients['kite'] = KiteConnectClient(
                self.config.api.kite_api_key,
                self.config.api.kite_access_token
            )
        
        # Upstox
        if (self.config.api.upstox_enabled and 
            self.config.api.upstox_api_key and 
            self.config.api.upstox_access_token):
            self.clients['upstox'] = UpstoxClient(
                self.config.api.upstox_api_key,
                self.config.api.upstox_access_token
            )
    
    def get_primary_client(self) -> BaseAPIClient:
        """Get the primary API client"""
        # Priority: Yahoo Finance > Kite Connect > Upstox
        for client_name in ['yahoo', 'kite', 'upstox']:
            if client_name in self.clients:
                return self.clients[client_name]
        
        raise RuntimeError("No API clients available")
    
    def fetch_data(self, symbol: str, interval: str = None, period: str = None) -> pd.DataFrame:
        """Fetch data using the primary client"""
        if interval is None:
            interval = self.config.data.data_interval
        if period is None:
            period = self.config.data.data_period
        
        client = self.get_primary_client()
        return client.fetch_data(symbol, interval, period)
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price using the primary client"""
        client = self.get_primary_client()
        return client.get_current_price(symbol)
    
    def fetch_multiple_symbols(self, symbols: List[str], interval: str = None, period: str = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols
        
        Args:
            symbols: List of stock symbols
            interval: Data interval
            period: Data period
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        for symbol in symbols:
            try:
                data = self.fetch_data(symbol, interval, period)
                if not data.empty:
                    results[symbol] = data
                else:
                    logger.warning(f"No data available for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        
        return results

# Convenience functions
def create_api_manager(config) -> APIManager:
    """Create an API manager instance"""
    return APIManager(config)

def fetch_stock_data(symbol: str, config, interval: str = None, period: str = None) -> pd.DataFrame:
    """Convenience function to fetch stock data"""
    manager = create_api_manager(config)
    return manager.fetch_data(symbol, interval, period)

def get_stock_price(symbol: str, config) -> float:
    """Convenience function to get stock price"""
    manager = create_api_manager(config)
    return manager.get_current_price(symbol)
