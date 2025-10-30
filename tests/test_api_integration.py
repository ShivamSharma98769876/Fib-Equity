"""
Integration tests for API data fetching
"""

import unittest
import sys
import os
import pandas as pd
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.api_client import YahooFinanceClient, KiteConnectClient, APIManager
from data.data_fetcher import DataFetcher
from config.config import config

class TestAPIIntegration(unittest.TestCase):
    """Test cases for API integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
        self.test_config = config
    
    def test_yahoo_finance_client_initialization(self):
        """Test Yahoo Finance client initialization"""
        client = YahooFinanceClient()
        self.assertEqual(client.name, "Yahoo Finance")
        self.assertIsInstance(client.rate_limit_delay, float)
    
    def test_kite_connect_client_initialization(self):
        """Test Kite Connect client initialization"""
        # Test with mock credentials
        api_key = "test_api_key"
        access_token = "test_access_token"
        
        with patch('src.data.api_client.KiteConnect') as mock_kite:
            mock_kite_instance = MagicMock()
            mock_kite.return_value = mock_kite_instance
            
            client = KiteConnectClient(api_key, access_token)
            
            self.assertEqual(client.name, "Kite Connect")
            self.assertEqual(client.api_key, api_key)
            self.assertEqual(client.access_token, access_token)
            mock_kite.assert_called_once_with(api_key=api_key)
            mock_kite_instance.set_access_token.assert_called_once_with(access_token)
    
    def test_api_manager_initialization(self):
        """Test API manager initialization"""
        manager = APIManager(self.test_config)
        
        # Check that clients are set up based on config
        self.assertIsInstance(manager.clients, dict)
    
    def test_interval_conversion(self):
        """Test interval conversion for Kite Connect"""
        api_key = "test_api_key"
        access_token = "test_access_token"
        
        with patch('src.data.api_client.KiteConnect'):
            client = KiteConnectClient(api_key, access_token)
            
            # Test interval conversions
            self.assertEqual(client._convert_interval("15m"), "15minute")
            self.assertEqual(client._convert_interval("1h"), "60minute")
            self.assertEqual(client._convert_interval("1d"), "day")
            self.assertEqual(client._convert_interval("unknown"), "15minute")
    
    def test_period_conversion(self):
        """Test period conversion for Kite Connect"""
        api_key = "test_api_key"
        access_token = "test_access_token"
        
        with patch('src.data.api_client.KiteConnect'):
            client = KiteConnectClient(api_key, access_token)
            
            # Test period conversions
            self.assertEqual(client._convert_period_to_days("5d"), 5)
            self.assertEqual(client._convert_period_to_days("1mo"), 30)
            self.assertEqual(client._convert_period_to_days("1y"), 365)
            self.assertEqual(client._convert_period_to_days("unknown"), 5)
    
    @patch('yfinance.Ticker')
    def test_yahoo_finance_data_fetching(self, mock_ticker):
        """Test Yahoo Finance data fetching with mock"""
        # Setup mock
        mock_ticker_instance = MagicMock()
        mock_ticker.return_value = mock_ticker_instance
        
        # Mock historical data
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [104, 105, 106],
            'Volume': [1000, 1100, 1200]
        })
        mock_ticker_instance.history.return_value = mock_data
        
        # Test data fetching
        client = YahooFinanceClient()
        result = client.fetch_data("RELIANCE.NS", "15m", "5d")
        
        # Verify results
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)
        self.assertIn('symbol', result.columns)
        mock_ticker.assert_called_once_with("RELIANCE.NS")
        mock_ticker_instance.history.assert_called_once_with(period="5d", interval="15m")
    
    @patch('yfinance.Ticker')
    def test_yahoo_finance_current_price(self, mock_ticker):
        """Test Yahoo Finance current price fetching with mock"""
        # Setup mock
        mock_ticker_instance = MagicMock()
        mock_ticker.return_value = mock_ticker_instance
        
        # Mock info data
        mock_ticker_instance.info = {'currentPrice': 105.5}
        
        # Test price fetching
        client = YahooFinanceClient()
        price = client.get_current_price("RELIANCE.NS")
        
        # Verify results
        self.assertEqual(price, 105.5)
        mock_ticker.assert_called_once_with("RELIANCE.NS")
    
    @patch('src.data.api_client.KiteConnect')
    def test_kite_connect_data_fetching(self, mock_kite_class):
        """Test Kite Connect data fetching with mock"""
        # Setup mock
        mock_kite_instance = MagicMock()
        mock_kite_class.return_value = mock_kite_instance
        
        # Mock instruments and historical data
        mock_instruments = [
            {'tradingsymbol': 'RELIANCE', 'instrument_token': '12345'},
            {'tradingsymbol': 'TCS', 'instrument_token': '67890'}
        ]
        mock_kite_instance.instruments.return_value = mock_instruments
        
        mock_historical_data = [
            {'date': '2024-01-01', 'open': 100, 'high': 105, 'low': 99, 'close': 104, 'volume': 1000},
            {'date': '2024-01-02', 'open': 104, 'high': 108, 'low': 103, 'close': 107, 'volume': 1100}
        ]
        mock_kite_instance.historical_data.return_value = mock_historical_data
        
        # Test data fetching
        client = KiteConnectClient("test_api_key", "test_access_token")
        result = client.fetch_data("RELIANCE.NS", "15m", "5d")
        
        # Verify results
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)
        self.assertIn('symbol', result.columns)
        mock_kite_instance.instruments.assert_called_once_with("NSE")
        mock_kite_instance.historical_data.assert_called_once()
    
    @patch('src.data.api_client.KiteConnect')
    def test_kite_connect_current_price(self, mock_kite_class):
        """Test Kite Connect current price fetching with mock"""
        # Setup mock
        mock_kite_instance = MagicMock()
        mock_kite_class.return_value = mock_kite_instance
        
        # Mock instruments and LTP data
        mock_instruments = [
            {'tradingsymbol': 'RELIANCE', 'instrument_token': '12345'}
        ]
        mock_kite_instance.instruments.return_value = mock_instruments
        
        mock_ltp_data = {'12345': {'last_price': 105.5}}
        mock_kite_instance.ltp.return_value = mock_ltp_data
        
        # Test price fetching
        client = KiteConnectClient("test_api_key", "test_access_token")
        price = client.get_current_price("RELIANCE.NS")
        
        # Verify results
        self.assertEqual(price, 105.5)
        mock_kite_instance.instruments.assert_called_once_with("NSE")
        mock_kite_instance.ltp.assert_called_once_with(['12345'])
    
    def test_data_fetcher_initialization(self):
        """Test DataFetcher initialization"""
        fetcher = DataFetcher(self.test_config)
        
        self.assertIsInstance(fetcher.api_manager, APIManager)
        self.assertIsInstance(fetcher.stock_loader, type(fetcher.stock_loader))
        self.assertIsInstance(fetcher.cache, type(fetcher.cache))
    
    @patch('src.data.data_fetcher.DataFetcher.api_manager')
    def test_data_fetcher_single_stock(self, mock_api_manager):
        """Test DataFetcher single stock fetching"""
        # Setup mock
        mock_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [99, 100, 101],
            'close': [104, 105, 106],
            'volume': [1000, 1100, 1200]
        })
        mock_api_manager.fetch_data.return_value = mock_data
        
        # Test fetching
        fetcher = DataFetcher(self.test_config)
        result = fetcher.fetch_stock_data("RELIANCE.NS")
        
        # Verify results
        self.assertIsInstance(result, pd.DataFrame)
        mock_api_manager.fetch_data.assert_called_once_with("RELIANCE.NS", None, None)
    
    @patch('src.data.data_fetcher.DataFetcher.api_manager')
    def test_data_fetcher_multiple_stocks(self, mock_api_manager):
        """Test DataFetcher multiple stocks fetching"""
        # Setup mock
        mock_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [99, 100, 101],
            'close': [104, 105, 106],
            'volume': [1000, 1100, 1200]
        })
        mock_api_manager.fetch_data.return_value = mock_data
        
        # Test fetching
        fetcher = DataFetcher(self.test_config)
        results = fetcher.fetch_multiple_stocks(self.test_symbols)
        
        # Verify results
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), len(self.test_symbols))
        
        for symbol in self.test_symbols:
            self.assertIn(symbol, results)
            self.assertIsInstance(results[symbol], pd.DataFrame)
    
    def test_error_handling(self):
        """Test error handling in API clients"""
        # Test Yahoo Finance with invalid symbol
        client = YahooFinanceClient()
        
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker_instance = MagicMock()
            mock_ticker.return_value = mock_ticker_instance
            mock_ticker_instance.history.return_value = pd.DataFrame()  # Empty data
            
            result = client.fetch_data("INVALID_SYMBOL", "15m", "5d")
            self.assertTrue(result.empty)
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        client = YahooFinanceClient()
        
        # Test that rate limiting delay is set
        self.assertGreater(client.rate_limit_delay, 0)
        self.assertLessEqual(client.rate_limit_delay, 1.0)  # Should be reasonable
    
    def test_data_validation(self):
        """Test data validation in DataFetcher"""
        fetcher = DataFetcher(self.test_config)
        
        # Test with invalid data
        invalid_data = pd.DataFrame({
            'open': [100, 0, 102],  # Zero price
            'high': [105, 106, 107],
            'low': [99, 100, 101],
            'close': [104, 105, 106],
            'volume': [1000, 1100, 1200]
        })
        
        validated_data = fetcher._validate_data(invalid_data, "TEST.NS")
        
        # Should remove invalid rows
        self.assertLess(len(validated_data), len(invalid_data))
    
    def test_cache_functionality(self):
        """Test caching functionality"""
        fetcher = DataFetcher(self.test_config)
        
        # Test cache stats
        stats = fetcher.get_cache_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('total_entries', stats)
        self.assertIn('active_entries', stats)
        
        # Test cache clearing
        fetcher.clear_cache()
        stats_after_clear = fetcher.get_cache_stats()
        self.assertEqual(stats_after_clear['total_entries'], 0)

if __name__ == '__main__':
    unittest.main()
