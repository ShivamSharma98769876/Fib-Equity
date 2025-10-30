"""
Unit tests for eligibility scanning logic
"""

import unittest
import sys
import os
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.eligibility_scanner import EligibilityScanner
from config.config import config

class TestEligibilityScanner(unittest.TestCase):
    """Test cases for EligibilityScanner class"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_config = config
        self.scanner = EligibilityScanner(self.test_config)
        
        # Mock test data
        self.mock_data = pd.DataFrame({
            'datetime': pd.date_range(start='2024-01-01', periods=50, freq='15min'),
            'open': [100 + i for i in range(50)],
            'high': [105 + i for i in range(50)],
            'low': [95 + i for i in range(50)],
            'close': [100 + i for i in range(50)],
            'volume': [1000] * 50
        })
        
        # Mock swing analysis result
        self.mock_swing_analysis = {
            'swing_highs': [(40, 140.0), (35, 135.0)],
            'swing_lows': [(30, 130.0), (25, 125.0)],
            'second_last_high': (35, 135.0),
            'second_last_low': (25, 125.0),
            'trend': 'uptrend',
            'total_swing_highs': 2,
            'total_swing_lows': 2
        }
        
        # Mock Fibonacci analysis result
        self.mock_fib_analysis = {
            'swing_high': (35, 135.0),
            'swing_low': (25, 125.0),
            'trend': 'uptrend',
            'current_price': 130.0,
            'fibonacci_levels': {
                '1.618': 141.18,
                '0.786': 132.86,
                '0.500': 130.0,
                '0.382': 127.14,
                '0.236': 124.28
            },
            'eligibility': {
                'eligible': True,
                'current_price': 130.0,
                'min_price': 130.0,
                'max_price': 131.18,
                'min_level': 0.5,
                'max_level': 0.618,
                'reason': 'Price is within eligibility range'
            },
            'analysis_timestamp': datetime.now()
        }
    
    @patch('src.analysis.eligibility_scanner.EligibilityScanner.data_fetcher')
    def test_analyze_single_stock_success(self, mock_data_fetcher):
        """Test successful analysis of a single stock"""
        # Setup mocks
        mock_data_fetcher.get_latest_data.return_value = self.mock_data
        mock_data_fetcher.api_manager.get_current_price.return_value = 130.0
        
        with patch.object(self.scanner.swing_detector, 'get_swing_analysis', return_value=self.mock_swing_analysis):
            with patch.object(self.scanner.fibonacci_analyzer, 'get_comprehensive_analysis', return_value=self.mock_fib_analysis):
                
                result = self.scanner.analyze_single_stock("RELIANCE.NS", 130.0)
                
                # Verify result structure
                self.assertIsInstance(result, dict)
                self.assertEqual(result['symbol'], "RELIANCE.NS")
                self.assertEqual(result['current_price'], 130.0)
                self.assertTrue(result['eligible'])
                self.assertIsNone(result['error'])
                
                # Verify required keys
                required_keys = ['symbol', 'current_price', 'swing_high', 'swing_low', 
                               'trend', 'fibonacci_levels', 'eligible', 'eligibility_details']
                for key in required_keys:
                    self.assertIn(key, result)
    
    @patch('src.analysis.eligibility_scanner.EligibilityScanner.data_fetcher')
    def test_analyze_single_stock_no_data(self, mock_data_fetcher):
        """Test analysis with no data available"""
        # Setup mock to return empty data
        mock_data_fetcher.get_latest_data.return_value = pd.DataFrame()
        
        result = self.scanner.analyze_single_stock("INVALID.NS")
        
        # Verify error result
        self.assertIsInstance(result, dict)
        self.assertEqual(result['symbol'], "INVALID.NS")
        self.assertFalse(result['eligible'])
        self.assertIsNotNone(result['error'])
        self.assertIn("No data available", result['error'])
    
    @patch('src.analysis.eligibility_scanner.EligibilityScanner.data_fetcher')
    def test_analyze_single_stock_insufficient_swing_points(self, mock_data_fetcher):
        """Test analysis with insufficient swing points"""
        # Setup mocks
        mock_data_fetcher.get_latest_data.return_value = self.mock_data
        mock_data_fetcher.api_manager.get_current_price.return_value = 130.0
        
        # Mock swing analysis with insufficient points
        insufficient_swing_analysis = {
            'swing_highs': [(40, 140.0)],
            'swing_lows': [(30, 130.0)],
            'second_last_high': None,
            'second_last_low': None,
            'trend': 'sideways',
            'total_swing_highs': 1,
            'total_swing_lows': 1
        }
        
        with patch.object(self.scanner.swing_detector, 'get_swing_analysis', return_value=insufficient_swing_analysis):
            
            result = self.scanner.analyze_single_stock("RELIANCE.NS", 130.0)
            
            # Verify error result
            self.assertFalse(result['eligible'])
            self.assertIsNotNone(result['error'])
            self.assertIn("Insufficient swing points", result['error'])
    
    def test_scan_multiple_stocks(self):
        """Test scanning multiple stocks"""
        symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
        current_prices = {"RELIANCE.NS": 130.0, "TCS.NS": 120.0, "HDFCBANK.NS": 110.0}
        
        with patch.object(self.scanner, 'analyze_single_stock') as mock_analyze:
            # Setup mock to return different results for each symbol
            mock_analyze.side_effect = [
                {'symbol': 'RELIANCE.NS', 'eligible': True, 'error': None},
                {'symbol': 'TCS.NS', 'eligible': False, 'error': None},
                {'symbol': 'HDFCBANK.NS', 'eligible': True, 'error': None}
            ]
            
            results = self.scanner.scan_multiple_stocks(symbols, current_prices)
            
            # Verify results
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), 3)
            
            # Check that all symbols were analyzed
            symbols_analyzed = [result['symbol'] for result in results]
            for symbol in symbols:
                self.assertIn(symbol, symbols_analyzed)
    
    def test_filter_eligible_stocks(self):
        """Test filtering eligible stocks"""
        results = [
            {'symbol': 'RELIANCE.NS', 'eligible': True},
            {'symbol': 'TCS.NS', 'eligible': False},
            {'symbol': 'HDFCBANK.NS', 'eligible': True},
            {'symbol': 'ITC.NS', 'eligible': False}
        ]
        
        eligible_stocks = self.scanner.filter_eligible_stocks(results)
        
        # Verify filtering
        self.assertEqual(len(eligible_stocks), 2)
        eligible_symbols = [stock['symbol'] for stock in eligible_stocks]
        self.assertIn('RELIANCE.NS', eligible_symbols)
        self.assertIn('HDFCBANK.NS', eligible_symbols)
        self.assertNotIn('TCS.NS', eligible_symbols)
        self.assertNotIn('ITC.NS', eligible_symbols)
    
    def test_create_results_dataframe(self):
        """Test creating results DataFrame"""
        results = [
            {
                'symbol': 'RELIANCE.NS',
                'current_price': 130.0,
                'swing_high': 135.0,
                'swing_low': 125.0,
                'trend': 'uptrend',
                'fib_1618': 141.18,
                'fib_0786': 132.86,
                'fib_0500': 130.0,
                'eligible': True
            },
            {
                'symbol': 'TCS.NS',
                'current_price': 120.0,
                'swing_high': 125.0,
                'swing_low': 115.0,
                'trend': 'downtrend',
                'fib_1618': 131.18,
                'fib_0786': 122.86,
                'fib_0500': 120.0,
                'eligible': False
            }
        ]
        
        df = self.scanner.create_results_dataframe(results)
        
        # Verify DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        
        # Check column names
        expected_columns = ['Symbol', 'Current Price', 'Swing High', 'Swing Low', 
                           'Trend', '1.618 Level', '0.786 Level', '0.5 Level', 'Eligible']
        for col in expected_columns:
            self.assertIn(col, df.columns)
    
    def test_export_results(self):
        """Test exporting results to CSV"""
        results = [
            {
                'symbol': 'RELIANCE.NS',
                'current_price': 130.0,
                'swing_high': 135.0,
                'swing_low': 125.0,
                'trend': 'uptrend',
                'fib_1618': 141.18,
                'fib_0786': 132.86,
                'fib_0500': 130.0,
                'eligible': True,
                'error': None
            }
        ]
        
        # Test export
        success = self.scanner.export_results(results, "test_results.csv", eligible_only=False)
        
        # Verify export success
        self.assertTrue(success)
        
        # Check if file was created
        self.assertTrue(os.path.exists("test_results.csv"))
        
        # Clean up
        if os.path.exists("test_results.csv"):
            os.remove("test_results.csv")
    
    def test_export_eligible_only(self):
        """Test exporting only eligible stocks"""
        results = [
            {'symbol': 'RELIANCE.NS', 'eligible': True},
            {'symbol': 'TCS.NS', 'eligible': False},
            {'symbol': 'HDFCBANK.NS', 'eligible': True}
        ]
        
        # Test export with eligible_only=True
        success = self.scanner.export_results(results, "test_eligible.csv", eligible_only=True)
        
        # Verify export success
        self.assertTrue(success)
        
        # Check if file was created and contains only eligible stocks
        if os.path.exists("test_eligible.csv"):
            df = pd.read_csv("test_eligible.csv")
            self.assertEqual(len(df), 2)  # Only 2 eligible stocks
            self.assertTrue(all(df['Eligible'] == True))
            
            # Clean up
            os.remove("test_eligible.csv")
    
    def test_get_scan_summary(self):
        """Test getting scan summary statistics"""
        results = [
            {'symbol': 'RELIANCE.NS', 'eligible': True, 'error': None},
            {'symbol': 'TCS.NS', 'eligible': False, 'error': None},
            {'symbol': 'HDFCBANK.NS', 'eligible': True, 'error': 'API Error'},
            {'symbol': 'ITC.NS', 'eligible': False, 'error': None}
        ]
        
        summary = self.scanner.get_scan_summary(results)
        
        # Verify summary structure
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary['total_stocks'], 4)
        self.assertEqual(summary['eligible_stocks'], 2)
        self.assertEqual(summary['error_stocks'], 1)
        self.assertEqual(summary['eligibility_rate'], 50.0)
        
        # Check required keys
        required_keys = ['total_stocks', 'eligible_stocks', 'error_stocks', 
                        'eligibility_rate', 'trend_distribution', 'scan_timestamp']
        for key in required_keys:
            self.assertIn(key, summary)
    
    def test_create_error_result(self):
        """Test creating error result"""
        result = self.scanner._create_error_result("TEST.NS", "Test error message")
        
        # Verify error result structure
        self.assertIsInstance(result, dict)
        self.assertEqual(result['symbol'], "TEST.NS")
        self.assertEqual(result['current_price'], 0)
        self.assertFalse(result['eligible'])
        self.assertEqual(result['error'], "Test error message")
        
        # Check required keys
        required_keys = ['symbol', 'current_price', 'swing_high', 'swing_low', 
                        'trend', 'fibonacci_levels', 'eligible', 'eligibility_details',
                        'analysis_timestamp', 'error']
        for key in required_keys:
            self.assertIn(key, result)
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Test with empty results list
        empty_results = []
        eligible = self.scanner.filter_eligible_stocks(empty_results)
        self.assertEqual(len(eligible), 0)
        
        # Test with None results
        df = self.scanner.create_results_dataframe([])
        self.assertTrue(df.empty)
        
        # Test export with empty results
        success = self.scanner.export_results([], "empty_test.csv", eligible_only=False)
        self.assertFalse(success)
    
    def test_data_validation(self):
        """Test data validation in scanner"""
        # Test with invalid data
        invalid_data = pd.DataFrame({
            'open': [100, 0, 102],  # Zero price
            'high': [105, 106, 107],
            'low': [99, 100, 101],
            'close': [104, 105, 106],
            'volume': [1000, 1100, 1200]
        })
        
        with patch.object(self.scanner.data_fetcher, 'get_latest_data', return_value=invalid_data):
            with patch.object(self.scanner.data_fetcher.api_manager, 'get_current_price', return_value=130.0):
                
                result = self.scanner.analyze_single_stock("TEST.NS")
                
                # Should handle invalid data gracefully
                self.assertIsInstance(result, dict)
                self.assertIn('error', result)

if __name__ == '__main__':
    unittest.main()
