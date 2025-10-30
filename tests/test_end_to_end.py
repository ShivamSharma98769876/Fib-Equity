"""
End-to-end integration tests for the complete workflow
"""

import unittest
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.config import config
from data.stock_loader import StockLoader
from data.data_fetcher import DataFetcher
from analysis.eligibility_scanner import EligibilityScanner
from analysis.swing_detector import SwingDetector
from analysis.fibonacci_analyzer import FibonacciAnalyzer

class TestEndToEndWorkflow(unittest.TestCase):
    """End-to-end tests for the complete workflow"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_config = config
        self.test_symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
        
        # Create mock OHLCV data
        self.mock_ohlcv_data = pd.DataFrame({
            'datetime': pd.date_range(start='2024-01-01', periods=50, freq='15min'),
            'open': [100 + i for i in range(50)],
            'high': [105 + i for i in range(50)],
            'low': [95 + i for i in range(50)],
            'close': [100 + i for i in range(50)],
            'volume': [1000] * 50,
            'symbol': ['RELIANCE.NS'] * 50
        })
    
    def test_complete_workflow_with_mocks(self):
        """Test complete workflow with mocked data"""
        
        # Step 1: Load stock symbols
        with patch.object(StockLoader, 'load_stocks') as mock_load:
            mock_load.return_value = self.test_symbols
            
            loader = StockLoader()
            symbols = loader.load_stocks("test_stocks.txt")
            
            self.assertEqual(symbols, self.test_symbols)
            mock_load.assert_called_once_with("test_stocks.txt")
        
        # Step 2: Fetch data for symbols
        with patch.object(DataFetcher, 'fetch_multiple_stocks') as mock_fetch:
            mock_fetch.return_value = {
                'RELIANCE.NS': self.mock_ohlcv_data,
                'TCS.NS': self.mock_ohlcv_data.copy(),
                'HDFCBANK.NS': self.mock_ohlcv_data.copy()
            }
            
            fetcher = DataFetcher(self.test_config)
            data_dict = fetcher.fetch_multiple_stocks(self.test_symbols)
            
            self.assertEqual(len(data_dict), 3)
            for symbol in self.test_symbols:
                self.assertIn(symbol, data_dict)
                self.assertIsInstance(data_dict[symbol], pd.DataFrame)
        
        # Step 3: Analyze swing points
        with patch.object(SwingDetector, 'get_swing_analysis') as mock_swing:
            mock_swing.return_value = {
                'swing_highs': [(40, 140.0), (35, 135.0)],
                'swing_lows': [(30, 130.0), (25, 125.0)],
                'second_last_high': (35, 135.0),
                'second_last_low': (25, 125.0),
                'trend': 'uptrend',
                'total_swing_highs': 2,
                'total_swing_lows': 2
            }
            
            detector = SwingDetector()
            swing_analysis = detector.get_swing_analysis(self.mock_ohlcv_data)
            
            self.assertIsInstance(swing_analysis, dict)
            self.assertIn('swing_highs', swing_analysis)
            self.assertIn('swing_lows', swing_analysis)
            self.assertIn('trend', swing_analysis)
        
        # Step 4: Calculate Fibonacci levels
        with patch.object(FibonacciAnalyzer, 'get_comprehensive_analysis') as mock_fib:
            mock_fib.return_value = {
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
                    'reason': 'Price is within eligibility range'
                }
            }
            
            analyzer = FibonacciAnalyzer()
            fib_analysis = analyzer.get_comprehensive_analysis(
                (35, 135.0), (25, 125.0), 'uptrend', 130.0
            )
            
            self.assertIsInstance(fib_analysis, dict)
            self.assertIn('fibonacci_levels', fib_analysis)
            self.assertIn('eligibility', fib_analysis)
            self.assertTrue(fib_analysis['eligibility']['eligible'])
        
        # Step 5: Complete eligibility scanning
        with patch.object(EligibilityScanner, 'scan_multiple_stocks') as mock_scan:
            mock_scan.return_value = [
                {
                    'symbol': 'RELIANCE.NS',
                    'current_price': 130.0,
                    'swing_high': 135.0,
                    'swing_low': 125.0,
                    'trend': 'uptrend',
                    'fibonacci_levels': {'0.500': 130.0, '0.618': 131.18},
                    'eligible': True,
                    'error': None
                },
                {
                    'symbol': 'TCS.NS',
                    'current_price': 120.0,
                    'swing_high': 125.0,
                    'swing_low': 115.0,
                    'trend': 'downtrend',
                    'fibonacci_levels': {'0.500': 120.0, '0.618': 121.18},
                    'eligible': False,
                    'error': None
                },
                {
                    'symbol': 'HDFCBANK.NS',
                    'current_price': 110.0,
                    'swing_high': 115.0,
                    'swing_low': 105.0,
                    'trend': 'sideways',
                    'fibonacci_levels': {'0.500': 110.0, '0.618': 111.18},
                    'eligible': True,
                    'error': None
                }
            ]
            
            scanner = EligibilityScanner(self.test_config)
            results = scanner.scan_multiple_stocks(self.test_symbols)
            
            # Verify results
            self.assertEqual(len(results), 3)
            
            # Check eligible stocks
            eligible_stocks = [r for r in results if r['eligible']]
            self.assertEqual(len(eligible_stocks), 2)
            
            # Check specific results
            reliance_result = next(r for r in results if r['symbol'] == 'RELIANCE.NS')
            self.assertTrue(reliance_result['eligible'])
            self.assertEqual(reliance_result['current_price'], 130.0)
    
    def test_workflow_with_real_data_structure(self):
        """Test workflow with realistic data structure"""
        
        # Create realistic OHLCV data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')
        
        # Create price data with clear swing points
        base_price = 100
        prices = []
        for i in range(100):
            # Create a pattern with swing points
            if i < 20:
                price = base_price + i * 0.5  # Uptrend
            elif i < 40:
                price = base_price + 10 - (i - 20) * 0.5  # Downtrend
            elif i < 60:
                price = base_price + (i - 40) * 0.3  # Slow uptrend
            elif i < 80:
                price = base_price + 6 - (i - 60) * 0.2  # Slow downtrend
            else:
                price = base_price + 2 + (i - 80) * 0.1  # Final uptrend
            
            prices.append(price)
        
        realistic_data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000 + i * 10 for i in range(100)],
            'symbol': ['RELIANCE.NS'] * 100
        })
        
        # Test swing detection with realistic data
        detector = SwingDetector()
        swing_analysis = detector.get_swing_analysis(realistic_data, lookback=5)
        
        # Verify swing analysis
        self.assertIsInstance(swing_analysis, dict)
        self.assertIn('swing_highs', swing_analysis)
        self.assertIn('swing_lows', swing_analysis)
        self.assertIn('trend', swing_analysis)
        
        # Test Fibonacci analysis
        if swing_analysis['second_last_high'] and swing_analysis['second_last_low']:
            analyzer = FibonacciAnalyzer()
            current_price = realistic_data['close'].iloc[-1]
            
            fib_analysis = analyzer.get_comprehensive_analysis(
                swing_analysis['second_last_high'],
                swing_analysis['second_last_low'],
                swing_analysis['trend'],
                current_price
            )
            
            # Verify Fibonacci analysis
            self.assertIsInstance(fib_analysis, dict)
            self.assertIn('fibonacci_levels', fib_analysis)
            self.assertIn('eligibility', fib_analysis)
    
    def test_error_handling_workflow(self):
        """Test error handling in the complete workflow"""
        
        # Test with invalid symbols
        invalid_symbols = ['INVALID1.NS', 'INVALID2.NS']
        
        with patch.object(DataFetcher, 'fetch_multiple_stocks') as mock_fetch:
            mock_fetch.return_value = {}  # No data returned
            
            fetcher = DataFetcher(self.test_config)
            data_dict = fetcher.fetch_multiple_stocks(invalid_symbols)
            
            # Should handle empty results gracefully
            self.assertEqual(len(data_dict), 0)
        
        # Test with insufficient data
        insufficient_data = pd.DataFrame({
            'datetime': pd.date_range(start='2024-01-01', periods=5, freq='15min'),
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        detector = SwingDetector()
        swing_analysis = detector.get_swing_analysis(insufficient_data, lookback=5)
        
        # Should handle insufficient data gracefully
        self.assertIsInstance(swing_analysis, dict)
        self.assertIn('swing_highs', swing_analysis)
        self.assertIn('swing_lows', swing_analysis)
    
    def test_performance_workflow(self):
        """Test performance with multiple symbols"""
        
        # Test with larger symbol list
        large_symbol_list = [f'STOCK{i}.NS' for i in range(20)]
        
        with patch.object(DataFetcher, 'fetch_multiple_stocks') as mock_fetch:
            # Mock data for all symbols
            mock_data = {}
            for symbol in large_symbol_list:
                mock_data[symbol] = self.mock_ohlcv_data.copy()
            
            mock_fetch.return_value = mock_data
            
            fetcher = DataFetcher(self.test_config)
            start_time = datetime.now()
            
            data_dict = fetcher.fetch_multiple_stocks(large_symbol_list)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Verify results
            self.assertEqual(len(data_dict), len(large_symbol_list))
            
            # Performance should be reasonable (less than 10 seconds for 20 symbols)
            self.assertLess(processing_time, 10.0)
    
    def test_data_consistency_workflow(self):
        """Test data consistency across the workflow"""
        
        # Create consistent test data
        test_data = self.mock_ohlcv_data.copy()
        
        # Test data validation
        fetcher = DataFetcher(self.test_config)
        validated_data = fetcher._validate_data(test_data, "TEST.NS")
        
        # Should pass validation
        self.assertIsInstance(validated_data, pd.DataFrame)
        self.assertGreater(len(validated_data), 0)
        
        # Test swing detection consistency
        detector = SwingDetector()
        swing_analysis = detector.get_swing_analysis(validated_data)
        
        # Should have consistent results
        self.assertIsInstance(swing_analysis, dict)
        
        # Test Fibonacci analysis consistency
        if swing_analysis['second_last_high'] and swing_analysis['second_last_low']:
            analyzer = FibonacciAnalyzer()
            current_price = validated_data['close'].iloc[-1]
            
            fib_analysis = analyzer.get_comprehensive_analysis(
                swing_analysis['second_last_high'],
                swing_analysis['second_last_low'],
                swing_analysis['trend'],
                current_price
            )
            
            # Should have consistent results
            self.assertIsInstance(fib_analysis, dict)
            self.assertIn('fibonacci_levels', fib_analysis)
            self.assertIn('eligibility', fib_analysis)
    
    def test_export_workflow(self):
        """Test export functionality in the workflow"""
        
        # Create test results
        test_results = [
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
                'eligible': False,
                'error': None
            }
        ]
        
        # Test export
        scanner = EligibilityScanner(self.test_config)
        success = scanner.export_results(test_results, "test_workflow_export.csv")
        
        # Verify export success
        self.assertTrue(success)
        
        # Check if file was created and contains expected data
        if os.path.exists("test_workflow_export.csv"):
            df = pd.read_csv("test_workflow_export.csv")
            
            # Verify DataFrame structure
            self.assertEqual(len(df), 2)
            self.assertIn('Symbol', df.columns)
            self.assertIn('Eligible', df.columns)
            
            # Clean up
            os.remove("test_workflow_export.csv")
    
    def test_configuration_workflow(self):
        """Test workflow with different configurations"""
        
        # Test with different lookback periods
        for lookback in [10, 20, 30]:
            detector = SwingDetector()
            swing_analysis = detector.get_swing_analysis(self.mock_ohlcv_data, lookback=lookback)
            
            # Should work with different lookback periods
            self.assertIsInstance(swing_analysis, dict)
        
        # Test with different Fibonacci levels
        custom_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        analyzer = FibonacciAnalyzer(custom_levels)
        
        levels = analyzer.calculate_uptrend_retracement(100.0, 120.0)
        
        # Should calculate custom levels
        for level in custom_levels:
            level_key = f"{level:.3f}"
            self.assertIn(level_key, levels)
    
    def test_monitoring_workflow(self):
        """Test continuous monitoring workflow"""
        
        from data.monitor import ContinuousMonitor, AlertManager
        
        # Test alert manager
        alert_manager = AlertManager()
        
        # Add test alert
        test_alert = {
            'symbol': 'RELIANCE.NS',
            'timestamp': datetime.now(),
            'current_price': 130.0,
            'trend': 'uptrend',
            'alert_type': 'swing_trade_opportunity'
        }
        alert_manager.add_alert(test_alert)
        
        # Verify alert was added
        recent_alerts = alert_manager.get_recent_alerts(hours=1)
        self.assertEqual(len(recent_alerts), 1)
        self.assertEqual(recent_alerts[0]['symbol'], 'RELIANCE.NS')
        
        # Test monitor initialization
        monitor = ContinuousMonitor(self.test_config, self.test_symbols, update_interval=30)
        
        # Verify monitor setup
        self.assertEqual(monitor.symbols, self.test_symbols)
        self.assertEqual(monitor.update_interval, 30)
        self.assertFalse(monitor.is_running)

if __name__ == '__main__':
    unittest.main()
