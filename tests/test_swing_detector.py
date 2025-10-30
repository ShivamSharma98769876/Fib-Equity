"""
Unit tests for swing point detection algorithms
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.swing_detector import SwingDetector

class TestSwingDetector(unittest.TestCase):
    """Test cases for SwingDetector class"""
    
    def setUp(self):
        """Set up test data"""
        self.detector = SwingDetector(min_bars_between=3)
        
        # Create sample data with known swing points
        dates = pd.date_range(start='2024-01-01', periods=50, freq='15min')
        
        # Create price data with clear swing points
        prices = [
            100, 102, 101, 103, 105, 104, 106, 108, 107, 109,  # Initial uptrend
            111, 110, 112, 114, 113, 115, 117, 116, 118, 120,  # Peak at 120
            118, 116, 114, 112, 110, 108, 106, 104, 102, 100,  # Downtrend to 100
            98, 96, 94, 92, 90, 88, 86, 84, 82, 80,           # Further decline to 80
            82, 84, 86, 88, 90, 92, 94, 96, 98, 100           # Recovery to 100
        ]
        
        # Create OHLCV data
        self.test_data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000] * len(prices)
        })
    
    def test_swing_high_detection(self):
        """Test swing high detection"""
        swing_highs = self.detector.detect_swing_highs(self.test_data, lookback=5)
        
        # Should detect the peak around index 19 (price 120)
        self.assertGreater(len(swing_highs), 0)
        
        # Check that we found the major swing high (should be around 120-121)
        swing_prices = [high[1] for high in swing_highs]
        self.assertTrue(any(price >= 120 for price in swing_prices), f"No swing high found around 120, found: {swing_prices}")
    
    def test_swing_low_detection(self):
        """Test swing low detection"""
        swing_lows = self.detector.detect_swing_lows(self.test_data, lookback=5)
        
        # Should detect the low around index 39 (price 80)
        self.assertGreater(len(swing_lows), 0)
        
        # Check that we found the major swing low (should be around 79-80)
        swing_prices = [low[1] for low in swing_lows]
        self.assertTrue(any(price <= 80 for price in swing_prices), f"No swing low found around 80, found: {swing_prices}")
    
    def test_trend_based_swing_detection_downtrend(self):
        """Test trend-based swing detection for downtrend"""
        # Create data with clear downtrend pattern that has swing points
        dates = pd.date_range(start='2024-01-01', periods=40, freq='15min')
        
        # Simple downtrend with clear swing points
        prices = [
            100, 95, 90, 85, 80, 75, 70, 65, 60, 55,            # Decline
            50, 45, 40, 35, 30, 25, 20, 15, 10, 5,              # Continue decline
            0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5  # Multiple bounces at end
        ]
        
        downtrend_data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000] * len(prices)
        })
        
        result = self.detector.detect_trend_based_swings(downtrend_data, lookback=3)
        
        # Should detect downtrend
        self.assertEqual(result['trend'], 'downtrend')
        
        # Should have trend-based swing points (at least the swing high for downtrend)
        self.assertIsNotNone(result['trend_swing_high'])
        
        # If we have both swing points, swing high should come before swing low in downtrend
        if result['trend_swing_high'] and result['trend_swing_low']:
            swing_high_index = result['trend_swing_high'][0]
            swing_low_index = result['trend_swing_low'][0]
            self.assertLess(swing_high_index, swing_low_index)
    
    def test_trend_based_swing_detection_uptrend(self):
        """Test trend-based swing detection for uptrend"""
        # Create data with clear uptrend pattern that has swing points
        dates = pd.date_range(start='2024-01-01', periods=40, freq='15min')
        
        # Uptrend with clear swing points: low first, then high
        prices = [
            5, 10, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0,  # Multiple bounces at start
            5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100  # Clear uptrend
        ]
        
        uptrend_data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000] * len(prices)
        })
        
        result = self.detector.detect_trend_based_swings(uptrend_data, lookback=3)
        
        # Should detect uptrend
        self.assertEqual(result['trend'], 'uptrend')
        
        # Should have trend-based swing points (at least the swing low for uptrend)
        self.assertIsNotNone(result['trend_swing_low'])
        
        # If we have both swing points, swing low should come before swing high in uptrend
        if result['trend_swing_low'] and result['trend_swing_high']:
            swing_low_index = result['trend_swing_low'][0]
            swing_high_index = result['trend_swing_high'][0]
            self.assertLess(swing_low_index, swing_high_index)
    
    def test_second_last_swing_high(self):
        """Test second last swing high detection"""
        second_last_high = self.detector.get_second_last_swing_high(self.test_data, lookback=5)
        
        self.assertIsNotNone(second_last_high)
        self.assertIsInstance(second_last_high, tuple)
        self.assertEqual(len(second_last_high), 3)  # Now includes datetime
    
    def test_second_last_swing_low(self):
        """Test second last swing low detection"""
        second_last_low = self.detector.get_second_last_swing_low(self.test_data, lookback=5)
        
        self.assertIsNotNone(second_last_low)
        self.assertIsInstance(second_last_low, tuple)
        self.assertEqual(len(second_last_low), 3)  # Now includes datetime
    
    def test_trend_detection(self):
        """Test trend detection"""
        trend = self.detector.detect_trend(self.test_data, lookback=5)
        
        # Should detect some trend
        self.assertIn(trend, ['uptrend', 'downtrend', 'sideways'])
    
    def test_swing_analysis(self):
        """Test comprehensive swing analysis"""
        analysis = self.detector.get_swing_analysis(self.test_data, lookback=5)
        
        # Check required keys
        required_keys = ['swing_highs', 'swing_lows', 'second_last_high', 
                        'second_last_low', 'trend', 'total_swing_highs', 'total_swing_lows']
        
        for key in required_keys:
            self.assertIn(key, analysis)
        
        # Check data types
        self.assertIsInstance(analysis['swing_highs'], list)
        self.assertIsInstance(analysis['swing_lows'], list)
        self.assertIsInstance(analysis['trend'], str)
        self.assertIsInstance(analysis['total_swing_highs'], int)
        self.assertIsInstance(analysis['total_swing_lows'], int)
    
    def test_empty_data(self):
        """Test with empty data"""
        empty_data = pd.DataFrame()
        
        swing_highs = self.detector.detect_swing_highs(empty_data)
        swing_lows = self.detector.detect_swing_lows(empty_data)
        
        self.assertEqual(len(swing_highs), 0)
        self.assertEqual(len(swing_lows), 0)
    
    def test_insufficient_data(self):
        """Test with insufficient data"""
        small_data = self.test_data.head(5)  # Only 5 rows
        
        swing_highs = self.detector.detect_swing_highs(small_data, lookback=5)
        swing_lows = self.detector.detect_swing_lows(small_data, lookback=5)
        
        # Should return empty lists due to insufficient data
        self.assertEqual(len(swing_highs), 0)
        self.assertEqual(len(swing_lows), 0)
    
    def test_minimum_bars_between(self):
        """Test minimum bars between swing points"""
        # Create data with swing points too close together
        dates = pd.date_range(start='2024-01-01', periods=20, freq='15min')
        prices = [100, 102, 100, 102, 100, 102, 100, 102, 100, 102,
                 100, 102, 100, 102, 100, 102, 100, 102, 100, 102]
        
        close_data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000] * len(prices)
        })
        
        swing_highs = self.detector.detect_swing_highs(close_data, lookback=2)
        
        # Should filter out points that are too close
        if len(swing_highs) > 1:
            for i in range(1, len(swing_highs)):
                prev_idx = swing_highs[i-1][0]
                curr_idx = swing_highs[i][0]
                self.assertGreaterEqual(curr_idx - prev_idx, self.detector.min_bars_between)
    
    def test_uptrend_data(self):
        """Test with clear uptrend data"""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='15min')
        prices = list(range(100, 130))  # Clear uptrend
        
        uptrend_data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000] * len(prices)
        })
        
        trend = self.detector.detect_trend(uptrend_data, lookback=3)
        # Should detect uptrend or sideways (depending on swing points)
        self.assertIn(trend, ['uptrend', 'sideways'])
    
    def test_downtrend_data(self):
        """Test with clear downtrend data"""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='15min')
        prices = list(range(130, 100, -1))  # Clear downtrend
        
        downtrend_data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000] * len(prices)
        })
        
        trend = self.detector.detect_trend(downtrend_data, lookback=3)
        # Should detect downtrend or sideways (depending on swing points)
        self.assertIn(trend, ['downtrend', 'sideways'])

if __name__ == '__main__':
    unittest.main()
