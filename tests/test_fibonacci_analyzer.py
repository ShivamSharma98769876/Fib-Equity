"""
Unit tests for Fibonacci analysis functions
"""

import unittest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.fibonacci_analyzer import FibonacciAnalyzer

class TestFibonacciAnalyzer(unittest.TestCase):
    """Test cases for FibonacciAnalyzer class"""
    
    def setUp(self):
        """Set up test data"""
        self.analyzer = FibonacciAnalyzer()
        
        # Test swing points
        self.swing_high = (10, 120.0)  # (index, price)
        self.swing_low = (5, 100.0)    # (index, price)
    
    def test_uptrend_retracement_calculation(self):
        """Test Fibonacci retracement calculation for uptrend"""
        swing_low = 100.0
        swing_high = 120.0
        
        levels = self.analyzer.calculate_uptrend_retracement(swing_low, swing_high)
        
        # Check that levels are calculated
        self.assertIsInstance(levels, dict)
        self.assertGreater(len(levels), 0)
        
        # Check specific levels
        self.assertIn('0.500', levels)
        self.assertIn('0.618', levels)
        self.assertIn('0.786', levels)
        
        # Verify calculations
        price_range = swing_high - swing_low  # 20
        expected_0_5 = swing_high - (price_range * 0.5)  # 120 - 10 = 110
        expected_0_618 = swing_high - (price_range * 0.618)  # 120 - 12.36 = 107.64
        
        self.assertAlmostEqual(levels['0.500'], expected_0_5, places=2)
        self.assertAlmostEqual(levels['0.618'], expected_0_618, places=2)
    
    def test_downtrend_retracement_calculation(self):
        """Test Fibonacci retracement calculation for downtrend"""
        swing_high = 120.0
        swing_low = 100.0
        
        levels = self.analyzer.calculate_downtrend_retracement(swing_high, swing_low)
        
        # Check that levels are calculated
        self.assertIsInstance(levels, dict)
        self.assertGreater(len(levels), 0)
        
        # Check specific levels
        self.assertIn('0.500', levels)
        self.assertIn('0.618', levels)
        self.assertIn('0.786', levels)
        
        # Verify calculations
        price_range = swing_high - swing_low  # 20
        expected_0_5 = swing_low + (price_range * 0.5)  # 100 + 10 = 110
        expected_0_618 = swing_low + (price_range * 0.618)  # 100 + 12.36 = 112.36
        
        self.assertAlmostEqual(levels['0.500'], expected_0_5, places=2)
        self.assertAlmostEqual(levels['0.618'], expected_0_618, places=2)
    
    def test_analyze_swing_points_uptrend(self):
        """Test analysis with uptrend swing points"""
        trend = 'uptrend'
        
        analysis = self.analyzer.analyze_swing_points(
            self.swing_high, self.swing_low, trend
        )
        
        # Check that levels are calculated
        self.assertIsInstance(analysis, dict)
        self.assertGreater(len(analysis), 0)
        
        # Check specific levels exist
        self.assertIn('0.500', analysis)
        self.assertIn('0.618', analysis)
        self.assertIn('0.786', analysis)
    
    def test_analyze_swing_points_downtrend(self):
        """Test analysis with downtrend swing points"""
        trend = 'downtrend'
        
        analysis = self.analyzer.analyze_swing_points(
            self.swing_high, self.swing_low, trend
        )
        
        # Check that levels are calculated
        self.assertIsInstance(analysis, dict)
        self.assertGreater(len(analysis), 0)
        
        # Check specific levels exist
        self.assertIn('0.500', analysis)
        self.assertIn('0.618', analysis)
        self.assertIn('0.786', analysis)
    
    def test_analyze_swing_points_sideways(self):
        """Test analysis with sideways trend"""
        trend = 'sideways'
        
        analysis = self.analyzer.analyze_swing_points(
            self.swing_high, self.swing_low, trend
        )
        
        # Should still calculate levels
        self.assertIsInstance(analysis, dict)
        self.assertGreater(len(analysis), 0)
    
    def test_price_eligibility_check(self):
        """Test price eligibility checking"""
        # Create test Fibonacci levels
        fib_levels = {
            '0.500': 110.0,
            '0.618': 107.64,
            '0.786': 104.28
        }
        
        # Test price within eligibility range
        current_price = 108.0  # Between 0.5 and 0.618 levels
        eligible = self.analyzer.check_price_eligibility(
            current_price, fib_levels, 0.5, 0.618
        )
        self.assertTrue(eligible)
        
        # Test price outside eligibility range
        current_price = 105.0  # Below 0.5 level
        eligible = self.analyzer.check_price_eligibility(
            current_price, fib_levels, 0.5, 0.618
        )
        self.assertFalse(eligible)
    
    def test_eligibility_details(self):
        """Test detailed eligibility information"""
        fib_levels = {
            '0.500': 110.0,
            '0.618': 107.64
        }
        
        current_price = 108.0
        
        details = self.analyzer.get_eligibility_details(
            current_price, fib_levels, 0.5, 0.618
        )
        
        # Check required keys
        required_keys = ['eligible', 'current_price', 'min_price', 'max_price', 
                        'min_level', 'max_level', 'fibonacci_levels', 'reason']
        
        for key in required_keys:
            self.assertIn(key, details)
        
        # Check values
        self.assertTrue(details['eligible'])
        self.assertEqual(details['current_price'], current_price)
        self.assertEqual(details['min_level'], 0.5)
        self.assertEqual(details['max_level'], 0.618)
    
    def test_comprehensive_analysis(self):
        """Test comprehensive Fibonacci analysis"""
        current_price = 108.0
        
        analysis = self.analyzer.get_comprehensive_analysis(
            self.swing_high, self.swing_low, 'uptrend', current_price
        )
        
        # Check required keys
        required_keys = ['swing_high', 'swing_low', 'trend', 'current_price',
                        'fibonacci_levels', 'eligibility', 'analysis_timestamp']
        
        for key in required_keys:
            self.assertIn(key, analysis)
        
        # Check data types
        self.assertIsInstance(analysis['swing_high'], tuple)
        self.assertIsInstance(analysis['swing_low'], tuple)
        self.assertIsInstance(analysis['trend'], str)
        self.assertIsInstance(analysis['current_price'], float)
        self.assertIsInstance(analysis['fibonacci_levels'], dict)
        self.assertIsInstance(analysis['eligibility'], dict)
    
    def test_invalid_swing_points(self):
        """Test with invalid swing points"""
        # Test with None values
        analysis = self.analyzer.analyze_swing_points(None, None, 'uptrend')
        self.assertEqual(analysis, {})
        
        # Test with empty swing points
        analysis = self.analyzer.analyze_swing_points((), (), 'uptrend')
        self.assertEqual(analysis, {})
    
    def test_invalid_price_range(self):
        """Test with invalid price ranges"""
        # Test with swing_low >= swing_high
        levels = self.analyzer.calculate_uptrend_retracement(120.0, 100.0)
        self.assertEqual(levels, {})
        
        # Test with swing_high <= swing_low
        levels = self.analyzer.calculate_downtrend_retracement(100.0, 120.0)
        self.assertEqual(levels, {})
    
    def test_custom_fibonacci_levels(self):
        """Test with custom Fibonacci levels"""
        custom_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        analyzer = FibonacciAnalyzer(custom_levels)
        
        levels = analyzer.calculate_uptrend_retracement(100.0, 120.0)
        
        # Check that custom levels are calculated
        for level in custom_levels:
            level_key = f"{level:.3f}"
            self.assertIn(level_key, levels)
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Test with zero price range
        levels = self.analyzer.calculate_uptrend_retracement(100.0, 100.0)
        self.assertEqual(levels, {})
        
        # Test with very small price range
        levels = self.analyzer.calculate_uptrend_retracement(100.0, 100.01)
        self.assertIsInstance(levels, dict)
        
        # Test with negative prices
        levels = self.analyzer.calculate_uptrend_retracement(-100.0, -80.0)
        self.assertIsInstance(levels, dict)
        self.assertGreater(len(levels), 0)
    
    def test_eligibility_boundary_cases(self):
        """Test eligibility checking at boundaries"""
        fib_levels = {
            '0.500': 110.0,
            '0.618': 107.64
        }
        
        # Test price exactly at minimum level
        current_price = 110.0
        eligible = self.analyzer.check_price_eligibility(
            current_price, fib_levels, 0.5, 0.618
        )
        self.assertTrue(eligible)
        
        # Test price exactly at maximum level
        current_price = 107.64
        eligible = self.analyzer.check_price_eligibility(
            current_price, fib_levels, 0.5, 0.618
        )
        self.assertTrue(eligible)
        
        # Test price just above maximum level
        current_price = 107.65
        eligible = self.analyzer.check_price_eligibility(
            current_price, fib_levels, 0.5, 0.618
        )
        self.assertFalse(eligible)

if __name__ == '__main__':
    unittest.main()
