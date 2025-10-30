#!/usr/bin/env python3
"""
Test Fibonacci direction according to user specification
"""

import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from analysis.fibonacci_analyzer import FibonacciAnalyzer

def test_fibonacci_direction():
    """Test Fibonacci direction according to user specification"""
    
    analyzer = FibonacciAnalyzer()
    
    print("=== Testing Fibonacci Direction ===")
    print()
    
    # Test uptrend: swing low to swing high
    print("UPTREND: Fibonacci drawn from swing low to swing high")
    print("Swing Low: 100, Swing High: 120")
    print("Expected: 0.0 level = 100 (swing low), 1.0 level = 120 (swing high)")
    
    uptrend_levels = analyzer.calculate_uptrend_retracement(100, 120)
    print("Calculated levels:")
    for level, price in uptrend_levels.items():
        print(f"  {level}: {price:.2f}")
    
    # Verify 0.0 and 1.0 levels
    print(f"\n0.0 level (should be 100): {uptrend_levels.get('0.000', 'Not found')}")
    print(f"1.0 level (should be 120): {uptrend_levels.get('1.000', 'Not found')}")
    
    print("\n" + "="*50)
    
    # Test downtrend: swing high to swing low  
    print("DOWNTREND: Fibonacci drawn from swing high to swing low")
    print("Swing High: 120, Swing Low: 100")
    print("Expected: 0.0 level = 120 (swing high), 1.0 level = 100 (swing low)")
    
    downtrend_levels = analyzer.calculate_downtrend_retracement(120, 100)
    print("Calculated levels:")
    for level, price in downtrend_levels.items():
        print(f"  {level}: {price:.2f}")
    
    # Verify 0.0 and 1.0 levels
    print(f"\n0.0 level (should be 120): {downtrend_levels.get('0.000', 'Not found')}")
    print(f"1.0 level (should be 100): {downtrend_levels.get('1.000', 'Not found')}")
    
    print("\n" + "="*50)
    
    # Test with trend-based swing detection
    print("TESTING WITH TREND-BASED SWING DETECTION")
    
    # Create test data
    from analysis.swing_detector import detect_trend_based_swings
    
    # Uptrend data
    uptrend_data = pd.DataFrame({
        'datetime': pd.date_range(start='2024-01-01', periods=20, freq='15min'),
        'open': [50, 55, 50, 60, 55, 65, 60, 70, 65, 75, 70, 80, 75, 85, 80, 90, 85, 95, 90, 100],
        'high': [51, 56, 51, 61, 56, 66, 61, 71, 66, 76, 71, 81, 76, 86, 81, 91, 86, 96, 91, 101],
        'low': [49, 54, 49, 59, 54, 64, 59, 69, 64, 74, 69, 79, 74, 84, 79, 89, 84, 94, 89, 99],
        'close': [50, 55, 50, 60, 55, 65, 60, 70, 65, 75, 70, 80, 75, 85, 80, 90, 85, 95, 90, 100],
        'volume': [1000] * 20
    })
    
    # Get trend-based swings
    swing_result = detect_trend_based_swings(uptrend_data, lookback=2)
    print(f"Detected trend: {swing_result['trend']}")
    print(f"Trend swing low: {swing_result['trend_swing_low']}")
    print(f"Trend swing high: {swing_result['trend_swing_high']}")
    
    if swing_result['trend_swing_low'] and swing_result['trend_swing_high']:
        # Calculate Fibonacci levels using trend-based swings
        fib_levels = analyzer.analyze_swing_points(
            swing_result['trend_swing_high'],
            swing_result['trend_swing_low'], 
            swing_result['trend']
        )
        
        print(f"\nFibonacci levels for trend-based swings:")
        for level, price in fib_levels.items():
            print(f"  {level}: {price:.2f}")

if __name__ == "__main__":
    test_fibonacci_direction()
