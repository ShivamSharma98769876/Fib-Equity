#!/usr/bin/env python3
"""
Test the convenience function for trend-based swing detection
"""

import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from analysis.swing_detector import detect_trend_based_swings

def test_convenience_function():
    """Test the convenience function"""
    
    # Create simple test data
    data = pd.DataFrame({
        'datetime': pd.date_range(start='2024-01-01', periods=20, freq='15min'),
        'open': [100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5],
        'high': [101, 96, 91, 86, 81, 76, 71, 66, 61, 56, 51, 46, 41, 36, 31, 26, 21, 16, 11, 6],
        'low': [99, 94, 89, 84, 79, 74, 69, 64, 59, 54, 49, 44, 39, 34, 29, 24, 19, 14, 9, 4],
        'close': [100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5],
        'volume': [1000] * 20
    })
    
    # Test the convenience function
    result = detect_trend_based_swings(data, lookback=2, min_bars=1)
    
    print("Convenience function test result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    # Should detect downtrend
    assert result['trend'] == 'downtrend', f"Expected downtrend, got {result['trend']}"
    print("\n✓ Convenience function works correctly!")

if __name__ == "__main__":
    test_convenience_function()
