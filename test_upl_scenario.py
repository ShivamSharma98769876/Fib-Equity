#!/usr/bin/env python3
"""
Test the UPL chart scenario with swing high 678, swing low 670 in downtrend
"""

import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from analysis.swing_detector import detect_trend_based_swings
from analysis.fibonacci_analyzer import FibonacciAnalyzer

def test_upl_scenario():
    """Test the UPL chart scenario exactly as described"""
    
    print("=== UPL Chart Scenario Test ===")
    print("Swing High: 678, Swing Low: 670, Trend: Downtrend")
    print()
    
    # Create data that simulates the UPL chart scenario
    today = datetime.now().date()
    
    # Create intraday data with the specific swing points
    dates = []
    prices = []
    
    # Create data that shows a downtrend with swing high at 678 and swing low at 670
    for hour in range(9, 16):  # 9 AM to 3 PM
        for minute in [0, 15, 30, 45]:  # Every 15 minutes
            dates.append(pd.Timestamp(today, hour=hour, minute=minute))
            
            # Simulate the price action from the chart
            if hour == 9:  # Morning session - around swing high
                prices.append(678)
            elif hour == 10:  # Decline starts
                prices.append(675)
            elif hour == 11:  # Further decline
                prices.append(672)
            elif hour == 12:  # Mid-day - around swing low
                prices.append(670)
            elif hour == 13:  # Slight recovery
                prices.append(672)
            elif hour == 14:  # Continued recovery
                prices.append(674)
            else:  # End of day
                prices.append(675)
    
    # Create DataFrame
    data = pd.DataFrame({
        'datetime': dates,
        'open': prices,
        'high': [p + 1 for p in prices],
        'low': [p - 1 for p in prices],
        'close': prices,
        'volume': [1000] * len(prices)
    })
    
    print(f"Created {len(data)} data points for today")
    print(f"Price range: {data['close'].min():.2f} to {data['close'].max():.2f}")
    print()
    
    # Test trend-based swing detection (intraday only)
    print("=== Trend-Based Swing Detection (Intraday Only) ===")
    result = detect_trend_based_swings(data, lookback=2, intraday_only=True)
    
    print(f"Detected trend: {result['trend']}")
    print(f"Trend swing high: {result['trend_swing_high']}")
    print(f"Trend swing low: {result['trend_swing_low']}")
    
    if result['trend_swing_high']:
        idx, price = result['trend_swing_high']
        time = data.iloc[idx]['datetime'].strftime('%H:%M')
        print(f"  Swing high: Index {idx}, Price {price:.2f}, Time {time}")
    
    if result['trend_swing_low']:
        idx, price = result['trend_swing_low']
        time = data.iloc[idx]['datetime'].strftime('%H:%M')
        print(f"  Swing low: Index {idx}, Price {price:.2f}, Time {time}")
    
    print()
    
    # Test Fibonacci analysis with the detected swing points
    if result['trend_swing_high'] and result['trend_swing_low']:
        print("=== Fibonacci Analysis ===")
        
        analyzer = FibonacciAnalyzer()
        fib_levels = analyzer.analyze_swing_points(
            result['trend_swing_high'],
            result['trend_swing_low'],
            result['trend']
        )
        
        print(f"Fibonacci levels for {result['trend']}:")
        for level, price in fib_levels.items():
            print(f"  {level}: {price:.2f}")
        
        # Verify the Fibonacci levels match the expected pattern for downtrend
        swing_high_price = result['trend_swing_high'][1]
        swing_low_price = result['trend_swing_low'][1]
        
        print(f"\nSwing High: {swing_high_price:.2f}")
        print(f"Swing Low: {swing_low_price:.2f}")
        
        # Check if 0.0 level is at swing high and 1.0 level is at swing low
        if '0.000' in fib_levels and '1.000' in fib_levels:
            print(f"0.0 level (should be swing high): {fib_levels['0.000']:.2f}")
            print(f"1.0 level (should be swing low): {fib_levels['1.000']:.2f}")
            
            if abs(fib_levels['0.000'] - swing_high_price) < 0.01:
                print("✅ 0.0 level correctly positioned at swing high")
            else:
                print("❌ 0.0 level not at swing high")
                
            if abs(fib_levels['1.000'] - swing_low_price) < 0.01:
                print("✅ 1.0 level correctly positioned at swing low")
            else:
                print("❌ 1.0 level not at swing low")
        
        print()
        
        # Test eligibility at different price levels
        print("=== Price Eligibility Test ===")
        test_prices = [670, 672, 674, 676, 678]
        
        for test_price in test_prices:
            eligibility = analyzer.get_eligibility_details(
                test_price, fib_levels, min_level=0.5, max_level=0.618
            )
            print(f"Price {test_price}: Eligible = {eligibility['eligible']}")
            if eligibility['eligible']:
                print(f"  Range: {eligibility['min_price']:.2f} to {eligibility['max_price']:.2f}")
    
    else:
        print("❌ Could not detect swing points for Fibonacci analysis")
    
    print()
    print("=== Summary ===")
    print("This test simulates the UPL chart scenario with:")
    print("- Swing High: 678")
    print("- Swing Low: 670") 
    print("- Downtrend pattern")
    print("- Intraday-only swing detection")
    print("- Fibonacci retracement from high to low (0.0 at high, 1.0 at low)")

if __name__ == "__main__":
    test_upl_scenario()
