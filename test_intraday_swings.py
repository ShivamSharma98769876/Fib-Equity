#!/usr/bin/env python3
"""
Test intraday swing detection functionality
"""

import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from analysis.swing_detector import SwingDetector, detect_trend_based_swings

def test_intraday_swing_detection():
    """Test intraday swing detection"""
    
    print("=== Testing Intraday Swing Detection ===")
    print()
    
    # Create test data with multiple days
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    
    # Create data spanning multiple days
    dates = []
    prices = []
    
    # Yesterday's data
    for hour in range(9, 16):  # 9 AM to 3 PM
        for minute in [0, 15, 30, 45]:  # Every 15 minutes
            dates.append(pd.Timestamp(yesterday, hour=hour, minute=minute))
            prices.append(100 + (hour - 9) * 2)  # Rising trend yesterday
    
    # Today's data with swing points
    for hour in range(9, 16):  # 9 AM to 3 PM
        for minute in [0, 15, 30, 45]:  # Every 15 minutes
            dates.append(pd.Timestamp(today, hour=hour, minute=minute))
            if hour < 12:
                prices.append(120 - (hour - 9) * 5)  # Declining in morning
            else:
                prices.append(105 + (hour - 12) * 3)  # Rising in afternoon
    
    # Create DataFrame
    data = pd.DataFrame({
        'datetime': dates,
        'open': prices,
        'high': [p + 1 for p in prices],
        'low': [p - 1 for p in prices],
        'close': prices,
        'volume': [1000] * len(prices)
    })
    
    print(f"Total data points: {len(data)}")
    print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")
    print(f"Today's data points: {len(data[data['datetime'].dt.date == today])}")
    print()
    
    detector = SwingDetector(min_bars_between=2)
    
    # Test regular swing detection (all data)
    print("=== Regular Swing Detection (All Data) ===")
    all_swing_highs = detector.detect_swing_highs(data, lookback=2)
    all_swing_lows = detector.detect_swing_lows(data, lookback=2)
    
    print(f"All swing highs: {len(all_swing_highs)}")
    for i, (idx, price) in enumerate(all_swing_highs):
        date = data.iloc[idx]['datetime'].date()
        print(f"  {i+1}: Index {idx}, Price {price:.2f}, Date {date}")
    
    print(f"All swing lows: {len(all_swing_lows)}")
    for i, (idx, price) in enumerate(all_swing_lows):
        date = data.iloc[idx]['datetime'].date()
        print(f"  {i+1}: Index {idx}, Price {price:.2f}, Date {date}")
    
    print()
    
    # Test intraday swing detection (today only)
    print("=== Intraday Swing Detection (Today Only) ===")
    intraday_swing_highs = detector.detect_intraday_swing_highs(data, lookback=2)
    intraday_swing_lows = detector.detect_intraday_swing_lows(data, lookback=2)
    
    print(f"Intraday swing highs: {len(intraday_swing_highs)}")
    for i, (idx, price) in enumerate(intraday_swing_highs):
        date = data.iloc[idx]['datetime'].date()
        print(f"  {i+1}: Index {idx}, Price {price:.2f}, Date {date}")
    
    print(f"Intraday swing lows: {len(intraday_swing_lows)}")
    for i, (idx, price) in enumerate(intraday_swing_lows):
        date = data.iloc[idx]['datetime'].date()
        print(f"  {i+1}: Index {idx}, Price {price:.2f}, Date {date}")
    
    print()
    
    # Test trend-based swing detection with intraday only
    print("=== Trend-Based Swing Detection (Intraday Only) ===")
    result = detect_trend_based_swings(data, lookback=2, intraday_only=True)
    
    print(f"Detected trend: {result['trend']}")
    print(f"Trend swing high: {result['trend_swing_high']}")
    print(f"Trend swing low: {result['trend_swing_low']}")
    
    if result['trend_swing_high']:
        idx, price = result['trend_swing_high']
        date = data.iloc[idx]['datetime'].date()
        print(f"  Trend swing high: Index {idx}, Price {price:.2f}, Date {date}")
    
    if result['trend_swing_low']:
        idx, price = result['trend_swing_low']
        date = data.iloc[idx]['datetime'].date()
        print(f"  Trend swing low: Index {idx}, Price {price:.2f}, Date {date}")
    
    print()
    
    # Verify all intraday swing points are from today
    print("=== Verification ===")
    all_today = True
    for idx, price in intraday_swing_highs + intraday_swing_lows:
        if data.iloc[idx]['datetime'].date() != today:
            all_today = False
            print(f"❌ Found swing point from {data.iloc[idx]['datetime'].date()}, expected {today}")
    
    if all_today:
        print("✅ All intraday swing points are from today")
    else:
        print("❌ Some intraday swing points are not from today")

if __name__ == "__main__":
    test_intraday_swing_detection()
