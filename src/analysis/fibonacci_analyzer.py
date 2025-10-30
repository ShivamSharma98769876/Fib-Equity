"""
Fibonacci retracement analysis module
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FibonacciAnalyzer:
    """Analyzes Fibonacci retracement levels based on swing points"""
    
    def __init__(self, levels: List[float] = None):
        """
        Initialize Fibonacci analyzer
        
        Args:
            levels: List of Fibonacci levels to calculate
        """
        if levels is None:
            self.levels = [1.618, 0.786, 0.618, 0.5, 0.382, 0.236]
        else:
            self.levels = levels
    
    def calculate_uptrend_retracement(self, swing_low: float, swing_high: float) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels for uptrend
        For uptrend: Draw from swing low to swing high
        - Retracement levels (0.236, 0.382, 0.5, 0.618, 0.786) are between swing low and swing high
        - Extension levels (1.618) are below swing low
        
        Args:
            swing_low: Swing low price (starting point)
            swing_high: Swing high price (ending point)
            
        Returns:
            Dictionary with Fibonacci levels
        """
        if swing_low >= swing_high:
            logger.warning("Invalid swing points: low >= high")
            return {}
        
        price_range = swing_high - swing_low
        retracement_levels = {}
        
        for level in self.levels:
            if level <= 1.0:
                # Retracement levels: swing_high - (price_range * level)
                # This gives levels between swing_high and swing_low
                price = swing_high - (price_range * level)
                retracement_levels[f"{level:.3f}"] = price
            else:
                # Extension levels: swing_low - (price_range * (level - 1))
                # This gives levels below swing_low (like 1.618)
                price = swing_low - (price_range * (level - 1))
                retracement_levels[f"{level:.3f}"] = price
        
        return retracement_levels
    
    def calculate_downtrend_retracement(self, swing_high: float, swing_low: float) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels for downtrend
        For downtrend: Draw from swing high to swing low
        - Retracement levels (0.236, 0.382, 0.5, 0.618, 0.786) are between swing high and swing low
        - Extension levels (1.618) are above swing high
        
        Args:
            swing_high: Swing high price (starting point)
            swing_low: Swing low price (ending point)
            
        Returns:
            Dictionary with Fibonacci levels
        """
        if swing_high <= swing_low:
            logger.warning("Invalid swing points: high <= low")
            return {}
        
        price_range = swing_high - swing_low
        retracement_levels = {}
        
        for level in self.levels:
            # For downtrends, compute levels from the swing low upwards.
            # This aligns retracement labels (e.g., 0.786) with common charting tools
            # and with our unit tests, which expect values like: low + range * level.
            price = swing_low + (price_range * level)
            retracement_levels[f"{level:.3f}"] = price
        
        return retracement_levels
    
    def analyze_swing_points(self, swing_high: Optional[Tuple[int, float]], 
                           swing_low: Optional[Tuple[int, float]], 
                           trend: str) -> Dict[str, float]:
        """
        Analyze Fibonacci levels based on swing points and trend
        
        Args:
            swing_high: Tuple (index, price) of swing high
            swing_low: Tuple (index, price) of swing low
            trend: Trend direction ('uptrend', 'downtrend', 'sideways')
            
        Returns:
            Dictionary with Fibonacci levels
        """
        if not swing_high or not swing_low:
            logger.warning("Missing swing points for Fibonacci analysis")
            return {}
        
        high_price = swing_high[1]
        low_price = swing_low[1]
        
        if trend == 'uptrend':
            return self.calculate_uptrend_retracement(low_price, high_price)
        elif trend == 'downtrend':
            return self.calculate_downtrend_retracement(high_price, low_price)
        else:
            # For sideways trend, use the most recent swing points
            if swing_high[0] > swing_low[0]:  # High is more recent
                return self.calculate_uptrend_retracement(low_price, high_price)
            else:  # Low is more recent
                return self.calculate_downtrend_retracement(high_price, low_price)
    
    def check_price_eligibility(self, current_price: float, fib_levels: Dict[str, float], 
                              min_level: float = 0.5, max_level: float = 0.618) -> bool:
        """
        Check if current price is eligible for swing trading
        
        Args:
            current_price: Current stock price
            fib_levels: Dictionary of Fibonacci levels
            min_level: Minimum Fibonacci level for eligibility
            max_level: Maximum Fibonacci level for eligibility
            
        Returns:
            True if price is between min_level and max_level
        """
        if not fib_levels:
            return False
        
        min_key = f"{min_level:.3f}"
        max_key = f"{max_level:.3f}"
        
        if min_key not in fib_levels or max_key not in fib_levels:
            logger.warning(f"Required Fibonacci levels not found: {min_key}, {max_key}")
            return False
        
        min_price = fib_levels[min_key]
        max_price = fib_levels[max_key]
        
        # Ensure min_price < max_price
        if min_price > max_price:
            min_price, max_price = max_price, min_price
        
        return min_price <= current_price <= max_price
    
    def get_eligibility_details(self, current_price: float, fib_levels: Dict[str, float], 
                               min_level: float = 0.5, max_level: float = 0.618) -> Dict:
        """
        Get detailed eligibility information
        
        Args:
            current_price: Current stock price
            fib_levels: Dictionary of Fibonacci levels
            min_level: Minimum Fibonacci level for eligibility
            max_level: Maximum Fibonacci level for eligibility
            
        Returns:
            Dictionary with eligibility details
        """
        if not fib_levels:
            return {
                'eligible': False,
                'reason': 'No Fibonacci levels available',
                'current_price': current_price,
                'fib_levels': fib_levels
            }
        
        min_key = f"{min_level:.3f}"
        max_key = f"{max_level:.3f}"
        
        if min_key not in fib_levels or max_key not in fib_levels:
            return {
                'eligible': False,
                'reason': f'Required Fibonacci levels not found: {min_key}, {max_key}',
                'current_price': current_price,
                'fib_levels': fib_levels
            }
        
        min_price = fib_levels[min_key]
        max_price = fib_levels[max_key]
        
        # Ensure min_price < max_price
        if min_price > max_price:
            min_price, max_price = max_price, min_price
        
        eligible = min_price <= current_price <= max_price
        
        return {
            'eligible': eligible,
            'current_price': current_price,
            'min_price': min_price,
            'max_price': max_price,
            'min_level': min_level,
            'max_level': max_level,
            'fibonacci_levels': fib_levels,
            'reason': 'Price is within eligibility range' if eligible else 'Price is outside eligibility range'
        }
    
    def get_comprehensive_analysis(self, swing_high: Optional[Tuple[int, float]], 
                                swing_low: Optional[Tuple[int, float]], 
                                trend: str, current_price: float,
                                min_level: float = 0.5, max_level: float = 0.618) -> Dict:
        """
        Get comprehensive Fibonacci analysis
        
        Args:
            swing_high: Tuple (index, price) of swing high
            swing_low: Tuple (index, price) of swing low
            trend: Trend direction
            current_price: Current stock price
            min_level: Minimum Fibonacci level for eligibility
            max_level: Maximum Fibonacci level for eligibility
            
        Returns:
            Dictionary with comprehensive analysis
        """
        # Calculate Fibonacci levels
        fib_levels = self.analyze_swing_points(swing_high, swing_low, trend)
        
        # Check eligibility
        eligibility = self.get_eligibility_details(current_price, fib_levels, min_level, max_level)
        
        return {
            'swing_high': swing_high,
            'swing_low': swing_low,
            'trend': trend,
            'current_price': current_price,
            'fibonacci_levels': fib_levels,
            'eligibility': eligibility,
            'analysis_timestamp': pd.Timestamp.now()
        }

# Convenience functions
def calculate_fibonacci_levels(swing_high: Tuple[int, float], swing_low: Tuple[int, float], 
                              trend: str, levels: List[float] = None) -> Dict[str, float]:
    """Convenience function to calculate Fibonacci levels"""
    analyzer = FibonacciAnalyzer(levels)
    return analyzer.analyze_swing_points(swing_high, swing_low, trend)

def check_swing_trade_eligibility(current_price: float, fib_levels: Dict[str, float], 
                                min_level: float = 0.5, max_level: float = 0.618) -> bool:
    """Convenience function to check swing trade eligibility"""
    analyzer = FibonacciAnalyzer()
    return analyzer.check_price_eligibility(current_price, fib_levels, min_level, max_level)

def get_fibonacci_analysis(swing_high: Tuple[int, float], swing_low: Tuple[int, float], 
                          trend: str, current_price: float, 
                          min_level: float = 0.5, max_level: float = 0.618) -> Dict:
    """Convenience function to get comprehensive Fibonacci analysis"""
    analyzer = FibonacciAnalyzer()
    return analyzer.get_comprehensive_analysis(swing_high, swing_low, trend, current_price, min_level, max_level)
