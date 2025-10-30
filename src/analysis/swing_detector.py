"""
Swing point detection algorithms for identifying swing highs and lows
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class SwingDetector:
    """Detects swing highs and lows in price data"""
    
    def __init__(self, min_bars_between: int = 3):
        """
        Initialize swing detector
        
        Args:
            min_bars_between: Minimum number of bars between swing points
        """
        self.min_bars_between = min_bars_between
    
    def detect_swing_highs(self, data: pd.DataFrame, lookback: int = 5, vix_data: pd.DataFrame = None) -> List[Tuple[int, float, str]]:
        """
        Detect swing highs in the data with improved sensitivity for recent trends
        
        Args:
            data: DataFrame with OHLCV data
            lookback: Number of bars to look back/forward for comparison
            vix_data: Optional VIX data for volatility-adjusted detection
            
        Returns:
            List of tuples (index, price, datetime) for swing highs
        """
        if data.empty or len(data) < lookback * 2:
            return []
        
        swing_highs = []
        high_prices = data['high'].values
        # Use only highs for determining swing highs
        
        # Use smaller lookback for more sensitive detection
        effective_lookback = min(lookback, 2)
        
        # Get datetime column for timestamps
        if 'datetime' in data.columns:
            datetime_values = lambda x: str(data['datetime'].iloc[x])
        elif 'Datetime' in data.columns:
            # Yahoo Finance uses 'Datetime' column after reset_index()
            datetime_values = lambda x: str(data['Datetime'].iloc[x])
        elif hasattr(data.index, 'to_pydatetime'):
            # If index is datetime-like, convert to string
            datetime_values = lambda x: str(data.index[x])
        else:
            # Fallback: create a simple timestamp
            datetime_values = lambda x: f"Index_{x}"
        
        for i in range(effective_lookback, len(data) - effective_lookback):
            current_high = high_prices[i]
            
            # Check if current bar is a local maximum using highs only
            is_local_max = True
            for j in range(i - effective_lookback, i + effective_lookback + 1):
                if j != i and high_prices[j] >= current_high:
                    is_local_max = False
                    break
            
            # Simple swing detection without VIX dependency
            if is_local_max:
                # Get datetime for this swing point
                swing_datetime = str(datetime_values(i))
                swing_highs.append((i, current_high, swing_datetime))
        
        # Filter out swing highs that are too close together
        filtered_highs = self._filter_nearby_points(swing_highs)
        
        logger.info(f"Detected {len(filtered_highs)} swing highs")
        return filtered_highs
    
    def detect_swing_lows(self, data: pd.DataFrame, lookback: int = 5, vix_data: pd.DataFrame = None) -> List[Tuple[int, float, str]]:
        """
        Detect swing lows in the data with improved sensitivity for recent trends
        
        Args:
            data: DataFrame with OHLCV data
            lookback: Number of bars to look back/forward for comparison
            vix_data: Optional VIX data for volatility-adjusted detection
            
        Returns:
            List of tuples (index, price, datetime) for swing lows
        """
        if data.empty or len(data) < lookback * 2:
            return []
        
        swing_lows = []
        low_prices = data['low'].values
        # Use only lows for determining swing lows
        
        # Use smaller lookback for more sensitive detection
        effective_lookback = min(lookback, 2)
        
        # Get datetime column for timestamps
        if 'datetime' in data.columns:
            datetime_values = lambda x: str(data['datetime'].iloc[x])
        elif 'Datetime' in data.columns:
            # Yahoo Finance uses 'Datetime' column after reset_index()
            datetime_values = lambda x: str(data['Datetime'].iloc[x])
        elif hasattr(data.index, 'to_pydatetime'):
            # If index is datetime-like, convert to string
            datetime_values = lambda x: str(data.index[x])
        else:
            # Fallback: create a simple timestamp
            datetime_values = lambda x: f"Index_{x}"
        
        for i in range(effective_lookback, len(data) - effective_lookback):
            current_low = low_prices[i]
            
            # Check if current bar is a local minimum using lows only
            is_local_min = True
            for j in range(i - effective_lookback, i + effective_lookback + 1):
                if j != i and low_prices[j] <= current_low:
                    is_local_min = False
                    break
            
            # Simple swing detection without VIX dependency
            if is_local_min:
                # Get datetime for this swing point
                swing_datetime = str(datetime_values(i))
                swing_lows.append((i, current_low, swing_datetime))
        
        # Filter out swing lows that are too close together
        filtered_lows = self._filter_nearby_points(swing_lows)
        
        logger.info(f"Detected {len(filtered_lows)} swing lows")
        return filtered_lows
    
    def _filter_nearby_points(self, points: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """
        Filter out swing points that are too close together
        
        Args:
            points: List of (index, price) tuples
            
        Returns:
            Filtered list of swing points
        """
        if len(points) <= 1:
            return points
        
        # Sort by index
        points.sort(key=lambda x: x[0])
        
        filtered = [points[0]]  # Keep the first point
        
        for i in range(1, len(points)):
            current_idx = points[i][0]
            last_idx = filtered[-1][0]
            
            # Only keep if minimum bars between points
            if current_idx - last_idx >= self.min_bars_between:
                filtered.append(points[i])
        
        return filtered
    
    def get_second_last_swing_high(self, data: pd.DataFrame, lookback: int = 5, vix_data: pd.DataFrame = None) -> Optional[Tuple[int, float, str]]:
        """
        Get the second last swing high
        
        Args:
            data: DataFrame with OHLCV data
            lookback: Number of bars to look back/forward for comparison
            vix_data: Optional VIX data for volatility-adjusted detection
            
        Returns:
            Tuple (index, price, datetime) of second last swing high or None
        """
        swing_highs = self.detect_swing_highs(data, lookback, vix_data)
        
        if len(swing_highs) >= 2:
            return swing_highs[-2]
        if len(swing_highs) == 1:
            # Fallback: if only one exists, return the latest
            return swing_highs[-1]
            return None
    
    def get_second_last_swing_low(self, data: pd.DataFrame, lookback: int = 5, vix_data: pd.DataFrame = None) -> Optional[Tuple[int, float, str]]:
        """
        Get the second last swing low
        
        Args:
            data: DataFrame with OHLCV data
            lookback: Number of bars to look back/forward for comparison
            vix_data: Optional VIX data for volatility-adjusted detection
            
        Returns:
            Tuple (index, price, datetime) of second last swing low or None
        """
        swing_lows = self.detect_swing_lows(data, lookback, vix_data)
        
        if len(swing_lows) >= 2:
            return swing_lows[-2]
        if len(swing_lows) == 1:
            # Fallback: if only one exists, return the latest
            return swing_lows[-1]
            return None
    
    def get_latest_swing_points(self, data: pd.DataFrame, lookback: int = 5) -> Dict[str, Optional[Tuple[int, float]]]:
        """
        Get the latest swing high and low
        
        Args:
            data: DataFrame with OHLCV data
            lookback: Number of bars to look back/forward for comparison
            
        Returns:
            Dictionary with 'swing_high' and 'swing_low' keys
        """
        swing_highs = self.detect_swing_highs(data, lookback)
        swing_lows = self.detect_swing_lows(data, lookback)
        
        result = {
            'swing_high': swing_highs[-1] if swing_highs else None,
            'swing_low': swing_lows[-1] if swing_lows else None
        }
        
        return result
    
    def get_second_last_swing_points(self, data: pd.DataFrame, lookback: int = 5) -> Dict[str, Optional[Tuple[int, float]]]:
        """
        Get the second last swing high and low
        
        Args:
            data: DataFrame with OHLCV data
            lookback: Number of bars to look back/forward for comparison
            
        Returns:
            Dictionary with 'swing_high' and 'swing_low' keys
        """
        swing_high = self.get_second_last_swing_high(data, lookback)
        swing_low = self.get_second_last_swing_low(data, lookback)
        
        return {
            'swing_high': swing_high,
            'swing_low': swing_low
        }
    
    def detect_trend(self, data: pd.DataFrame, lookback: int = 5) -> str:
        """
        Detect trend using higher-highs and lower-lows analysis
        
        Analyzes price action to identify:
        - Uptrend: Higher-highs and higher-lows pattern
        - Downtrend: Lower-lows and lower-highs pattern  
        - Sideways: No clear pattern, oscillating within range
        
        Args:
            data: DataFrame with OHLCV data
            lookback: Number of bars to look back/forward for comparison
            
        Returns:
            'uptrend', 'downtrend', or 'sideways'
        """
        if data.empty or len(data) < lookback * 2:
            return 'sideways'
        
        # Focus on recent data for trend analysis (last 50 bars)
        recent_data = data.tail(min(50, len(data)))
        
        # Get swing points for analysis from recent data
        swing_highs = self.detect_swing_highs(recent_data, lookback)
        swing_lows = self.detect_swing_lows(recent_data, lookback)
        
        # Need at least 2 swing points for trend analysis
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            return self._analyze_swing_trend(swing_highs, swing_lows)
        
        # Fallback: Use recent price action analysis
        return self._analyze_price_action_trend(recent_data, lookback)
    
    def _analyze_swing_trend(self, swing_highs: List[Tuple[int, float]], 
                           swing_lows: List[Tuple[int, float]]) -> str:
        """
        Analyze trend based on swing point patterns using proper higher-highs/lower-lows logic
        
        Args:
            swing_highs: List of (index, price) tuples for swing highs
            swing_lows: List of (index, price) tuples for swing lows
            
        Returns:
            Trend classification
        """
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 'sideways'
        
        # Get the last 2 swing points for clear trend identification
        last_high = swing_highs[-1][1]
        second_last_high = swing_highs[-2][1] if len(swing_highs) >= 2 else last_high
        last_low = swing_lows[-1][1]
        second_last_low = swing_lows[-2][1] if len(swing_lows) >= 2 else last_low
        
        # Check for uptrend: Higher high AND higher low
        higher_high = last_high > second_last_high
        higher_low = last_low > second_last_low
        
        # Check for downtrend: Lower high AND lower low  
        lower_high = last_high < second_last_high
        lower_low = last_low < second_last_low
        
        # Simple and clear trend classification
        if higher_high and higher_low:
            return 'uptrend'
        elif lower_high and lower_low:
            return 'downtrend'
        else:
            # For mixed patterns, check the most recent significant move
            # If we have more recent data, check the overall direction
            if len(swing_highs) >= 3 and len(swing_lows) >= 3:
                # Check the last 3 swing points for overall direction
                third_last_high = swing_highs[-3][1]
                third_last_low = swing_lows[-3][1]
                
                # Check if there's a clear progression over 3 points
                uptrend_progression = (last_high > second_last_high > third_last_high and 
                                     last_low > second_last_low > third_last_low)
                downtrend_progression = (last_high < second_last_high < third_last_high and 
                                       last_low < second_last_low < third_last_low)
                
                if uptrend_progression:
                    return 'uptrend'
                elif downtrend_progression:
                    return 'downtrend'
            
            return 'sideways'
    
    def _analyze_price_action_trend(self, data: pd.DataFrame, lookback: int) -> str:
        """
        Analyze trend using price action when swing points are insufficient
        
        Args:
            data: DataFrame with OHLCV data
            lookback: Number of bars to analyze
            
        Returns:
            Trend classification
        """
        if len(data) < 10:
            return 'sideways'
        
        # Use recent price data for analysis (last 20-30 bars)
        recent_data = data.tail(min(30, len(data)))
        
        # Find local highs and lows in recent data
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        
        # Find significant highs and lows (simplified swing detection)
        local_highs = []
        local_lows = []
        
        # Find local highs
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1] and highs[i] > highs[i-2] and highs[i] > highs[i+2]:
                local_highs.append(highs[i])
        
        # Find local lows
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1] and lows[i] < lows[i-2] and lows[i] < lows[i+2]:
                local_lows.append(lows[i])
        
        # Need at least 2 local highs and 2 local lows for trend analysis
        if len(local_highs) < 2 or len(local_lows) < 2:
            # Fallback: Use simple price direction
            start_price = recent_data['close'].iloc[0]
            end_price = recent_data['close'].iloc[-1]
            price_change = (end_price - start_price) / start_price
            
            if price_change > 0.02:  # 2% increase
                return 'uptrend'
            elif price_change < -0.02:  # 2% decrease
                return 'downtrend'
            else:
                return 'sideways'
        
        # Analyze the last 2 local highs and lows
        last_high = local_highs[-1]
        second_last_high = local_highs[-2] if len(local_highs) >= 2 else last_high
        last_low = local_lows[-1]
        second_last_low = local_lows[-2] if len(local_lows) >= 2 else last_low
        
        # Check for uptrend: Higher high AND higher low
        higher_high = last_high > second_last_high
        higher_low = last_low > second_last_low
        
        # Check for downtrend: Lower high AND lower low
        lower_high = last_high < second_last_high
        lower_low = last_low < second_last_low
        
        # Simple and clear trend classification
        if higher_high and higher_low:
            return 'uptrend'
        elif lower_high and lower_low:
            return 'downtrend'
        else:
            return 'sideways'
    
    def get_detailed_trend_analysis(self, data: pd.DataFrame, lookback: int = 5) -> Dict:
        """
        Get detailed trend analysis with explanations
        
        Args:
            data: DataFrame with OHLCV data
            lookback: Number of bars to analyze
            
        Returns:
            Dictionary with detailed trend analysis
        """
        if data.empty or len(data) < lookback * 2:
            return {
                'trend': 'sideways',
                'confidence': 0.0,
                'explanation': 'Insufficient data for trend analysis',
                'swing_highs': [],
                'swing_lows': [],
                'trend_indicators': {}
            }
        
        # Get swing points
        swing_highs = self.detect_swing_highs(data, lookback)
        swing_lows = self.detect_swing_lows(data, lookback)
        
        # Analyze trend
        trend = self.detect_trend(data, lookback)
        
        # Calculate confidence and get detailed indicators
        confidence, indicators = self._calculate_trend_confidence(swing_highs, swing_lows, data, lookback)
        
        # Generate explanation
        explanation = self._generate_trend_explanation(trend, swing_highs, swing_lows, indicators)
        
        return {
            'trend': trend,
            'confidence': confidence,
            'explanation': explanation,
            'swing_highs': swing_highs,
            'swing_lows': swing_lows,
            'trend_indicators': indicators
        }
    
    def _calculate_trend_confidence(self, swing_highs: List[Tuple[int, float]], 
                                  swing_lows: List[Tuple[int, float]], 
                                  data: pd.DataFrame, lookback: int) -> Tuple[float, Dict]:
        """
        Calculate trend confidence and indicators
        
        Returns:
            Tuple of (confidence_score, indicators_dict)
        """
        indicators = {
            'higher_highs_count': 0,
            'higher_lows_count': 0,
            'lower_highs_count': 0,
            'lower_lows_count': 0,
            'swing_point_count': len(swing_highs) + len(swing_lows),
            'price_range': 0.0,
            'volatility': 0.0
        }
        
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # Analyze swing point patterns
            for i in range(1, len(swing_highs)):
                if swing_highs[i][1] > swing_highs[i-1][1]:
                    indicators['higher_highs_count'] += 1
                elif swing_highs[i][1] < swing_highs[i-1][1]:
                    indicators['lower_highs_count'] += 1
            
            for i in range(1, len(swing_lows)):
                if swing_lows[i][1] > swing_lows[i-1][1]:
                    indicators['higher_lows_count'] += 1
                elif swing_lows[i][1] < swing_lows[i-1][1]:
                    indicators['lower_lows_count'] += 1
        
        # Calculate price range and volatility
        if not data.empty:
            indicators['price_range'] = (data['high'].max() - data['low'].min()) / data['close'].mean()
            indicators['volatility'] = data['close'].pct_change().std()
        
        # Calculate confidence based on pattern consistency
        total_swing_changes = (indicators['higher_highs_count'] + indicators['lower_highs_count'] + 
                             indicators['higher_lows_count'] + indicators['lower_lows_count'])
        
        if total_swing_changes == 0:
            confidence = 0.3  # Low confidence for sideways
        else:
            # Higher confidence for consistent patterns
            pattern_consistency = max(
                indicators['higher_highs_count'] + indicators['higher_lows_count'],
                indicators['lower_highs_count'] + indicators['lower_lows_count']
            ) / total_swing_changes
            
            confidence = min(0.9, 0.3 + pattern_consistency * 0.6)
        
        return confidence, indicators
    
    def _generate_trend_explanation(self, trend: str, swing_highs: List[Tuple[int, float]], 
                                  swing_lows: List[Tuple[int, float]], indicators: Dict) -> str:
        """
        Generate human-readable trend explanation
        
        Args:
            trend: Detected trend
            swing_highs: List of swing highs
            swing_lows: List of swing lows
            indicators: Trend indicators dictionary
            
        Returns:
            Explanation string
        """
        if trend == 'uptrend':
            return (f"Uptrend detected: {indicators['higher_highs_count']} higher highs and "
                   f"{indicators['higher_lows_count']} higher lows found. Price is forming "
                   f"a pattern of higher highs and higher lows, indicating bullish momentum.")
        
        elif trend == 'downtrend':
            return (f"Downtrend detected: {indicators['lower_highs_count']} lower highs and "
                   f"{indicators['lower_lows_count']} lower lows found. Price is forming "
                   f"a pattern of lower highs and lower lows, indicating bearish momentum.")
        
        else:  # sideways
            return (f"Sideways/Consolidation detected: Mixed pattern with "
                   f"{indicators['higher_highs_count']} higher highs, {indicators['lower_highs_count']} lower highs, "
                   f"{indicators['higher_lows_count']} higher lows, and {indicators['lower_lows_count']} lower lows. "
                   f"Price is oscillating within a range without clear directional bias.")
    
    def get_swing_analysis(self, data: pd.DataFrame, lookback: int = 5, swing_high_index: int = -2, swing_low_index: int = -2, vix_data: pd.DataFrame = None) -> Dict:
        """
        Get comprehensive swing analysis with selectable swing points
        
        Args:
            data: DataFrame with OHLCV data
            lookback: Number of bars to look back/forward for comparison
            swing_high_index: Index of swing high to use (-1 for last, -2 for second last, etc.)
            swing_low_index: Index of swing low to use (-1 for last, -2 for second last, etc.)
            vix_data: Optional VIX data for volatility-adjusted detection
            
        Returns:
            Dictionary with swing analysis results
        """
        swing_highs = self.detect_swing_highs(data, lookback, vix_data)
        swing_lows = self.detect_swing_lows(data, lookback, vix_data)
        
        # Get selected swing points
        selected_high = self.get_swing_high_by_index(data, lookback, swing_high_index, vix_data)
        selected_low = self.get_swing_low_by_index(data, lookback, swing_low_index, vix_data)
        
        # Keep backward compatibility
        second_last_high = self.get_second_last_swing_high(data, lookback, vix_data)
        second_last_low = self.get_second_last_swing_low(data, lookback, vix_data)
        
        trend = self.detect_trend(data, lookback)
        
        return {
            'swing_highs': swing_highs,
            'swing_lows': swing_lows,
            'selected_high': selected_high,
            'selected_low': selected_low,
            'second_last_high': second_last_high,
            'second_last_low': second_last_low,
            'trend': trend,
            'total_swing_highs': len(swing_highs),
            'total_swing_lows': len(swing_lows),
            'swing_high_index': swing_high_index,
            'swing_low_index': swing_low_index
        }
    
    def get_swing_high_by_index(self, data: pd.DataFrame, lookback: int = 5, index: int = -2, vix_data: pd.DataFrame = None) -> Optional[Tuple[int, float, str]]:
        """
        Get swing high by index
        
        Args:
            data: DataFrame with OHLCV data
            lookback: Number of bars to look back/forward for comparison
            index: Index of swing high (-1 for last, -2 for second last, etc.)
            vix_data: Optional VIX data for volatility-adjusted detection
            
        Returns:
            Tuple (index, price, datetime) of selected swing high or None
        """
        swing_highs = self.detect_swing_highs(data, lookback, vix_data)
        
        if len(swing_highs) == 0:
            return None
        
        try:
            return swing_highs[index]
        except IndexError:
            return None
    
    def get_swing_low_by_index(self, data: pd.DataFrame, lookback: int = 5, index: int = -2, vix_data: pd.DataFrame = None) -> Optional[Tuple[int, float, str]]:
        """
        Get swing low by index
        
        Args:
            data: DataFrame with OHLCV data
            lookback: Number of bars to look back/forward for comparison
            index: Index of swing low (-1 for last, -2 for second last, etc.)
            vix_data: Optional VIX data for volatility-adjusted detection
            
        Returns:
            Tuple (index, price, datetime) of selected swing low or None
        """
        swing_lows = self.detect_swing_lows(data, lookback, vix_data)
        
        if len(swing_lows) == 0:
            return None
        
        try:
            return swing_lows[index]
        except IndexError:
            return None
    
    def detect_intraday_swing_highs(self, data: pd.DataFrame, lookback: int = 5) -> List[Tuple[int, float]]:
        """
        Detect swing highs only within the current day
        
        Args:
            data: DataFrame with OHLCV data
            lookback: Number of bars to look back/forward for comparison
            
        Returns:
            List of tuples (index, price) for intraday swing highs
        """
        if data.empty or len(data) < lookback * 2:
            return []
        
        # Filter data to current day only
        current_date = pd.Timestamp.now().date()
        if 'datetime' in data.columns:
            intraday_data = data[data['datetime'].dt.date == current_date].copy()
        else:
            # If no datetime column, assume all data is from current day
            intraday_data = data.copy()
        
        if intraday_data.empty:
            logger.warning("No intraday data available for swing detection")
            return []
        
        # Adjust indices to match original data
        intraday_swing_highs = self.detect_swing_highs(intraday_data, lookback)
        
        # Map back to original data indices
        if 'datetime' in data.columns:
            original_indices = data[data['datetime'].dt.date == current_date].index.tolist()
            mapped_swing_highs = []
            for idx, price in intraday_swing_highs:
                if idx < len(original_indices):
                    mapped_swing_highs.append((original_indices[idx], price))
            return mapped_swing_highs
        else:
            return intraday_swing_highs
    
    def detect_intraday_swing_lows(self, data: pd.DataFrame, lookback: int = 5) -> List[Tuple[int, float]]:
        """
        Detect swing lows only within the current day
        
        Args:
            data: DataFrame with OHLCV data
            lookback: Number of bars to look back/forward for comparison
            
        Returns:
            List of tuples (index, price) for intraday swing lows
        """
        if data.empty or len(data) < lookback * 2:
            return []
        
        # Filter data to current day only
        current_date = pd.Timestamp.now().date()
        if 'datetime' in data.columns:
            intraday_data = data[data['datetime'].dt.date == current_date].copy()
        else:
            # If no datetime column, assume all data is from current day
            intraday_data = data.copy()
        
        if intraday_data.empty:
            logger.warning("No intraday data available for swing detection")
            return []
        
        # Adjust indices to match original data
        intraday_swing_lows = self.detect_swing_lows(intraday_data, lookback)
        
        # Map back to original data indices
        if 'datetime' in data.columns:
            original_indices = data[data['datetime'].dt.date == current_date].index.tolist()
            mapped_swing_lows = []
            for idx, price in intraday_swing_lows:
                if idx < len(original_indices):
                    mapped_swing_lows.append((original_indices[idx], price))
            return mapped_swing_lows
        else:
            return intraday_swing_lows
    
    def detect_trend_based_swings(self, data: pd.DataFrame, lookback: int = 5, intraday_only: bool = True, vix_data: pd.DataFrame = None, time_interval_minutes: int = None) -> Dict[str, Optional[Tuple[int, float, str]]]:
        """
        Detect swing points based on trend direction:
        - For downtrends: First find swing high, then swing low on further candles
        - For uptrends: First find swing low, then swing high on further candles
        
        Args:
            data: DataFrame with OHLCV data
            lookback: Number of bars to look back/forward for comparison
            intraday_only: If True, only consider swing points from current day
            vix_data: Optional VIX data for volatility-adjusted detection
            
        Returns:
            Dictionary with trend-based swing analysis
        """
        if data.empty or len(data) < lookback * 2:
            return {
                'trend': 'insufficient_data',
                'swing_high': None,
                'swing_low': None,
                'trend_swing_high': None,
                'trend_swing_low': None
            }
        
        # Detect swing points (intraday or all data)
        if intraday_only:
            swing_highs = self.detect_intraday_swing_highs(data, lookback)
            swing_lows = self.detect_intraday_swing_lows(data, lookback)
        else:
            swing_highs = self.detect_swing_highs(data, lookback, vix_data)
            swing_lows = self.detect_swing_lows(data, lookback, vix_data)
        
        # Determine trend
        trend = self.detect_trend(data, lookback)
        
        result = {
            'trend': trend,
            'swing_high': swing_highs[-1] if swing_highs else None,
            'swing_low': swing_lows[-1] if swing_lows else None,
            'swing_highs': swing_highs,  # Include all swing highs
            'swing_lows': swing_lows,    # Include all swing lows
            'trend_swing_high': None,
            'trend_swing_low': None
        }
        
        if trend == 'downtrend':
            # For downtrend: First find swing high, then swing low on further candles
            result['trend_swing_high'] = self._find_first_swing_high_in_downtrend(data, swing_highs, lookback)
            result['trend_swing_low'] = self._find_swing_low_after_high_in_downtrend(data, swing_lows, result['trend_swing_high'], lookback)
            
        elif trend == 'uptrend':
            # For uptrend: First find swing low, then swing high on further candles
            result['trend_swing_low'] = self._find_first_swing_low_in_uptrend(data, swing_lows, lookback)
            result['trend_swing_high'] = self._find_swing_high_after_low_in_uptrend(data, swing_highs, result['trend_swing_low'], lookback)
        
        return result
    
    def find_swings_in_time_window(self, data: pd.DataFrame, time_interval_minutes: int, vix_data: pd.DataFrame = None, market_hours_manager=None) -> Dict[str, Optional[Tuple[int, float, str]]]:
        """
        Find swing points within a specific time window from current time
        
        Args:
            data: DataFrame with OHLCV data
            time_interval_minutes: Minutes to look back from current time
            vix_data: Optional VIX data for volatility-adjusted detection
            market_hours_manager: Optional market hours manager for handling market hours
            
        Returns:
            Dictionary with swing analysis within time window
        """
        if data.empty or 'Datetime' not in data.columns:
            return {
                'swing_high': None,
                'swing_low': None,
                'time_window_start': None,
                'time_window_end': None
            }
        
        # Get current time (last data point)
        current_time = pd.to_datetime(data['Datetime'].iloc[-1])
        
        # Use market hours manager if provided
        if market_hours_manager is not None:
            # Get effective time considering market hours
            effective_time = market_hours_manager.get_effective_time_for_analysis(
                current_time.to_pydatetime(), time_interval_minutes
            )
            effective_time = pd.to_datetime(effective_time)
            
            # Calculate target time for analysis
            target_time = market_hours_manager.get_target_time_for_analysis(
                current_time.to_pydatetime(), time_interval_minutes
            )
            target_time = pd.to_datetime(target_time)
            
            # Use target time as window start
            window_start = target_time
        else:
            # Fallback to original logic
            window_start = current_time - pd.Timedelta(minutes=time_interval_minutes)
            target_time = current_time - pd.Timedelta(minutes=time_interval_minutes)
        
        # Filter data within time window
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        window_data = data[data['Datetime'] >= window_start].copy()

        # If no candles in current day window, fallback to previous trading day
        if window_data.empty and market_hours_manager is not None:
            try:
                # Try up to 5 previous market days
                for back_days in range(1, 6):
                    prev_ref_time = current_time - pd.Timedelta(days=back_days)
                    # Use previous day's market close as effective time
                    prev_close = pd.to_datetime(
                        market_hours_manager.get_last_market_close(prev_ref_time.to_pydatetime())
                    )
                    prev_window_start = market_hours_manager.get_target_time_for_analysis(
                        prev_close.to_pydatetime(), time_interval_minutes
                    )
                    prev_window_start = pd.to_datetime(prev_window_start)

                    window_data = data[(data['Datetime'] >= prev_window_start) & (data['Datetime'] <= prev_close)].copy()
                    if not window_data.empty:
                        window_start = prev_window_start
                        current_time = prev_close
                        break
            except Exception:
                # If anything goes wrong, keep window_data as is (likely empty)
                pass

        if window_data.empty:
            return {
                'swing_high': None,
                'swing_low': None,
                'time_window_start': str(window_start),
                'time_window_end': str(current_time)
            }
        
        # Find swing points in the time window
        swing_highs = self.detect_swing_highs(window_data, lookback=3, vix_data=vix_data)
        swing_lows = self.detect_swing_lows(window_data, lookback=3, vix_data=vix_data)
        
        # target_time is already calculated above
        
        nearest_high = self._find_nearest_swing_to_time(swing_highs, target_time, data)
        nearest_low = self._find_nearest_swing_to_time(swing_lows, target_time, data)
        
        return {
            'swing_high': nearest_high,
            'swing_low': nearest_low,
            'time_window_start': str(window_start),
            'time_window_end': str(current_time),
            'target_time': str(target_time)
        }
    
    def _find_nearest_swing_to_time(self, swing_points: List[Tuple[int, float, str]], target_time: pd.Timestamp, data: pd.DataFrame) -> Optional[Tuple[int, float, str]]:
        """
        Find the swing point closest to the target time
        
        Args:
            swing_points: List of swing points (index, price, datetime)
            target_time: Target time to find nearest swing
            data: Original data for index mapping
            
        Returns:
            Nearest swing point or None
        """
        if not swing_points:
            return None
        
        # Convert target time to pandas timestamp
        target_time = pd.to_datetime(target_time)
        
        min_diff = float('inf')
        nearest_swing = None
        
        for swing in swing_points:
            swing_time = pd.to_datetime(swing[2])
            time_diff = abs((swing_time - target_time).total_seconds())
            
            if time_diff < min_diff:
                min_diff = time_diff
                nearest_swing = swing
        
        return nearest_swing
    
    def _find_first_swing_high_in_downtrend(self, data: pd.DataFrame, swing_highs: List[Tuple[int, float, str]], lookback: int) -> Optional[Tuple[int, float, str]]:
        """
        Find the first significant swing high in a downtrend
        """
        if not swing_highs:
            # If no swing highs found, try with smaller lookback
            swing_highs = self.detect_swing_highs(data, lookback=1)
            if not swing_highs:
                return None
        
        # Look for the most recent swing high that could be the start of downtrend
        # This would typically be the highest point before the decline
        return swing_highs[-1] if swing_highs else None
    
    def _find_swing_low_after_high_in_downtrend(self, data: pd.DataFrame, swing_lows: List[Tuple[int, float, str]], swing_high: Optional[Tuple[int, float, str]], lookback: int) -> Optional[Tuple[int, float, str]]:
        """
        Find swing low that occurs after the swing high in a downtrend
        """
        if not swing_high:
            return None
            
        if not swing_lows:
            # If no swing lows found, try with smaller lookback
            swing_lows = self.detect_swing_lows(data, lookback=1)
            if not swing_lows:
                return None
        
        swing_high_index = swing_high[0]
        
        # Find the first swing low that occurs after the swing high
        for swing_low in reversed(swing_lows):  # Start from most recent
            if swing_low[0] > swing_high_index:
                return swing_low
        
        return None
    
    def _find_first_swing_low_in_uptrend(self, data: pd.DataFrame, swing_lows: List[Tuple[int, float, str]], lookback: int) -> Optional[Tuple[int, float, str]]:
        """
        Find the first significant swing low in an uptrend
        """
        if not swing_lows:
            # If no swing lows found, try with smaller lookback
            swing_lows = self.detect_swing_lows(data, lookback=1)
            if not swing_lows:
                return None
        
        # Look for the most recent swing low that could be the start of uptrend
        # This would typically be the lowest point before the rise
        return swing_lows[-1] if swing_lows else None
    
    def _find_swing_high_after_low_in_uptrend(self, data: pd.DataFrame, swing_highs: List[Tuple[int, float, str]], swing_low: Optional[Tuple[int, float, str]], lookback: int) -> Optional[Tuple[int, float, str]]:
        """
        Find swing high that occurs after the swing low in an uptrend
        """
        if not swing_low:
            return None
            
        if not swing_highs:
            # If no swing highs found, try with smaller lookback
            swing_highs = self.detect_swing_highs(data, lookback=1)
            if not swing_highs:
                return None
        
        swing_low_index = swing_low[0]
        
        # Find the first swing high that occurs after the swing low
        for swing_high in reversed(swing_highs):  # Start from most recent
            if swing_high[0] > swing_low_index:
                return swing_high
        
        return None

# Convenience functions
def detect_swing_highs(data: pd.DataFrame, lookback: int = 5, min_bars: int = 3) -> List[Tuple[int, float]]:
    """Convenience function to detect swing highs"""
    detector = SwingDetector(min_bars)
    return detector.detect_swing_highs(data, lookback)

def detect_swing_lows(data: pd.DataFrame, lookback: int = 5, min_bars: int = 3) -> List[Tuple[int, float]]:
    """Convenience function to detect swing lows"""
    detector = SwingDetector(min_bars)
    return detector.detect_swing_lows(data, lookback)

def get_swing_analysis(data: pd.DataFrame, lookback: int = 5, min_bars: int = 3) -> Dict:
    """Convenience function to get swing analysis"""
    detector = SwingDetector(min_bars)
    return detector.get_swing_analysis(data, lookback)

def detect_trend_based_swings(data: pd.DataFrame, lookback: int = 5, min_bars: int = 3, intraday_only: bool = True) -> Dict:
    """Convenience function to detect trend-based swings"""
    detector = SwingDetector(min_bars)
    return detector.detect_trend_based_swings(data, lookback, intraday_only)
