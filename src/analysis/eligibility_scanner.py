"""
Eligibility scanning and filtering logic for swing trade candidates
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime

from .swing_detector import SwingDetector
from .fibonacci_analyzer import FibonacciAnalyzer
from ..data.data_fetcher import DataFetcher
from ..data.stock_loader import StockLoader
from ..utils.market_hours import create_market_hours_manager

logger = logging.getLogger(__name__)

class EligibilityScanner:
    """Scans stocks for swing trade eligibility based on Fibonacci analysis"""
    
    def __init__(self, config):
        """
        Initialize eligibility scanner
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.swing_detector = SwingDetector(config.analysis.swing_minimum_bars)
        self.fibonacci_analyzer = FibonacciAnalyzer(config.analysis.fibonacci_levels)
        self.data_fetcher = DataFetcher(config)
        self.stock_loader = StockLoader()
        self.market_hours_manager = create_market_hours_manager(config)
    
    def analyze_single_stock(self, symbol: str, current_price: float = None) -> Dict[str, Any]:
        """
        Analyze a single stock for swing trade eligibility
        
        Args:
            symbol: Stock symbol
            current_price: Current price (fetched if not provided)
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Fetch data - get full data for swing analysis
            data = self.data_fetcher.fetch_stock_data(symbol)
            
            if data.empty:
                return self._create_error_result(symbol, "No data available")
            
            # Get current price if not provided
            if current_price is None:
                try:
                    # Use the data fetcher's method to get current price
                    prices_dict = self.data_fetcher.get_current_prices([symbol])
                    current_price = prices_dict.get(symbol, 0)
                    if current_price <= 0:
                        return self._create_error_result(symbol, "Invalid current price")
                except Exception as e:
                    return self._create_error_result(symbol, f"Error getting current price: {e}")
            
            # Detect swing points
            swing_analysis = self.swing_detector.get_swing_analysis(data, self.config.analysis.lookback_period)
            
            # Use selected swing points if available, otherwise fall back to second last
            selected_high = swing_analysis.get('selected_high') or swing_analysis.get('second_last_high')
            selected_low = swing_analysis.get('selected_low') or swing_analysis.get('second_last_low')
            
            if not selected_high or not selected_low:
                return self._create_error_result(symbol, "Insufficient swing points")
            
            # Calculate Fibonacci levels
            fib_analysis = self.fibonacci_analyzer.get_comprehensive_analysis(
                selected_high,
                selected_low,
                swing_analysis['trend'],
                current_price,
                self.config.analysis.eligibility_min_level,
                self.config.analysis.eligibility_max_level
            )
            
            # Create result
            result = {
                'symbol': symbol,
                'current_price': current_price,
                'swing_high': selected_high[1],
                'swing_low': selected_low[1],
                'swing_high_datetime': selected_high[2] if len(selected_high) > 2 else '',
                'swing_low_datetime': selected_low[2] if len(selected_low) > 2 else '',
                'trend': swing_analysis['trend'],
                'fibonacci_levels': fib_analysis['fibonacci_levels'],
                'eligible': fib_analysis['eligibility']['eligible'],
                'eligibility_details': fib_analysis['eligibility'],
                'analysis_timestamp': datetime.now(),
                'error': None,
                'swing_high_index': swing_analysis.get('swing_high_index', -2),
                'swing_low_index': swing_analysis.get('swing_low_index', -2)
            }
            
            # Add specific Fibonacci levels for display
            fib_levels = fib_analysis['fibonacci_levels']
            result.update({
                'fib_1618': fib_levels.get('1.618', 0),
                'fib_0786': fib_levels.get('0.786', 0),
                'fib_0500': fib_levels.get('0.500', 0),
                'fib_0382': fib_levels.get('0.382', 0),
                'fib_0236': fib_levels.get('0.236', 0)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return self._create_error_result(symbol, str(e))
    
    def analyze_single_stock_with_swing_selection(self, symbol: str, swing_high_index: int = -2, swing_low_index: int = -2, current_price: float = None, time_interval_minutes: int = None) -> Dict[str, Any]:
        """
        Analyze a single stock with custom swing point selection
        
        Args:
            symbol: Stock symbol
            swing_high_index: Index of swing high to use (-1 for last, -2 for second last, etc.)
            swing_low_index: Index of swing low to use (-1 for last, -2 for second last, etc.)
            current_price: Current price (optional, will be fetched if not provided)
            time_interval_minutes: Time interval in minutes to look back for swing points
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Fetch data - get full data for swing analysis
            data = self.data_fetcher.fetch_stock_data(symbol)
            
            if data.empty:
                return self._create_error_result(symbol, "No data available")
            
            # Get current price if not provided
            if current_price is None:
                try:
                    # Use the data fetcher's method to get current price
                    prices_dict = self.data_fetcher.get_current_prices([symbol])
                    current_price = prices_dict.get(symbol, 0)
                    if current_price <= 0:
                        return self._create_error_result(symbol, "Invalid current price")
                except Exception as e:
                    return self._create_error_result(symbol, f"Error getting current price: {e}")
            
            # Detect swing points based on time interval or traditional method
            if time_interval_minutes is not None:
                # Use time-based swing detection with market hours consideration
                swing_analysis = self.swing_detector.find_swings_in_time_window(
                    data, 
                    time_interval_minutes,
                    vix_data=None,  # Remove VIX dependency
                    market_hours_manager=self.market_hours_manager
                )
            else:
                # Use traditional swing detection with custom selection
                swing_analysis = self.swing_detector.detect_trend_based_swings(
                    data, 
                    self.config.analysis.lookback_period,
                    intraday_only=False,  # Use all available data for better swing detection
                    vix_data=None  # Remove VIX dependency
                )
            
            # Get detailed trend analysis for better insights
            detailed_trend = self.swing_detector.get_detailed_trend_analysis(
                data, 
                self.config.analysis.lookback_period
            )
            
            # Select swing points based on method used
            if time_interval_minutes is not None:
                # For time-based detection, use the found swings directly
                selected_high = swing_analysis.get('swing_high')
                selected_low = swing_analysis.get('swing_low')
                
                if not selected_high or not selected_low:
                    return self._create_error_result(symbol, f"No swing points found within {time_interval_minutes} minutes")
                
                # Validate that swing high is actually higher than swing low
                if selected_high[1] <= selected_low[1]:
                    return self._create_error_result(symbol, "Invalid swing point pair: high <= low")
            else:
                # Use trend-based swing points (all data)
                selected_high = swing_analysis.get('trend_swing_high')
                selected_low = swing_analysis.get('trend_swing_low')
                
                # If trend-based swing points are not available, use regular swing points
                if not selected_high or not selected_low:
                    swing_highs = swing_analysis.get('swing_highs', [])
                    swing_lows = swing_analysis.get('swing_lows', [])
                    
                    if not swing_highs or not swing_lows:
                        return self._create_error_result(symbol, "Insufficient swing points")
                    
                    # Use the most recent swing points, but ensure high > low
                    selected_high = swing_highs[-1] if swing_highs else None
                    selected_low = swing_lows[-1] if swing_lows else None
                    
                    # Validate that swing high is actually higher than swing low
                    if selected_high and selected_low and selected_high[1] <= selected_low[1]:
                        # Try to find a valid pair by looking at different combinations
                        valid_pair_found = False
                        for high in reversed(swing_highs):
                            for low in reversed(swing_lows):
                                if high[1] > low[1]:  # High should be higher than low
                                    selected_high = high
                                    selected_low = low
                                    valid_pair_found = True
                                    break
                            if valid_pair_found:
                                break
                        
                        if not valid_pair_found:
                            return self._create_error_result(symbol, "No valid swing point pairs found")
                    
                    if not selected_high or not selected_low:
                        return self._create_error_result(symbol, "Insufficient swing points")
            
            # Calculate Fibonacci levels
            fib_analysis = self.fibonacci_analyzer.get_comprehensive_analysis(
                selected_high,
                selected_low,
                swing_analysis['trend'],
                current_price,
                self.config.analysis.eligibility_min_level,
                self.config.analysis.eligibility_max_level
            )
            
            # Create result
            result = {
                'symbol': symbol,
                'current_price': current_price,
                'swing_high': selected_high[1],
                'swing_low': selected_low[1],
                'swing_high_datetime': selected_high[2] if len(selected_high) > 2 else '',
                'swing_low_datetime': selected_low[2] if len(selected_low) > 2 else '',
                'trend': swing_analysis['trend'],
                'trend_confidence': detailed_trend.get('confidence', 0.0),
                'trend_explanation': detailed_trend.get('explanation', ''),
                'fibonacci_levels': fib_analysis['fibonacci_levels'],
                'eligible': fib_analysis['eligibility']['eligible'],
                'eligibility_details': fib_analysis['eligibility'],
                'analysis_timestamp': datetime.now(),
                'error': None,
                'swing_high_index': swing_high_index,
                'swing_low_index': swing_low_index,
                'available_swing_highs': len(swing_analysis['swing_highs']),
                'available_swing_lows': len(swing_analysis['swing_lows']),
                'trend_indicators': detailed_trend.get('trend_indicators', {})
            }
            
            # Add specific Fibonacci levels for display
            fib_levels = fib_analysis['fibonacci_levels']
            result.update({
                'fib_1618': fib_levels.get('1.618', 0),
                'fib_0786': fib_levels.get('0.786', 0),
                'fib_0500': fib_levels.get('0.500', 0),
                'fib_0382': fib_levels.get('0.382', 0),
                'fib_0236': fib_levels.get('0.236', 0)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return self._create_error_result(symbol, str(e))
    
    def scan_multiple_stocks(self, symbols: List[str], current_prices: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """
        Scan multiple stocks for eligibility
        
        Args:
            symbols: List of stock symbols
            current_prices: Dictionary of current prices (optional)
            
        Returns:
            List of analysis results
        """
        results = []
        
        for symbol in symbols:
            current_price = current_prices.get(symbol) if current_prices else None
            result = self.analyze_single_stock(symbol, current_price)
            results.append(result)
        
        return results
    
    def scan_stocks_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Scan stocks from a file
        
        Args:
            file_path: Path to stock list file
            
        Returns:
            List of analysis results
        """
        try:
            # Load stock symbols
            symbols = self.data_fetcher.stock_loader.load_stocks(file_path)
            symbols = self.data_fetcher.stock_loader.validate_symbols(symbols)
            
            logger.info(f"Scanning {len(symbols)} stocks from {file_path}")
            
            # Get current prices
            current_prices = self.data_fetcher.get_current_prices(symbols)
            
            # Scan all stocks
            return self.scan_multiple_stocks(symbols, current_prices)
            
        except Exception as e:
            logger.error(f"Error scanning stocks from file {file_path}: {e}")
            return []
    
    def scan_stocks_from_file_with_swing_selection(self, file_path: str, swing_high_index: int = -2, swing_low_index: int = -2) -> List[Dict[str, Any]]:
        """
        Scan stocks from a file with custom swing point selection
        
        Args:
            file_path: Path to stock list file
            swing_high_index: Index of swing high to use
            swing_low_index: Index of swing low to use
            
        Returns:
            List of analysis results
        """
        try:
            # Load stock symbols
            symbols = self.data_fetcher.stock_loader.load_stocks(file_path)
            symbols = self.data_fetcher.stock_loader.validate_symbols(symbols)
            
            logger.info(f"Scanning {len(symbols)} stocks from {file_path} with swing selection")
            
            # Get current prices
            current_prices = self.data_fetcher.get_current_prices(symbols)
            
            # Scan all stocks with custom swing selection
            results = []
            for symbol in symbols:
                current_price = current_prices.get(symbol, 0)
                result = self.analyze_single_stock_with_swing_selection(
                    symbol, swing_high_index, swing_low_index, current_price
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error scanning stocks from file {file_path}: {e}")
            return []
    
    def filter_eligible_stocks(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter results to show only eligible stocks
        
        Args:
            results: List of analysis results
            
        Returns:
            List of eligible stocks
        """
        return [result for result in results if result.get('eligible', False)]
    
    def create_results_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a DataFrame from analysis results
        
        Args:
            results: List of analysis results
            
        Returns:
            DataFrame with results
        """
        if not results:
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Select and rename columns for display
        display_columns = {
            'symbol': 'Symbol',
            'current_price': 'Current Price',
            'swing_high': 'Swing High',
            'swing_low': 'Swing Low',
            'trend': 'Trend',
            'fib_1618': '1.618 Level',
            'fib_0786': '0.786 Level',
            'fib_0500': '0.5 Level',
            'eligible': 'Eligible'
        }
        
        # Filter to available columns
        available_columns = {k: v for k, v in display_columns.items() if k in df.columns}
        
        return df[list(available_columns.keys())].rename(columns=available_columns)
    
    def export_results(self, results: List[Dict[str, Any]], file_path: str, 
                      eligible_only: bool = False) -> bool:
        """
        Export results to CSV file
        
        Args:
            results: List of analysis results
            file_path: Path to export file
            eligible_only: Whether to export only eligible stocks
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Filter results if needed
            if eligible_only:
                results = self.filter_eligible_stocks(results)
            
            # Create DataFrame
            df = self.create_results_dataframe(results)
            
            if df.empty:
                logger.warning("No data to export")
                return False
            
            # Export to CSV
            df.to_csv(file_path, index=False)
            logger.info(f"Exported {len(df)} results to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return False
    
    def get_scan_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics from scan results
        
        Args:
            results: List of analysis results
            
        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {
                'total_stocks': 0,
                'eligible_stocks': 0,
                'error_stocks': 0,
                'eligibility_rate': 0.0
            }
        
        total_stocks = len(results)
        eligible_stocks = len([r for r in results if r.get('eligible', False)])
        error_stocks = len([r for r in results if r.get('error') is not None])
        eligibility_rate = (eligible_stocks / total_stocks) * 100 if total_stocks > 0 else 0
        
        # Trend distribution
        trends = [r.get('trend', 'unknown') for r in results if r.get('trend')]
        trend_distribution = pd.Series(trends).value_counts().to_dict()
        
        return {
            'total_stocks': total_stocks,
            'eligible_stocks': eligible_stocks,
            'error_stocks': error_stocks,
            'eligibility_rate': eligibility_rate,
            'trend_distribution': trend_distribution,
            'scan_timestamp': datetime.now()
        }
    
    def _create_error_result(self, symbol: str, error_message: str) -> Dict[str, Any]:
        """Create error result for a stock"""
        return {
            'symbol': symbol,
            'current_price': 0,
            'swing_high': 0,
            'swing_low': 0,
            'trend': 'unknown',
            'fibonacci_levels': {},
            'eligible': False,
            'eligibility_details': {'eligible': False, 'reason': error_message},
            'analysis_timestamp': datetime.now(),
            'error': error_message,
            'fib_1618': 0,
            'fib_0786': 0,
            'fib_0500': 0,
            'fib_0382': 0,
            'fib_0236': 0
        }
    
    def scan_stocks_from_file_with_time_interval(self, file_path: str, time_interval_minutes: int) -> List[Dict[str, Any]]:
        """
        Scan stocks from file using time-based swing detection
        
        Args:
            file_path: Path to file containing stock symbols
            time_interval_minutes: Time interval in minutes to look back for swing points
            
        Returns:
            List of analysis results
        """
        try:
            # Load stock symbols
            symbols = self.stock_loader.load_stocks(file_path)
            
            if not symbols:
                logger.warning("No stock symbols found in file")
                return []
            
            logger.info(f"Scanning {len(symbols)} stocks with {time_interval_minutes} minute time interval")
            
            # Analyze each stock
            results = []
            for symbol in symbols:
                try:
                    # Primary: time-window based swing detection
                    result = self.analyze_single_stock_with_swing_selection(
                        symbol,
                        time_interval_minutes=time_interval_minutes
                    )

                    # Fallback: if no swing points found or error, use standard swing detection
                    if result.get('error') or (result.get('swing_high', 0) == 0 and result.get('swing_low', 0) == 0):
                        fallback = self.analyze_single_stock_with_swing_selection(symbol)
                        # Prefer fallback if it produced valid data
                        if not fallback.get('error') and (fallback.get('swing_high', 0) or fallback.get('swing_low', 0)):
                            result = fallback

                    results.append(result)
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    results.append(self._create_error_result(symbol, str(e)))
            
            logger.info(f"Completed scanning {len(symbols)} stocks")
            return results
            
        except Exception as e:
            logger.error(f"Error scanning stocks from file: {e}")
            return []

# Convenience functions
def scan_stocks_for_eligibility(symbols: List[str], config, current_prices: Dict[str, float] = None) -> List[Dict[str, Any]]:
    """Convenience function to scan stocks for eligibility"""
    scanner = EligibilityScanner(config)
    return scanner.scan_multiple_stocks(symbols, current_prices)

def scan_stocks_from_file(file_path: str, config) -> List[Dict[str, Any]]:
    """Convenience function to scan stocks from file"""
    scanner = EligibilityScanner(config)
    return scanner.scan_stocks_from_file(file_path)

def export_scan_results(results: List[Dict[str, Any]], file_path: str, eligible_only: bool = False) -> bool:
    """Convenience function to export scan results"""
    scanner = EligibilityScanner(None)  # We only need the export functionality
    return scanner.export_results(results, file_path, eligible_only)
