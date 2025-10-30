"""
Market hours utility functions for handling trading hours and market status
"""

import pandas as pd
from datetime import datetime, time, timedelta
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class MarketHoursManager:
    """Manages market hours and trading time calculations"""
    
    def __init__(self, market_config):
        """
        Initialize market hours manager
        
        Args:
            market_config: Market configuration object
        """
        self.market_start_hour = market_config.market_start_hour
        self.market_start_minute = market_config.market_start_minute
        self.market_end_hour = market_config.market_end_hour
        self.market_end_minute = market_config.market_end_minute
        self.market_days = market_config.market_days
    
    def is_market_open(self, current_time: datetime = None) -> bool:
        """
        Check if market is currently open
        
        Args:
            current_time: Time to check (defaults to current time)
            
        Returns:
            True if market is open, False otherwise
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Check if it's a market day
        if current_time.weekday() not in self.market_days:
            return False
        
        # Check if it's within market hours
        market_start = time(self.market_start_hour, self.market_start_minute)
        market_end = time(self.market_end_hour, self.market_end_minute)
        current_time_only = current_time.time()
        
        return market_start <= current_time_only <= market_end
    
    def get_effective_time_for_analysis(self, current_time: datetime = None, time_interval_minutes: int = 30) -> datetime:
        """
        Get the effective time to use for swing analysis based on market hours
        
        Args:
            current_time: Current time (defaults to current time)
            time_interval_minutes: Time interval in minutes to look back
            
        Returns:
            Effective time to use for analysis
        """
        if current_time is None:
            current_time = datetime.now()
        
        # If market is open, use current time
        if self.is_market_open(current_time):
            return current_time
        
        # If market is closed, use the last market close time
        last_market_close = self.get_last_market_close(current_time)
        logger.info(f"Market is closed. Using last market close time: {last_market_close}")
        return last_market_close
    
    def get_last_market_close(self, current_time: datetime = None) -> datetime:
        """
        Get the last market close time
        
        Args:
            current_time: Current time (defaults to current time)
            
        Returns:
            Last market close datetime
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Create market close time for today
        today_close = current_time.replace(
            hour=self.market_end_hour,
            minute=self.market_end_minute,
            second=0,
            microsecond=0
        )
        
        # If current time is before today's market close, use yesterday's close
        if current_time < today_close:
            # Go back to find the last market day
            last_market_day = current_time - timedelta(days=1)
            while last_market_day.weekday() not in self.market_days:
                last_market_day -= timedelta(days=1)
            
            return last_market_day.replace(
                hour=self.market_end_hour,
                minute=self.market_end_minute,
                second=0,
                microsecond=0
            )
        else:
            # Current time is after today's market close, use today's close
            return today_close
    
    def get_target_time_for_analysis(self, current_time: datetime = None, time_interval_minutes: int = 30) -> datetime:
        """
        Get the target time for swing analysis (effective time - interval)
        
        Args:
            current_time: Current time (defaults to current time)
            time_interval_minutes: Time interval in minutes to look back
            
        Returns:
            Target time for swing analysis
        """
        effective_time = self.get_effective_time_for_analysis(current_time, time_interval_minutes)
        target_time = effective_time - timedelta(minutes=time_interval_minutes)
        
        logger.info(f"Target time for analysis: {target_time} (effective: {effective_time}, interval: {time_interval_minutes}min)")
        return target_time
    
    def get_market_hours_info(self, current_time: datetime = None) -> dict:
        """
        Get market hours information
        
        Args:
            current_time: Current time (defaults to current time)
            
        Returns:
            Dictionary with market hours information
        """
        if current_time is None:
            current_time = datetime.now()
        
        is_open = self.is_market_open(current_time)
        last_close = self.get_last_market_close(current_time)
        
        return {
            'is_market_open': is_open,
            'current_time': current_time,
            'last_market_close': last_close,
            'market_start': f"{self.market_start_hour:02d}:{self.market_start_minute:02d}",
            'market_end': f"{self.market_end_hour:02d}:{self.market_end_minute:02d}",
            'market_days': [['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][i] for i in self.market_days]
        }

def create_market_hours_manager(config) -> MarketHoursManager:
    """Create a market hours manager instance"""
    return MarketHoursManager(config.market)
