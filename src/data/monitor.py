"""
Continuous monitoring module for real-time stock analysis
"""

import time
import threading
import logging
from typing import Dict, List, Callable, Optional
from datetime import datetime, timedelta
import pandas as pd

from .api_client import KiteConnectClient
from .data_fetcher import DataFetcher
from ..analysis.eligibility_scanner import EligibilityScanner

logger = logging.getLogger(__name__)

class ContinuousMonitor:
    """Continuous monitoring for real-time swing trade opportunities"""
    
    def __init__(self, config, symbols: List[str], update_interval: int = 30):
        """
        Initialize continuous monitor
        
        Args:
            config: Configuration object
            symbols: List of stock symbols to monitor
            update_interval: Update interval in seconds
        """
        self.config = config
        self.symbols = symbols
        self.update_interval = update_interval
        self.is_running = False
        self.monitor_thread = None
        self.callbacks = []
        
        # Initialize components
        self.data_fetcher = DataFetcher(config)
        self.scanner = EligibilityScanner(config)
        
        # Results storage
        self.latest_results = {}
        self.alert_history = []
        
        # Performance tracking
        self.scan_count = 0
        self.last_scan_time = None
        self.error_count = 0
    
    def add_callback(self, callback: Callable):
        """
        Add callback function to be called on updates
        
        Args:
            callback: Function to call with (results, timestamp) parameters
        """
        self.callbacks.append(callback)
        logger.info(f"Added callback: {callback.__name__}")
    
    def remove_callback(self, callback: Callable):
        """Remove callback function"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            logger.info(f"Removed callback: {callback.__name__}")
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        if self.is_running:
            logger.warning("Monitoring already running")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Started monitoring {len(self.symbols)} symbols")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Perform scan
                results = self._perform_scan()
                
                # Update results
                self.latest_results = results
                self.scan_count += 1
                self.last_scan_time = datetime.now()
                
                # Check for new alerts
                new_alerts = self._check_for_alerts(results)
                if new_alerts:
                    self.alert_history.extend(new_alerts)
                    logger.info(f"Found {len(new_alerts)} new alerts")
                
                # Notify callbacks
                self._notify_callbacks(results)
                
                # Calculate sleep time
                elapsed = time.time() - start_time
                sleep_time = max(0, self.update_interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                self.error_count += 1
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _perform_scan(self) -> List[Dict]:
        """Perform a single scan of all symbols"""
        try:
            # Get current prices for all symbols
            current_prices = self.data_fetcher.get_current_prices(self.symbols)
            
            # Scan all symbols
            results = self.scanner.scan_multiple_stocks(self.symbols, current_prices)
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing scan: {e}")
            return []
    
    def _check_for_alerts(self, results: List[Dict]) -> List[Dict]:
        """Check for new alerts in scan results"""
        new_alerts = []
        
        for result in results:
            if result.get('eligible', False):
                # Check if this is a new alert
                symbol = result['symbol']
                if self._is_new_alert(symbol, result):
                    alert = {
                        'symbol': symbol,
                        'timestamp': datetime.now(),
                        'current_price': result.get('current_price', 0),
                        'swing_high': result.get('swing_high', 0),
                        'swing_low': result.get('swing_low', 0),
                        'trend': result.get('trend', 'unknown'),
                        'fibonacci_levels': result.get('fibonacci_levels', {}),
                        'alert_type': 'swing_trade_opportunity'
                    }
                    new_alerts.append(alert)
        
        return new_alerts
    
    def _is_new_alert(self, symbol: str, result: Dict) -> bool:
        """Check if this is a new alert for the symbol"""
        # Simple check: if symbol was not eligible in previous scan
        if symbol in self.latest_results:
            previous_result = self.latest_results[symbol]
            return not previous_result.get('eligible', False)
        
        return True  # First time seeing this symbol
    
    def _notify_callbacks(self, results: Dict):
        """Notify all registered callbacks"""
        for callback in self.callbacks:
            try:
                callback(results, self.last_scan_time)
            except Exception as e:
                logger.error(f"Error in callback {callback.__name__}: {e}")
    
    def get_latest_results(self) -> Dict:
        """Get latest scan results"""
        return self.latest_results
    
    def get_eligible_stocks(self) -> List[Dict]:
        """Get currently eligible stocks"""
        if not self.latest_results:
            return []
        
        return [result for result in self.latest_results.values() 
                if result.get('eligible', False)]
    
    def get_alert_history(self, limit: int = 100) -> List[Dict]:
        """Get recent alert history"""
        return self.alert_history[-limit:]
    
    def get_monitoring_stats(self) -> Dict:
        """Get monitoring statistics"""
        return {
            'is_running': self.is_running,
            'scan_count': self.scan_count,
            'last_scan_time': self.last_scan_time,
            'error_count': self.error_count,
            'symbols_monitored': len(self.symbols),
            'update_interval': self.update_interval,
            'total_alerts': len(self.alert_history),
            'eligible_stocks': len(self.get_eligible_stocks())
        }
    
    def update_symbols(self, new_symbols: List[str]):
        """Update the list of symbols to monitor"""
        self.symbols = new_symbols
        logger.info(f"Updated symbols list: {len(new_symbols)} symbols")
    
    def set_update_interval(self, interval: int):
        """Set the update interval in seconds"""
        self.update_interval = interval
        logger.info(f"Updated interval: {interval} seconds")

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self):
        self.alerts = []
        self.alert_filters = {}
    
    def add_alert(self, alert: Dict):
        """Add a new alert"""
        self.alerts.append(alert)
        logger.info(f"Added alert for {alert.get('symbol', 'unknown')}")
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get alerts from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts 
                if alert.get('timestamp', datetime.min) > cutoff_time]
    
    def clear_old_alerts(self, days: int = 7):
        """Clear alerts older than N days"""
        cutoff_time = datetime.now() - timedelta(days=days)
        self.alerts = [alert for alert in self.alerts 
                      if alert.get('timestamp', datetime.min) > cutoff_time]
        logger.info(f"Cleared alerts older than {days} days")

# Convenience functions
def start_monitoring(symbols: List[str], config, update_interval: int = 30) -> ContinuousMonitor:
    """Start monitoring symbols"""
    monitor = ContinuousMonitor(config, symbols, update_interval)
    monitor.start_monitoring()
    return monitor

def create_alert_callback(alert_manager: AlertManager):
    """Create a callback function for alerts"""
    def alert_callback(results, timestamp):
        eligible_stocks = [r for r in results if r.get('eligible', False)]
        for stock in eligible_stocks:
            alert = {
                'symbol': stock['symbol'],
                'timestamp': timestamp,
                'current_price': stock.get('current_price', 0),
                'trend': stock.get('trend', 'unknown'),
                'alert_type': 'swing_trade_opportunity'
            }
            alert_manager.add_alert(alert)
    
    return alert_callback
