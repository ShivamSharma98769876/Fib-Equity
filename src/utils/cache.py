"""
Data caching module for storing and retrieving stock data
"""

import time
import pickle
import hashlib
from typing import Any, Optional, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataCache:
    """Simple in-memory cache with optional persistence"""
    
    def __init__(self, cache_duration: int = 300, persistent: bool = False, cache_dir: str = "cache"):
        """
        Initialize cache
        
        Args:
            cache_duration: Cache duration in seconds
            persistent: Whether to persist cache to disk
            cache_dir: Directory for persistent cache files
        """
        self.cache_duration = cache_duration
        self.persistent = persistent
        self.cache_dir = Path(cache_dir)
        self.memory_cache = {}
        
        if persistent:
            self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, key: str) -> str:
        """Generate cache key"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry is expired"""
        return time.time() - timestamp > self.cache_duration
    
    def _load_from_disk(self, key: str) -> Optional[Any]:
        """Load data from disk cache"""
        if not self.persistent:
            return None
        
        cache_file = self.cache_dir / f"{key}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            # Check if expired
            if self._is_expired(data['timestamp']):
                cache_file.unlink()  # Remove expired file
                return None
            
            return data['value']
        except Exception as e:
            logger.error(f"Error loading cache from disk: {e}")
            return None
    
    def _save_to_disk(self, key: str, value: Any):
        """Save data to disk cache"""
        if not self.persistent:
            return
        
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            data = {
                'value': value,
                'timestamp': time.time()
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving cache to disk: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        cache_key = self._get_cache_key(key)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            data = self.memory_cache[cache_key]
            if not self._is_expired(data['timestamp']):
                return data['value']
            else:
                # Remove expired entry
                del self.memory_cache[cache_key]
        
        # Check disk cache
        if self.persistent:
            value = self._load_from_disk(cache_key)
            if value is not None:
                # Store in memory cache
                self.memory_cache[cache_key] = {
                    'value': value,
                    'timestamp': time.time()
                }
                return value
        
        return None
    
    def set(self, key: str, value: Any):
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
        """
        cache_key = self._get_cache_key(key)
        timestamp = time.time()
        
        # Store in memory cache
        self.memory_cache[cache_key] = {
            'value': value,
            'timestamp': timestamp
        }
        
        # Store in disk cache if persistent
        if self.persistent:
            self._save_to_disk(cache_key, value)
    
    def delete(self, key: str):
        """
        Delete value from cache
        
        Args:
            key: Cache key
        """
        cache_key = self._get_cache_key(key)
        
        # Remove from memory cache
        if cache_key in self.memory_cache:
            del self.memory_cache[cache_key]
        
        # Remove from disk cache
        if self.persistent:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
    
    def clear(self):
        """Clear all cache entries"""
        self.memory_cache.clear()
        
        if self.persistent:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_entries = len(self.memory_cache)
        expired_entries = 0
        current_time = time.time()
        
        for data in self.memory_cache.values():
            if self._is_expired(data['timestamp']):
                expired_entries += 1
        
        stats = {
            'total_entries': total_entries,
            'expired_entries': expired_entries,
            'active_entries': total_entries - expired_entries,
            'cache_duration': self.cache_duration,
            'persistent': self.persistent
        }
        
        if self.persistent:
            disk_files = len(list(self.cache_dir.glob("*.pkl")))
            stats['disk_files'] = disk_files
        
        return stats
    
    def cleanup_expired(self):
        """Remove expired entries from cache"""
        expired_keys = []
        current_time = time.time()
        
        for key, data in self.memory_cache.items():
            if self._is_expired(data['timestamp']):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
