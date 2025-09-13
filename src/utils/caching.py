"""
Caching utilities for function result persistence.

This module provides decorators for caching function results to disk
to avoid recomputation across program runs.
"""

import functools
import hashlib
import json
import pickle
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _generate_cache_key(func: Callable, args: tuple, kwargs: dict) -> str:
    """
    Generate a unique cache key for function arguments.

    Args:
        func: The function being cached
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        A unique string key for the function call
    """
    # Create a string representation of the function call
    func_name = f"{func.__module__}.{func.__name__}"

    # For class methods, exclude 'self' from args to ensure consistent caching
    # across different instances of the same class
    cache_args = args
    if (
        args
        and hasattr(args[0], "__class__")
        and hasattr(args[0].__class__, func.__name__)
    ):
        # This is likely a class method, exclude self (first argument)
        cache_args = args[1:]

    # Convert arguments to a hashable representation
    # Use JSON serialization with string fallback for complex types
    args_str = json.dumps(cache_args, sort_keys=True, default=str)
    kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)

    # Create hash of the combined representation
    key_data = f"{func_name}:{args_str}:{kwargs_str}"
    return hashlib.md5(key_data.encode()).hexdigest()


def _save_to_disk(cache_dir: Path, key: str, value: Any) -> bool:
    """Save cached value to disk using pickle.

    Returns:
        True if save was successful, False otherwise
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{key}.pkl"

    try:
        with open(cache_file, "wb") as f:
            pickle.dump(value, f)
        return True
    except (OSError, pickle.PickleError) as e:
        print(f"Warning: Failed to save cache to disk: {e}")
        return False


def _load_from_disk(cache_dir: Path, key: str) -> Any:
    """Load cached value from disk using pickle."""
    cache_file = cache_dir / f"{key}.pkl"

    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    except (OSError, pickle.PickleError, EOFError) as e:
        print(f"Warning: Failed to load cache from disk: {e}")
        # Remove corrupted file
        cache_file.unlink(missing_ok=True)
        return None


def _is_cache_expired(cache_file: Path, ttl_seconds: float | None) -> bool:
    """Check if a cache file has expired based on TTL."""
    if ttl_seconds is None:
        return False

    file_age = time.time() - cache_file.stat().st_mtime
    return file_age > ttl_seconds


def cache(
    cache_dir: str | Path | None = None, ttl_seconds: float | None = None
) -> Callable:
    """
    Decorator for caching function results to disk.

    Args:
        cache_dir: Directory for disk cache (defaults to ./cache)
        ttl_seconds: Time-to-live for cached items in seconds (None = no expiration)

    Returns:
        Decorated function with disk caching capability

    Example:
        @cache(cache_dir="./my_cache", ttl_seconds=3600)
        def expensive_function(x, y):
            time.sleep(2)  # Simulate expensive computation
            return x + y

        # First call - computed and cached to disk
        result1 = expensive_function(1, 2)  # Takes 2 seconds

        # Second call - retrieved from disk cache
        result2 = expensive_function(1, 2)  # Fast

        # After restart - still cached
        result3 = expensive_function(1, 2)  # Fast
    """

    def decorator(func: Callable) -> Callable:
        # Setup disk cache directory
        if cache_dir is None:
            disk_cache_dir = Path("cache") / func.__name__
        else:
            disk_cache_dir = Path(cache_dir) / func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = _generate_cache_key(func, args, kwargs)
            cache_file = disk_cache_dir / f"{cache_key}.pkl"

            # Check if cached result exists and is not expired
            if cache_file.exists() and not _is_cache_expired(cache_file, ttl_seconds):
                return _load_from_disk(disk_cache_dir, cache_key)

            # Cache miss or expired - compute the result
            result = func(*args, **kwargs)

            # Store result to disk
            _save_to_disk(disk_cache_dir, cache_key, result)

            return result

        # Add cache management methods to the wrapper
        def clear_cache():
            """Clear all cached results."""
            if not disk_cache_dir.exists():
                return

            for cache_file in disk_cache_dir.glob("*.pkl"):
                cache_file.unlink(missing_ok=True)

        def cache_info():
            """Get information about cache usage."""
            info = {
                "disk_cache_dir": str(disk_cache_dir),
                "ttl_seconds": ttl_seconds,
            }

            if disk_cache_dir.exists():
                cache_files = list(disk_cache_dir.glob("*.pkl"))
                info["cache_size"] = len(cache_files)

                if cache_files:
                    # Get oldest and newest cache files
                    file_times = [(f, f.stat().st_mtime) for f in cache_files]
                    file_times.sort(key=lambda x: x[1])

                    info["oldest_cache"] = time.ctime(file_times[0][1])
                    info["newest_cache"] = time.ctime(file_times[-1][1])
            else:
                info["cache_size"] = 0

            return info

        def cleanup_expired():
            """Remove expired cache files."""
            if ttl_seconds is None or not disk_cache_dir.exists():
                return 0

            removed_count = 0
            for cache_file in disk_cache_dir.glob("*.pkl"):
                if _is_cache_expired(cache_file, ttl_seconds):
                    cache_file.unlink(missing_ok=True)
                    removed_count += 1

            return removed_count

        wrapper.clear_cache = clear_cache
        wrapper.cache_info = cache_info
        wrapper.cleanup_expired = cleanup_expired

        return wrapper

    return decorator


# Convenience decorators for common use cases
def persistent_cache(cache_dir: str | Path | None = None):
    """
    Persistent disk cache decorator with no expiration.

    Args:
        cache_dir: Directory for disk cache (defaults to ./cache)

    Returns:
        Decorated function with persistent disk caching

    Example:
        @persistent_cache("./model_cache")
        def load_model(model_name):
            # Expensive model loading
            return load_heavy_model(model_name)
    """
    return cache(cache_dir=cache_dir, ttl_seconds=None)


def timed_cache(
    ttl_seconds: float = 3600, cache_dir: str | Path | None = None
):
    """
    Time-limited disk cache decorator.

    Args:
        ttl_seconds: Time-to-live for cached items in seconds
        cache_dir: Directory for disk cache (defaults to ./cache)

    Returns:
        Decorated function with time-limited disk caching

    Example:
        @timed_cache(ttl_seconds=1800)  # 30 minutes
        def fetch_data(url):
            return requests.get(url).json()
    """
    return cache(cache_dir=cache_dir, ttl_seconds=ttl_seconds)
