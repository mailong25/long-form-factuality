# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Class for querying the Google Serper API with thread-safe JSON file caching."""

import json
import os
import random
import time
import hashlib
import threading
from typing import Any, Optional, Literal
from datetime import datetime, timedelta

import requests

_SERPER_URL = 'https://google.serper.dev'
NO_RESULT_MSG = 'No good Google Search result was found'


class SerperAPI:
  """Class for querying the Google Serper API with thread-safe JSON file caching."""

  # Class-level lock shared across all instances to ensure thread safety
  _cache_lock = threading.Lock()

  def __init__(
      self,
      serper_api_key: str,
      gl: str = 'us',
      hl: str = 'en',
      k: int = 1,
      tbs: Optional[str] = None,
      search_type: Literal['news', 'search', 'places', 'images'] = 'search',
      cache_file: str = 'serper_cache.json',
      cache_expiry_hours: int = 240,
  ):
    self.serper_api_key = serper_api_key
    self.gl = gl
    self.hl = hl
    self.k = k
    self.tbs = tbs
    self.search_type = search_type
    self.cache_file = cache_file
    self.cache_expiry_hours = cache_expiry_hours
    self.result_key_for_type = {
        'news': 'news',
        'places': 'places',
        'images': 'images',
        'search': 'organic',
    }

  def _generate_cache_key(self, query: str, **kwargs: Any) -> str:
    """Generate a unique cache key for the query and parameters."""
    cache_params = {
        'query': query,
        'gl': self.gl,
        'hl': self.hl,
        'k': self.k,
        'tbs': self.tbs,
        'search_type': self.search_type,
        **kwargs
    }
    # Create a hash of the parameters for a unique key
    cache_string = json.dumps(cache_params, sort_keys=True)
    return hashlib.md5(cache_string.encode()).hexdigest()

  def _load_cache(self) -> dict:
    """Load cache from JSON file. Must be called within lock."""
    if not os.path.exists(self.cache_file):
      return {}
    
    try:
      with open(self.cache_file, 'r', encoding='utf-8') as f:
        return json.load(f)
    except (json.JSONDecodeError, IOError):
      # If file is corrupted or unreadable, start with empty cache
      return {}

  def _save_cache(self, cache: dict) -> None:
    """Save cache to JSON file. Must be called within lock."""
    try:
      # Create directory if it doesn't exist
      os.makedirs(os.path.dirname(self.cache_file) if os.path.dirname(self.cache_file) else '.', exist_ok=True)
      
      # Write to temporary file first, then rename for atomic operation
      temp_file = f"{self.cache_file}.tmp"
      with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
      
      # Atomic rename (on most systems)
      os.replace(temp_file, self.cache_file)
    except IOError as e:
      # Clean up temp file if it exists
      if os.path.exists(temp_file):
        try:
          os.remove(temp_file)
        except OSError:
          pass
      # If we can't write to file, continue without caching
      print(f"Warning: Could not save cache to {self.cache_file}: {e}")

  def _is_cache_expired(self, timestamp: str) -> bool:
    """Check if cache entry is expired."""
    try:
      cached_time = datetime.fromisoformat(timestamp)
      expiry_time = cached_time + timedelta(hours=self.cache_expiry_hours)
      return datetime.now() > expiry_time
    except ValueError:
      # If timestamp is invalid, consider it expired
      return True

  def _clean_expired_cache(self, cache: dict) -> dict:
    """Remove expired entries from cache."""
    cleaned_cache = {}
    
    for key, value in cache.items():
      if isinstance(value, dict) and 'timestamp' in value:
        if not self._is_cache_expired(value['timestamp']):
          cleaned_cache[key] = value
    
    return cleaned_cache

  def run(self, query: str, **kwargs: Any) -> str:
    """Run query through GoogleSearch with thread-safe caching support."""
    assert self.serper_api_key, 'Missing serper_api_key.'
    
    # Generate cache key
    cache_key = self._generate_cache_key(query, **kwargs)
    
    # Check cache first (thread-safe)
    with self._cache_lock:
      cache = self._load_cache()
      
      # Check if we have a valid cached result
      if cache_key in cache and not self._is_cache_expired(cache[cache_key]['timestamp']):
        return cache[cache_key]['result']
    
    # Make API call outside the lock to allow other threads to access cache
    # while this thread is waiting for API response
    results = self._google_serper_api_results(
        query,
        gl=self.gl,
        hl=self.hl,
        num=self.k,
        tbs=self.tbs,
        search_type=self.search_type,
        **kwargs,
    )

    parsed_result = self._parse_results(results)
    
    # Update cache (thread-safe)
    with self._cache_lock:
      # Reload cache in case it was modified by another thread
      cache = self._load_cache()
      
      # Check again if another thread already cached this result
      if cache_key in cache and not self._is_cache_expired(cache[cache_key]['timestamp']):
        # Another thread already cached this, return their result
        return cache[cache_key]['result']
      
      # Cache our result
      cache[cache_key] = {
        'result': parsed_result,
        'timestamp': datetime.now().isoformat(),
        'query': query,  # Store for debugging/reference
        'search_type': self.search_type
      }
      
      # Clean expired entries periodically (every 10th cache operation)
      if len(cache) % 10 == 0:
        cache = self._clean_expired_cache(cache)
      
      self._save_cache(cache)
    
    return parsed_result

  def clear_cache(self) -> None:
    """Clear all cached results. Thread-safe."""
    with self._cache_lock:
      if os.path.exists(self.cache_file):
        try:
          os.remove(self.cache_file)
        except OSError:
          pass

  def get_cache_stats(self) -> dict:
    """Get statistics about the cache. Thread-safe."""
    with self._cache_lock:
      cache = self._load_cache()
      total_entries = len(cache)
      expired_entries = sum(1 for entry in cache.values() 
                           if isinstance(entry, dict) and 'timestamp' in entry 
                           and self._is_cache_expired(entry['timestamp']))
      
      return {
        'total_entries': total_entries,
        'valid_entries': total_entries - expired_entries,
        'expired_entries': expired_entries,
        'cache_file': self.cache_file,
        'cache_expiry_hours': self.cache_expiry_hours
      }

  def cleanup_expired_cache(self) -> int:
    """Manually clean up expired cache entries. Returns number of entries removed."""
    with self._cache_lock:
      cache = self._load_cache()
      original_size = len(cache)
      cleaned_cache = self._clean_expired_cache(cache)
      
      if len(cleaned_cache) < original_size:
        self._save_cache(cleaned_cache)
      
      return original_size - len(cleaned_cache)

  def _google_serper_api_results(
      self,
      search_term: str,
      search_type: str = 'search',
      max_retries: int = 20,
      **kwargs: Any,
  ) -> dict[Any, Any]:
    """Run query through Google Serper."""
    headers = {
        'X-API-KEY': self.serper_api_key or '',
        'Content-Type': 'application/json',
    }
    params = {
        'q': search_term,
        **{key: value for key, value in kwargs.items() if value is not None},
    }
    response, num_fails, sleep_time = None, 0, 0

    while not response and num_fails < max_retries:
      try:
        response = requests.post(
            f'{_SERPER_URL}/{search_type}', headers=headers, params=params
        )
      except AssertionError as e:
        raise e
      except Exception:  # pylint: disable=broad-exception-caught
        response = None
        num_fails += 1
        sleep_time = min(sleep_time * 2, 600)
        sleep_time = random.uniform(1, 10) if not sleep_time else sleep_time
        time.sleep(sleep_time)

    if not response:
      raise ValueError('Failed to get result from Google Serper API')

    response.raise_for_status()
    search_results = response.json()
    return search_results

  def _parse_snippets(self, results: dict[Any, Any]) -> list[str]:
    """Parse results with robust normalization (flatten lists, ensure strings)."""

    def add(value):
      """Normalize and append string or list-of-strings to snippets."""
      if not value:
        return
      if isinstance(value, str):
        snippets.append(value.replace('\n', ' '))
      elif isinstance(value, list):
        for v in value:
          if isinstance(v, str):
            snippets.append(v.replace('\n', ' '))
          else:
            snippets.append(str(v))
      else:
        snippets.append(str(value))

    snippets = []

    # --- ANSWER BOX ---
    if results.get('answerBox'):
      answer_box = results['answerBox']
      add(answer_box.get('answer'))
      add(answer_box.get('snippet'))
      add(answer_box.get('snippetHighlighted'))

    # --- KNOWLEDGE GRAPH ---
    if results.get('knowledgeGraph'):
      kg = results['knowledgeGraph']
      title = kg.get('title')

      entity_type = kg.get('type')
      if entity_type:
        add(f'{title}: {entity_type}.')

      description = kg.get('description')
      add(description)

      for attribute, value in kg.get('attributes', {}).items():
        add(f'{title} {attribute}: {value}.')

    # --- ORGANIC RESULTS / MAIN LIST ---
    result_key = self.result_key_for_type[self.search_type]
    if result_key in results:
      for result in results[result_key][:self.k]:
        add(result.get('snippet'))

        for attribute, value in result.get('attributes', {}).items():
          add(f'{attribute}: {value}.')

    if not snippets:
      return [NO_RESULT_MSG]
    
    return snippets

  def _parse_results(self, results: dict[Any, Any]) -> str:
    return ' '.join(self._parse_snippets(results))