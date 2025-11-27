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
"""Rates a single atomic fact for accuracy."""

import dataclasses
import re
from typing import Any

# pylint: disable=g-bad-import-order
from common import modeling
from common import shared_config
from common import utils
from eval.safe import config as safe_config
from eval.safe import query_serper
# pylint: enable=g-bad-import-order

import hashlib
import json
import os
from typing import Dict, Optional, List, Tuple
import threading
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
import pickle


SUPPORTED_LABEL = 'Supported'
NOT_SUPPORTED_LABEL = 'Not Supported'

_STATEMENT_PLACEHOLDER = '[STATEMENT]'
_KNOWLEDGE_PLACEHOLDER = '[KNOWLEDGE]'
_NEXT_SEARCH_FORMAT = f"""\
Instructions:
1. You have been given a STATEMENT and some KNOWLEDGE points.
2. Your goal is to try to find evidence that either supports or does not \
support the factual accuracy of the given STATEMENT.
3. To do this, you are allowed to issue ONE Google Search query that you think \
will allow you to find additional useful evidence.
4. Your query should aim to obtain new information that does not appear in the \
KNOWLEDGE. This new information should be useful for determining the factual \
accuracy of the given STATEMENT.
5. Format your final query by putting it in a markdown code block.

KNOWLEDGE:
{_KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""
_FINAL_ANSWER_FORMAT = f"""\
Instructions:
1. You have been given a STATEMENT and some KNOWLEDGE points.
2. Determine whether the given STATEMENT is supported by the given KNOWLEDGE. \
The STATEMENT does not need to be explicitly supported by the KNOWLEDGE, but \
should be strongly implied by the KNOWLEDGE.
3. Before showing your answer, think step-by-step and show your specific \
reasoning. As part of your reasoning, summarize the main points of the \
KNOWLEDGE.
4. If the STATEMENT is supported by the KNOWLEDGE, be sure to show the \
supporting evidence.
5. After stating your reasoning, restate the STATEMENT and then determine your \
final answer based on your reasoning and the STATEMENT.
6. Your final answer should be either "{SUPPORTED_LABEL}" or \
"{NOT_SUPPORTED_LABEL}". Wrap your final answer in square brackets.

KNOWLEDGE:
{_KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""


@dataclasses.dataclass()
class GoogleSearchResult:
  query: str
  result: str


@dataclasses.dataclass()
class FinalAnswer:
  response: str
  answer: str


def call_search(
    search_query: str,
    search_type: str = safe_config.search_type,
    num_searches: int = safe_config.num_searches,
    serper_api_key: str = shared_config.serper_api_key,
    search_postamble: str = '',  # ex: 'site:https://en.wikipedia.org'
) -> str:
  """Call Google Search to get the search result."""
  search_query += f' {search_postamble}' if search_postamble else ''

  if search_type == 'serper':
    serper_searcher = query_serper.SerperAPI(serper_api_key, k=num_searches)
    return serper_searcher.run(search_query, k=num_searches)
  else:
    raise ValueError(f'Unsupported search type: {search_type}')

def maybe_get_next_search(
    atomic_fact: str,
    past_searches: list[GoogleSearchResult],
    model: modeling.Model,
    debug: bool = safe_config.debug_safe,
) -> GoogleSearchResult | None:
  """Get the next query from the model."""
  knowledge = '\n'.join([s.result for s in past_searches])
  knowledge = 'N/A' if not knowledge else knowledge
  full_prompt = _NEXT_SEARCH_FORMAT.replace(_STATEMENT_PLACEHOLDER, atomic_fact)
  full_prompt = full_prompt.replace(_KNOWLEDGE_PLACEHOLDER, knowledge)
  full_prompt = utils.strip_string(full_prompt)
  model_response = model.generate(full_prompt, do_debug=debug)
  query = utils.extract_first_code_block(model_response, ignore_language=True)

  if model_response and query:
    return GoogleSearchResult(query=query, result=call_search(query))

  return None


def maybe_get_final_answer(
    atomic_fact: str,
    searches: list[GoogleSearchResult],
    model: modeling.Model,
    debug: bool = safe_config.debug_safe,
) -> FinalAnswer | None:
  """Get the final answer from the model."""
  knowledge = '\n'.join([search.result for search in searches])
  full_prompt = _FINAL_ANSWER_FORMAT.replace(
      _STATEMENT_PLACEHOLDER, atomic_fact
  )
  full_prompt = full_prompt.replace(_KNOWLEDGE_PLACEHOLDER, knowledge)
  full_prompt = utils.strip_string(full_prompt)
  model_response = model.generate(full_prompt, do_debug=debug)
  answer = utils.extract_first_square_brackets(model_response)
  answer = re.sub(r'[^\w\s]', '', answer).strip()

  if model_response and answer in [SUPPORTED_LABEL, NOT_SUPPORTED_LABEL]:
    return FinalAnswer(response=model_response, answer=answer)

  return None

##### ---------------

class SemanticAtomicFactCache:
    """Cache for atomic fact checking results with semantic similarity matching."""
    
    def __init__(self, cache_file: str = "atomic_fact_cache.json", 
                 embeddings_file: str = "fact_embeddings.pkl",
                 openai_api_key: str = None,
                 similarity_threshold: float = 0.8):
        self.cache_file = cache_file
        self.embeddings_file = embeddings_file
        self.cache: Dict[str, Dict] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.similarity_threshold = similarity_threshold
        self._lock = threading.Lock()
        
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        
        self._load_cache()
        self._load_embeddings()
    
    def _generate_cache_key(self, atomic_fact: str) -> str:
        """Generate a unique cache key for the atomic fact."""
        return hashlib.md5(atomic_fact.encode()).hexdigest()
    
    def _load_cache(self):
        """Load cache from file if it exists."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.cache = {}
    
    def _save_cache(self):
        """Save cache to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except IOError:
            pass
    
    def _load_embeddings(self):
        """Load embeddings from file if it exists."""
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
            except (pickle.PickleError, IOError):
                self.embeddings = {}
    
    def _save_embeddings(self):
        """Save embeddings to file."""
        try:
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
        except IOError:
            pass
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text, checking cache first before calling OpenAI API."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized. Please provide API key.")
        
        # Generate cache key for the text
        text_key = self._generate_cache_key(text)
        
        # Check if embedding already exists in cache
        if text_key in self.embeddings:
            return self.embeddings[text_key]
        
        # If not in cache, call OpenAI API
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",  # or "text-embedding-3-large"
                input=text
            )
            embedding = np.array(response.data[0].embedding)
            
            # Cache the embedding for future use
            self.embeddings[text_key] = embedding
            self._save_embeddings()
            
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def _get_top_similar_facts(self, atomic_fact: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Get top k similar facts from cache based on embedding similarity."""
        if not self.embeddings:
            return []
        
        # Get embedding for the input fact
        query_embedding = self._get_embedding(atomic_fact)
        if query_embedding is None:
            return []
        
        similarities = []
        for cache_key, cached_embedding in self.embeddings.items():
            if cache_key in self.cache:
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    cached_embedding.reshape(1, -1)
                )[0][0]
                
                cached_fact = self.cache[cache_key]['atomic_fact']
                similarities.append((cache_key, cached_fact, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:top_k]
    
    def _check_semantic_equivalence(self, fact1: str, fact2: str) -> bool:
        """Use OpenAI API to check if two facts have the same meaning."""
        if not self.openai_client:
            return False
        
        prompt = f"""
Please determine if these two statements have the same meaning:

Statement 1: "{fact1}"
Statement 2: "{fact2}"

Consider them the same if they convey identical factual information, even if worded differently.
Consider them different if they have different subjects, objects, actions, or any other factual details.

Respond with only "YES" if they have the same meaning, or "NO" if they don't.
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0
            )
            
            answer = response.choices[0].message.content.strip().upper()
            return answer == "YES"
        except Exception as e:
            print(f"Error checking semantic equivalence: {e}")
            return False
    
    def get(self, atomic_fact: str) -> Optional[tuple]:
        """Get cached result if semantically equivalent fact exists."""
        with self._lock:
            # First try exact match
            exact_key = self._generate_cache_key(atomic_fact)
            if exact_key in self.cache:
                cached_data = self.cache[exact_key]
                final_answer = None
                if cached_data.get('final_answer'):
                    final_answer = FinalAnswer(
                        response=cached_data['final_answer']['response'],
                        answer=cached_data['final_answer']['answer']
                    )
                return final_answer, cached_data['search_dicts']
            
            # Get top 5 similar facts
            similar_facts = self._get_top_similar_facts(atomic_fact, top_k=5)
            
            # Check semantic equivalence for each similar fact
            for cache_key, cached_fact, similarity in similar_facts:
                if similarity >= self.similarity_threshold:
                    if self._check_semantic_equivalence(atomic_fact, cached_fact):
                        cached_data = self.cache[cache_key]
                        final_answer = None
                        if cached_data.get('final_answer'):
                            final_answer = FinalAnswer(
                                response=cached_data['final_answer']['response'],
                                answer=cached_data['final_answer']['answer']
                            )
                        print("Cached for", atomic_fact, "with", cached_fact)
                        return final_answer, cached_data['search_dicts']
        
        return None
    
    def set(self, atomic_fact: str, final_answer: FinalAnswer | None, search_dicts: dict):
        """Cache the result with embedding."""
        cache_key = self._generate_cache_key(atomic_fact)
        
        # Get embedding for the atomic fact
        embedding = self._get_embedding(atomic_fact)
        if embedding is None:
            print(f"Warning: Could not generate embedding for fact: {atomic_fact[:50]}...")
            return
        
        # Serialize the FinalAnswer object
        final_answer_dict = None
        if final_answer:
            final_answer_dict = {
                'response': final_answer.response,
                'answer': final_answer.answer
            }
        
        cache_data = {
            'final_answer': final_answer_dict,
            'search_dicts': search_dicts,
            'atomic_fact': atomic_fact,
        }
        
        with self._lock:
            self.cache[cache_key] = cache_data
            self.embeddings[cache_key] = embedding
            self._save_cache()
            self._save_embeddings()
    
    def clear(self):
        """Clear the cache and embeddings."""
        with self._lock:
            self.cache.clear()
            self.embeddings.clear()
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
            if os.path.exists(self.embeddings_file):
                os.remove(self.embeddings_file)
    
    def size(self) -> int:
        """Get the number of cached items."""
        return len(self.cache)
    
    def get_similar_facts_debug(self, atomic_fact: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Debug method to see similar facts without semantic checking."""
        similar_facts = self._get_top_similar_facts(atomic_fact, top_k)
        return [(fact, similarity) for _, fact, similarity in similar_facts]

# # Global cache instance
_fact_cache = SemanticAtomicFactCache()

def check_atomic_fact(
    atomic_fact: str,
    rater: modeling.Model,
    max_steps: int = safe_config.max_steps,
    max_retries: int = safe_config.max_retries,
    debug: bool = safe_config.debug_safe,
    use_cache: bool = True,
) -> tuple[FinalAnswer | None, dict[str, Any]]:
    """Check if the given atomic fact is supported."""
    
    if _fact_cache is None:
        raise ValueError("Semantic cache not initialized. Call initialize_semantic_cache() first.")
    
    # Check cache first
    if use_cache:
        cached_result = _fact_cache.get(atomic_fact)
        if cached_result is not None:
            if debug:
                print(f"Semantic cache hit for atomic fact: {atomic_fact[:50]}...")
            return cached_result
    
    # Original implementation
    search_results = []

    for _ in range(max_steps):
        next_search, num_tries = None, 0

        while not next_search and num_tries <= max_retries:
            next_search = maybe_get_next_search(atomic_fact, search_results, rater)
            num_tries += 1

        if next_search is None:
            utils.maybe_print_error('Unsuccessful parsing for `next_search`')
            break
        else:
            search_results.append(next_search)

    search_dicts = {
        'google_searches': [dataclasses.asdict(s) for s in search_results]
    }
    final_answer, num_tries = None, 0

    while not final_answer and num_tries <= max_retries:
        num_tries += 1
        final_answer = maybe_get_final_answer(
            atomic_fact, searches=search_results, model=rater, debug=debug
        )

    if final_answer is None:
        utils.maybe_print_error('Unsuccessful parsing for `final_answer`')

    # Cache the result
    if use_cache:
        _fact_cache.set(atomic_fact, final_answer, search_dicts)
        # if debug:
        #     print(f"Cached result for atomic fact: {atomic_fact[:50]}...")

    return final_answer, search_dicts


# def check_atomic_fact(
#     atomic_fact: str,
#     rater: modeling.Model,
#     max_steps: int = safe_config.max_steps,
#     max_retries: int = safe_config.max_retries,
#     debug: bool = safe_config.debug_safe,
# ) -> tuple[FinalAnswer | None, dict[str, Any]]:
#   """Check if the given atomic fact is supported."""
#   search_results = []

#   for _ in range(max_steps):
#     next_search, num_tries = None, 0

#     while not next_search and num_tries <= max_retries:
#       next_search = maybe_get_next_search(atomic_fact, search_results, rater)
#       num_tries += 1

#     if next_search is None:
#       utils.maybe_print_error('Unsuccessful parsing for `next_search`')
#       break
#     else:
#       search_results.append(next_search)

#   search_dicts = {
#       'google_searches': [dataclasses.asdict(s) for s in search_results]
#   }
#   final_answer, num_tries = None, 0

#   while not final_answer and num_tries <= max_retries:
#     num_tries += 1
#     final_answer = maybe_get_final_answer(
#         atomic_fact, searches=search_results, model=rater, debug=debug
#     )

#   if final_answer is None:
#     utils.maybe_print_error('Unsuccessful parsing for `final_answer`')

#   return final_answer, search_dicts
