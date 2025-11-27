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
import os
import json
import threading
import hashlib
from pathlib import Path
import openai
import logging
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("_client").setLevel(logging.WARNING)
from helm.clients.custom_client import generate_response
from datetime import datetime

##############################################################
###### IMPLEMENT YOUR CUSTORM MODEL HERE ######

MODEL_NAME = os.getenv("GEN_MODEL").split('/')[1]
MODEL_PROVIDER = os.getenv("GEN_MODEL").split('/')[0]

def custom_llm_call(model_name, prompt, gen_temp, gen_max_tokens, timeout):
    raw_request = {"provider": MODEL_PROVIDER, "model": MODEL_NAME,
                   "prompt": prompt, "n" : 1,
                   "temperature": gen_temp, "max_tokens": gen_max_tokens}
    response = generate_response(raw_request)
    return response['completions'][0]
# ----------------------------------

# Constants
CACHE_FILE = "llm_cache.json"
CACHE_LOCK = threading.Lock()

# Ensure cache file exists
Path(CACHE_FILE).touch(exist_ok=True)
if os.stat(CACHE_FILE).st_size == 0:
    with open(CACHE_FILE, "w") as f:
        json.dump({}, f)

def _make_cache_key(prompt, gen_temp, gen_max_tokens, model_name):
    """
    Generate a unique hash key for given input parameters.
    """
    date_now = datetime.now().strftime("%Y-%m-%d")
    key_data = f"{prompt}|{gen_temp}|{gen_max_tokens}|{model_name}|{date_now}"
    return hashlib.sha256(key_data.encode("utf-8")).hexdigest()

def _read_cache():
    with CACHE_LOCK:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)

def _write_cache(cache):
    with CACHE_LOCK:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)

def custom_model_generate(prompt, gen_temp, gen_max_tokens, timeout, model_name=MODEL_NAME, use_cache = True):
    if not use_cache:
        return custom_llm_call(model_name, prompt, gen_temp, gen_max_tokens, timeout)
    else:
        cache_key = _make_cache_key(prompt, gen_temp, gen_max_tokens, model_name)
        cache = _read_cache()
        if cache_key in cache:
            return cache[cache_key]
        response_text = custom_llm_call(model_name, prompt, gen_temp, gen_max_tokens, timeout)
        cache[cache_key] = response_text
        _write_cache(cache)
        return response_text
###############################################################


from concurrent import futures
import functools
import logging
import os
import threading
import time
from typing import Any, Annotated, Optional

import anthropic
import langfun as lf
import pyglove as pg

# pylint: disable=g-bad-import-order
from common import modeling_utils
from common import shared_config
from common import utils
# pylint: enable=g-bad-import-order

_DEBUG_PRINT_LOCK = threading.Lock()
_ANTHROPIC_MODELS = [
    'claude-3-opus-20240229',
    'claude-3-sonnet-20240229',
    'claude-3-haiku-20240307',
    'claude-2.1',
    'claude-2.0',
    'claude-instant-1.2',
]


class Usage(pg.Object):
  """Usage information per completion."""

  prompt_tokens: int
  completion_tokens: int

class Model:
  """Class for storing any single language model."""

  def __init__(
      self,
      model_name: str,
      temperature: float = 0.5,
      max_tokens: int = 2048,
      show_responses: bool = False,
      show_prompts: bool = False,
      use_cache = True,
  ) -> None:
    """Initializes a model."""
    self.model_name = model_name
    self.temperature = temperature
    self.max_tokens = max_tokens
    self.show_responses = show_responses
    self.show_prompts = show_prompts
    self.use_cache = use_cache

  def load(
      self, model_name: str, temperature: float, max_tokens: int
  ) -> lf.LanguageModel:
    """Loads a language model from string representation."""
    sampling = lf.LMSamplingOptions(
        temperature=temperature, max_tokens=max_tokens
    )

    if model_name.lower().startswith('openai:'):
      if not shared_config.openai_api_key:
        utils.maybe_print_error('No OpenAI API Key specified.')
        utils.stop_all_execution(True)

      return lf.llms.OpenAI(
          model=model_name[7:],
          api_key=shared_config.openai_api_key,
          sampling_options=sampling,
      )
    elif model_name.lower().startswith('anthropic:'):
      if not shared_config.anthropic_api_key:
        utils.maybe_print_error('No Anthropic API Key specified.')
        utils.stop_all_execution(True)

      return AnthropicModel(
          model=model_name[10:],
          api_key=shared_config.anthropic_api_key,
          sampling_options=sampling,
      )
    elif 'unittest' == model_name.lower():
      return lf.llms.Echo()
    else:
      raise ValueError(f'ERROR: Unsupported model type: {model_name}.')

  def generate(
        self,
        prompt: str,
        do_debug: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_attempts: int = 5,
        timeout: int = 60,
        retry_interval: int = 5,
    ) -> str:
        """
        """
              
        gen_temp = temperature if temperature is not None else self.temperature
        gen_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        attempt = 0
        response_text = ""
        while attempt < max_attempts and not response_text:
            attempt += 1
            try:
                response_text = custom_model_generate(prompt, gen_temp, gen_max_tokens, timeout, self.use_cache)
            except Exception as e:
                print(f"[Unexpected error attempt {attempt}]: {e}")
                time.sleep(retry_interval)
        
        if do_debug:
            if self.show_prompts:
                print("PROMPT:", prompt)
            if self.show_responses:
                print("RESPONSE:", response_text)
    
        return response_text

  def print_config(self) -> None:
    settings = {
        'model_name': self.model_name,
        'temperature': self.temperature,
        'max_tokens': self.max_tokens,
        'show_responses': self.show_responses,
        'show_prompts': self.show_prompts,
    }
    print(utils.to_readable_json(settings))

class FakeModel(Model):
  """Class for faking responses during unit tests."""

  def __init__(
      self,
      static_response: str = '',
      sequential_responses: Optional[list[str]] = None,
  ) -> None:
    Model.__init__(self, model_name='unittest')
    self.static_response = static_response
    self.sequential_responses = sequential_responses
    self.sequential_response_idx = 0

    if static_response:
      self.model = lf.llms.StaticResponse(static_response)
    elif sequential_responses:
      self.model = lf.llms.StaticSequence(sequential_responses)
    else:
      self.model = lf.llms.Echo()

  def generate(
      self,
      prompt: str,
      do_debug: bool = False,
      temperature: Optional[float] = None,
      max_tokens: Optional[int] = None,
      max_attempts: int = 1000,
      timeout: int = 60,
      retry_interval: int = 10,
  ) -> str:
    if self.static_response:
      return self.static_response
    elif self.sequential_responses:
      response = self.sequential_responses[
          self.sequential_response_idx % len(self.sequential_responses)
      ]
      self.sequential_response_idx += 1
      return response
    else:
      return ''
