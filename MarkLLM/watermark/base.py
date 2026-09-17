# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ===============================================================
# base.py
# Description: This is a generic watermark class that will be 
#              inherited by the watermark classes of the library.
# ===============================================================

from typing import Union
from MarkLLM.utils.transformers_config import TransformersConfig
from MarkLLM.visualize.data_for_visualization import DataForVisualization
from MarkLLM.utils.utils import load_config_file
from MarkLLM.exceptions.exceptions import AlgorithmNameMismatchError


class BaseConfig:
    """Common helper for loading config JSON and binding transformer resources."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig | None, *args, **kwargs) -> None:
        if algorithm_config is None:
            raise ValueError("algorithm_config path must be provided.")
        if transformers_config is None:
            raise ValueError("transformers_config must be provided to initialize watermark config.")

        self.config_path = algorithm_config
        self.config_dict = load_config_file(algorithm_config)
        if self.config_dict is None:
            raise ValueError(f"Failed to load algorithm config from {algorithm_config}")

        expected = self.algorithm_name
        actual = self.config_dict.get("algorithm_name")
        if expected and actual and expected != actual:
            raise AlgorithmNameMismatchError(expected, actual)

        self.transformers_config = transformers_config
        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = dict(transformers_config.gen_kwargs)
        # Ensure default decoding constraints for plain generation.
        self.gen_kwargs.setdefault("no_repeat_ngram_size", 4)
        self._ensure_model_pad_token()

        self.initialize_parameters()

    @property
    def algorithm_name(self) -> str:
        raise NotImplementedError("Subclasses must override algorithm_name property.")

    def initialize_parameters(self) -> None:
        raise NotImplementedError("Subclasses must implement initialize_parameters().")

    def _ensure_model_pad_token(self) -> None:
        pad_id = getattr(self.generation_tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.generation_tokenizer, "eos_token_id", None)
        if pad_id is None and hasattr(self.generation_model, "config"):
            pad_id = getattr(self.generation_model.config, "eos_token_id", None)
        if pad_id is None:
            return
        if hasattr(self.generation_model, "config") and getattr(self.generation_model.config, "pad_token_id", None) is None:
            self.generation_model.config.pad_token_id = pad_id
        gen_cfg = getattr(self.generation_model, "generation_config", None)
        if gen_cfg is not None and getattr(gen_cfg, "pad_token_id", None) is None:
            gen_cfg.pad_token_id = pad_id


class BaseWatermark:
    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        pass

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str: 
        pass

    def generate_unwatermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate unwatermarked text."""
        
        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        # Generate unwatermarked text
        encoded_unwatermarked_text = self.config.generation_model.generate(**encoded_prompt, **self.config.gen_kwargs)
        # Decode
        unwatermarked_text = self.config.generation_tokenizer.batch_decode(encoded_unwatermarked_text, skip_special_tokens=True)[0]
        return unwatermarked_text

    def detect_watermark(self, text:str, return_dict: bool=True, *args, **kwargs) -> Union[tuple, dict]:
        pass

    def get_data_for_visualize(self, text, *args, **kwargs) -> DataForVisualization:
        pass
