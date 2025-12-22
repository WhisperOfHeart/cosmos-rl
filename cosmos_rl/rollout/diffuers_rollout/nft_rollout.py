# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List

from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker
from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.rollout.schema import RolloutResult

from cosmos_rl.policy.config import Config as CosmosConfig
import torch
from cosmos_rl.rollout.rollout_base import RolloutBase, RolloutRegistry
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.logging import logger
from torch.distributed.fsdp import (
    register_fsdp_forward_method,
    FSDPModule,
)
from cosmos_rl.dispatcher.data.data_fetcher import DataFetcherBase


@RolloutRegistry.register("diffusion_nft_rollout")
class DiffusionNFTRollout(RolloutBase):
    def __init__(
        self,
        config: CosmosConfig,
        parallel_dims: ParallelDims,
        device: torch.device,
        **kwargs,
    ):
        """
        Initialize the RolloutBase class.
        """
        super().__init__(config, parallel_dims, device)

    def post_init_hook(self, **kwargs):
        self.rollout_config = self.config.rollout
        self.validation_config = self.config.validation
        self._model_param_map = None  # key: compatible name, value: param

    def rollout_generation(
        self,
        payloads: List[RLPayload],
        stream: torch.cuda.Stream,
        data_packer: BaseDataPacker,
        data_fetcher: DataFetcherBase,
        is_validation: bool = False,
        *args,
        **kwargs,
    ) -> List[RolloutResult]:
        """Generate sequences"""
        assert (
            self.parallel_dims.world_size == self.parallel_dims.dp_shard
        ), "HF Rollout only supports world size equal to dp_shard"
        response = []
        if isinstance(self.model, FSDPModule):
            register_fsdp_forward_method(self.model, "generate")
        self.model.eval()
        for pl in payloads:
            prompt = data_packer.rollout_collate_fn(
                [data_packer.get_rollout_input(pl.prompt)]
            )[0]
            model_inputs = self.tokenizer(prompt, return_tensors="pt").to(
                self.model.device
            )
            generated_ids = self.model.generate(
                **model_inputs,
                **(
                    self.hf_generate_kwargs
                    if not is_validation
                    else self.hf_val_generate_kwargs
                ),
            ).to(self.model.device)
            generated_ids = [
                output_ids[len(model_inputs.input_ids) :]
                for output_ids in generated_ids
            ]
            texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for text in texts:
                logger.debug(f"[ExampleHFRollout] Generated: {text}")
            response.append(
                RolloutResult(
                    prompt=pl.prompt,
                    completions=texts,
                    completion_logprobs=None,
                    completion_token_ids=None,
                )
            )
        return response

    def init_engine(self, quantization: str, seed: int, load_format: str, **kwargs):
        pass

    def get_underlying_model(self):
        """Get the underlying model"""
        return self.model

    def set_underlying_model(self, model: torch.nn.Module):
        """Set the underlying model"""
        self.model = model
