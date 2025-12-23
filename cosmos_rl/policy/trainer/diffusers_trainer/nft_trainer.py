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

from typing import Optional

import torch

from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.ema import EMAModuleWrapper
from cosmos_rl.utils.util import copy_weights
from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker
from cosmos_rl.policy.trainer.base import TrainerRegistry
from cosmos_rl.policy.trainer.diffusers_trainer.diffusers_trainer import (
    DiffusersTrainer,
)


@TrainerRegistry.register(trainer_type="diffusion_nft")
class NFTTrainer(DiffusersTrainer):
    def __init__(
        self,
        config: CosmosConfig,
        parallel_dims: ParallelDims,
        train_stream: Optional[torch.cuda.Stream] = None,
        data_packer: BaseDataPacker = None,
        val_data_packer: BaseDataPacker = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            parallel_dims=parallel_dims,
            train_stream=train_stream,
            data_packer=data_packer,
            val_data_packer=val_data_packer,
            **kwargs,
        )

        # Create ref model for RL training
        trainable_params = list(
            filter(
                lambda p: p.requires_grad, self.model.transformer.module.parameters()
            )
        )
        self.model.transformer.set_adapter("ref")
        ref_trainable_params = list(
            filter(
                lambda p: p.requires_grad, self.model.transformer.module.parameters()
            )
        )
        self.model.transformer.set_adapter("default")
        copy_weights(
            src_params=trainable_params,
            tgt_params=ref_trainable_params,
        )

        # Create ema if needed
        if self.config.train.ema_enable:
            self.ema = EMAModuleWrapper(
                parameters=trainable_params,
                decay=self.config.train.ema_decay,
                update_step_interval=self.config.train.ema_update_step_interval,
                device=self.device,
            )

    def step_training(self):
        pass
