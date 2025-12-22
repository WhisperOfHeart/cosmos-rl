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

import argparse
import json
import os
import toml

import torch
from torch.utils.data import Dataset, Sampler

from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.config import DatasetConfig


# https://github.com/NVlabs/DiffusionNFT/blob/24af5554898d85e93efa492f42b5e9cdf4156e9b/scripts/train_nft_sd3.py#L118
class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert (
            self.total_samples % self.k == 0
        ), f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            shuffled_indices = torch.randperm(
                len(repeated_indices), generator=g
            ).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}.txt")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}_metadata.jsonl")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}


def get_dataset(dataset_config: DatasetConfig) -> Dataset:
    assert dataset_config.name in [
        "pickscore",
        "ocr",
        "geneval",
    ], f"Unknown dataset name: {dataset_config.name}"
    prompt_fn = "geneval" if dataset_config.name == "geneval" else "general_ocr"
    if prompt_fn == "general_ocr":
        dataset = TextPromptDataset(dataset_config.name, split=dataset_config.split)
    elif prompt_fn == "geneval":
        dataset = GenevalPromptDataset(dataset_config.name, split=dataset_config.split)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_config.name}")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_known_args()[0]
    with open(args.config, "r") as f:
        config = toml.load(f)
    config = CosmosConfig.from_dict(config)

    train_dataset = get_dataset(config.train.train_policy.dataset)
    val_dataset = get_dataset(config.validation.dataset)
    rollout_batch_size = (
        config.train.train_policy.dataloader_batch_size or config.rollout.batch_size
    )
    batch_sampler = DistributedKRepeatSampler(
        train_dataset,
        batch_size=rollout_batch_size,
        k=config.rollout.n_generation,
    )

    launch_worker(
        dataset=train_dataset,
        val_dataset=val_dataset,
        batch_sampler=batch_sampler,
    )
