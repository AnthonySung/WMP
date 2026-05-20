# Copyright (c) Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: BSD-3-Clause

"""
DreamerReplay: Chunk-step replay buffer for Dreamer Branch training.

Wraps the existing wm_dataset (populated by WMPRunner's data collection) and
provides Dreamer-style sequence sampling with is_first/is_terminal markers.
"""

import numpy as np
import torch


class DreamerReplay:
    """Replay buffer for Dreamer Branch.

    Reuses the wm_dataset tensors populated during real rollout. Each entry
    corresponds to one chunk-step (wm_update_interval env-steps aggregated).

    Sequence format: (prop, action, reward, image?) at each chunk-step.
    is_first marks episode start; is_terminal marks episode end.
    """

    def __init__(self, wm_dataset, wm_dataset_size, wm_config, depth_predictor=None,
                 depth_index=None, depth_index_inverse=None, env=None):
        """
        Args:
            wm_dataset: dict of tensors from WMPRunner.init_wm_dataset().
            wm_dataset_size: np.array of valid lengths per env.
            wm_config: WM config namespace (has batch_size, batch_length, device).
            depth_predictor: Optional DepthPredictor for non-camera envs.
            depth_index: Indices of camera envs in the dataset.
            depth_index_inverse: Mapping from env index to camera buffer index.
            env: Optional env reference for camera config.
        """
        self._dataset = wm_dataset
        self._dataset_size = wm_dataset_size
        self._config = wm_config
        self._depth_predictor = depth_predictor
        self._depth_index = depth_index
        self._depth_index_inverse = depth_index_inverse
        self._env = env

    @property
    def num_envs(self):
        return len(self._dataset_size)

    @property
    def total_steps(self):
        return int(np.sum(self._dataset_size))

    def ready(self):
        """Check if enough data has been collected."""
        return self.total_steps > self._config.train_start_steps

    def sample_batch(self, batch_size=None, batch_length=None):
        """Sample a batch of sequences from the replay buffer.

        Args:
            batch_size: Number of sequences. Defaults to config.batch_size.
            batch_length: Sequence length in chunk-steps. Defaults to config.batch_length.

        Returns:
            batch_data: dict with keys:
                - prop: (batch_size, batch_length, prop_dim)
                - action: (batch_size, batch_length, action_dim)
                - reward: (batch_size, batch_length)
                - image: (batch_size, batch_length, H, W, 1) [if use_camera]
                - is_first: (batch_size, batch_length)
                - is_terminal: (batch_size, batch_length)
        """
        batch_size = batch_size or self._config.batch_size
        batch_length = batch_length or self._config.batch_length

        # Weighted sampling by dataset size
        p = self._dataset_size / np.sum(self._dataset_size)
        batch_idx = np.random.choice(
            range(self.num_envs), batch_size, replace=True, p=p
        )

        # Ensure we have enough data
        min_len = int(self._dataset_size[batch_idx].min())
        batch_length = min(batch_length, min_len)
        if batch_length <= 1:
            return None

        # Sample end indices
        batch_end_idx = [
            np.random.randint(batch_length, self._dataset_size[idx] + 1)
            for idx in batch_idx
        ]

        batch_data = {}
        for k, v in self._dataset.items():
            if k == "forward_height_map":
                continue
            value = []
            for idx, end_idx in zip(batch_idx, batch_end_idx):
                if k == "image":
                    idx_in_buffer = np.where(self._depth_index == idx)[0]
                    if len(idx_in_buffer) == 0:
                        # Non-camera env: use predicted depth
                        if self._depth_predictor is not None:
                            tmp_fhm = self._dataset["forward_height_map"][
                                idx, end_idx - batch_length : end_idx
                            ]
                            tmp_prop = self._dataset["prop"][
                                idx, end_idx - batch_length : end_idx
                            ]
                            pred_depth = self._depth_predictor(tmp_fhm, tmp_prop)
                            value.append(pred_depth)
                        else:
                            # Fallback: zeros
                            shape = (batch_length,) + v.shape[2:]
                            value.append(torch.zeros(shape, device=v.device))
                    else:
                        value.append(v[idx_in_buffer[0], end_idx - batch_length : end_idx])
                else:
                    value.append(v[idx, end_idx - batch_length : end_idx])
            value = torch.stack(value)
            batch_data[k] = value

        # Determine device from the first dataset tensor
        first_key = next(iter(self._dataset))
        device = self._dataset[first_key].device

        # is_first: first step of each sequence is always the start
        is_first = torch.zeros((batch_size, batch_length), device=device)
        is_first[:, 0] = 1
        batch_data["is_first"] = is_first

        # is_terminal: derived from is_first (terminal[t] = is_first[t+1])
        is_terminal = torch.zeros((batch_size, batch_length), device=device)
        is_terminal[:, :-1] = is_first[:, 1:]
        batch_data["is_terminal"] = is_terminal

        return batch_data

    def get_latent_rollout_init(self, batch_data):
        """Extract initial latent state for imagined rollout.

        Runs the encoder and RSSM observe on the first few steps of a batch
        to get a posterior state that can seed imagination.

        Args:
            batch_data: Output of sample_batch().

        Returns:
            post_state: Initial posterior state dict for imagination.
        """
        # This is called from DreamerBehavior; actual implementation
        # requires WorldModel reference, so it's wired in DreamerRunner.
        raise NotImplementedError("Latent rollout init is handled by DreamerRunner")
