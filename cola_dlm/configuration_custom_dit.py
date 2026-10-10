# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
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

from transformers import PretrainedConfig


class CustomDiTConfig(PretrainedConfig):
    model_type = "custom_dit"

    def __init__(
        self,
        hidden_dim=1024,
        num_heads=16,
        num_layers=24,
        txt_dim=1024,
        rope_theta=500000.0,
        layer_norm_eps=1e-6,
        attn_dropout=0.05,
        time_emb_num_freqs=32,
        time_emb_f_max=100,
        act='silu',
        adaLN_act='silu',
        **kwargs,
    ):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.txt_dim = txt_dim
        self.rope_theta = rope_theta
        self.layer_norm_eps = layer_norm_eps
        self.attn_dropout = attn_dropout
        self.time_emb_num_freqs = time_emb_num_freqs
        self.time_emb_f_max = time_emb_f_max
        self.act = act
        self.adaLN_act = act
        super().__init__(**kwargs)
