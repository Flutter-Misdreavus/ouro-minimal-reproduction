"""Optional upstream compatibility checks after downloading its source files."""
import os
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not Path('runtime/model/configuration_ouro.py').exists(), reason='Download upstream first')
def test_upstream_causality_and_fixed_depth():
    os.environ['HF_HOME'] = str(Path('runtime/hf').resolve())
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained('runtime/model', trust_remote_code=True, local_files_only=True)
    cfg.hidden_size, cfg.intermediate_size = 32, 64
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 4, 8
    cfg.num_hidden_layers, cfg.layer_types = 2, ['full_attention'] * 2
    cfg.vocab_size, cfg.total_ut_steps, cfg.use_cache = 64, 4, False
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, attn_implementation='sdpa').eval()
    model.early_exit_step = 3
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        outputs, states, _ = model.model(tokens, use_cache=False)
        full = model(tokens, use_cache=False).logits
        torch.testing.assert_close(full, model.lm_head(states[-1]))
        for depth in (1, 2, 3):
            model.config.total_ut_steps = model.model.total_ut_steps = depth
            model.early_exit_step = depth - 1
            short = model(tokens, use_cache=False).logits
            torch.testing.assert_close(short, model.lm_head(states[depth - 1]))
            changed = tokens.clone()
            changed[:, -1] = 9
            torch.testing.assert_close(short[:, :-1], model(changed, use_cache=False).logits[:, :-1])
