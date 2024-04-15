import torch

from vllm._C import ops

HIDDEN_SIZE = 4096
NUM_TOKENS = 2048

# M, _ = hidden_states.shape

hidden_states = torch.zeros((NUM_TOKENS, HIDDEN_SIZE), device="cuda", dtype=torch.float16)

intermediate_cache0 = torch.empty(hidden_states.shape,
                                  device=hidden_states.device,
                                  dtype=torch.float8_e4m3fn)

s = torch.ones(HIDDEN_SIZE, device="cuda", dtype=torch.float8_e4m3fn)

ops.scaled_fp8_quant(intermediate_cache0, hidden_states, s)

print("intermediate_cache0", intermediate_cache0)
