from vllm.model_executor.layers.fused_moe import fused_moe, fused_moe_cuda
from vllm.model_executor.layers.activation import SiluAndMul
import torch
import torch.nn.functional as F
import sys

def single_run():
    bs = 4
    d_model = 4096
    num_total_experts = 8
    top_k = 2
    tp_size = 8
    model_intermediate_size = 14336
    num_layers = 32
    num_calls = 10000

    num_warmup_trials = 1
    num_trials = 1

    # warmup
    print(f'warming up')
    for _ in range(num_warmup_trials):
        run_timing(
            num_calls=num_calls,
            bs=bs,
            d_model=d_model,
            num_total_experts=num_total_experts,
            top_k=top_k,
            tp_size=tp_size,
            model_intermediate_size=model_intermediate_size,
        )

    # trial
    print(f'benchmarking')
    for _ in range(num_trials):
        kernel_dur_ms = run_timing(
            num_calls=num_calls,
            bs=bs,
            d_model=d_model,
            num_total_experts=num_total_experts,
            top_k=top_k,
            tp_size=tp_size,
            model_intermediate_size=model_intermediate_size,
        )

        kernel_dur_us = 1000 * kernel_dur_ms
        model_dur_ms = kernel_dur_ms * num_layers

        print(f'{kernel_dur_us=:.1f} {model_dur_ms=:.1f} {bs=} {tp_size=} {top_k=} {num_total_experts=} {d_model=} {model_intermediate_size=} {num_layers=}')

def torch_moe(a, w1, w2, topk_weight, topk_ids):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk_ids.shape[1], 1).reshape(-1, D)
    out = torch.zeros(B * topk_ids.shape[1],
                      w2.shape[1],
                      dtype=a.dtype,
                      device=a.device)
    topk_ids = topk_ids.view(-1)
    topk_weight = topk_weight.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul()(
                a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)
    return (out.view(B, -1, w2.shape[1]) *
            topk_weight.view(B, -1, 1)).sum(dim=1)

def run_timing(num_calls: int, bs: int, d_model: int, num_total_experts: int, top_k: int, tp_size: int, model_intermediate_size: int) -> float:
    shard_intermediate_size = model_intermediate_size // tp_size

    hidden_states = torch.rand(
        (bs, d_model),
        device="cuda:0",
        dtype=torch.bfloat16,
    )

    permutation = torch.zeros(bs * top_k, device=hidden_states.device, dtype=torch.int32)
    hidden_states2 = hidden_states[permutation,:]

    ws = 0.01 * torch.rand(
        (num_total_experts, 2 * shard_intermediate_size, d_model),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    w2s = 0.01 * torch.rand(
        (num_total_experts, d_model, shard_intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    routing_weights = F.softmax(torch.rand(
        (num_calls, bs, top_k),
        device=hidden_states.device,
        dtype=torch.float32,
    ), dim=-1)

    selected_experts = torch.randint_like(
        routing_weights,
        low=0,
        high=num_total_experts,
        device=hidden_states.device,
        dtype=torch.int64,
    )

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for i in range(num_calls):
        hidden_states = fused_moe_cuda(
        # hidden_states = fused_moe(
            hidden_states,
            ws,
            w2s,
            routing_weights[i],
            selected_experts[i],
            inplace=True,
        )
    end_event.record()
    end_event.synchronize()

    """
    hidden_states = fused_moe(
        hidden_states,
        ws,
        w2s,
        routing_weights[0],
        selected_experts[0],
        inplace=True,
    )

    # Test if it was correct:
    hidden_states_copy = torch_moe(
        hidden_states_copy,
        ws,
        w2s,
        routing_weights[0],
        selected_experts[0],
    )

    import IPython
    IPython.embed()
    """

    dur_ms = start_event.elapsed_time(end_event) / num_calls
    return dur_ms

if __name__ == "__main__":
    sys.exit(single_run())