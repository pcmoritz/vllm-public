import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from vllm.model_executor.layers.fused_moe import fused_moe
import torch
import torch.nn.functional as F
import sys


def main():
    method = fused_moe
    run_grid(method=method)


def run_grid(method):
    bs_grid = [1, 2, 4, 8, 16, 32, 64, 128, 1024, 2048, 4096, 8192]
    d_model = 4096
    num_total_experts = 8
    top_k = 2
    tp_size_grid = [8]
    model_intermediate_size = 14336
    num_layers = 32
    num_calls = 100

    num_warmup_trials = 1
    num_trials = 1

    for tp_size in tp_size_grid:
        for bs in bs_grid:
            print(f'{tp_size=} {bs=}')
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
                    method=method,
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
                    method=method,
                )

                kernel_dur_us = 1000 * kernel_dur_ms
                model_dur_ms = kernel_dur_ms * num_layers

                print(
                    f'{kernel_dur_us=:.1f} {model_dur_ms=:.1f} {bs=} {tp_size=} {top_k=} {num_total_experts=} {d_model=} {model_intermediate_size=} {num_layers=}'
                )


def run_timing(num_calls: int, bs: int, d_model: int, num_total_experts: int,
               top_k: int, tp_size: int, model_intermediate_size: int,
               method) -> float:
    shard_intermediate_size = model_intermediate_size // tp_size

    hidden_states = torch.rand(
        (bs, d_model),
        device="cuda:0",
        dtype=torch.bfloat16,
    )

    ws = torch.rand(
        (num_total_experts, 2 * shard_intermediate_size, d_model),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    w2s = torch.rand(
        (num_total_experts, d_model, shard_intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    routing_weights = F.softmax(torch.rand(
        (num_calls, bs, top_k),
        device=hidden_states.device,
        dtype=torch.float32,
    ),
                                dim=-1)

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
        hidden_states = method(
            hidden_states=hidden_states,
            w1=ws,
            w2=w2s,
            topk_weights=routing_weights[i],
            topk_ids=selected_experts[i],
            inplace=True,
        )
    end_event.record()
    end_event.synchronize()

    dur_ms = start_event.elapsed_time(end_event) / num_calls
    return dur_ms