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
    bs = 1
    # bs_grid = [1, 2, 4, 8, 16, 32, 64, 128, 1024, 2048, 4096, 8192]
    d_model = 4096
    num_total_experts = 8
    top_k = 2
    tp_size_grid = [2]
    model_intermediate_size = 14336
    num_layers = 32
    num_calls = 100

    num_warmup_trials = 1
    num_trials = 1

    configs = []
    for block_size_n1 in [8, 16]:
        for block_size_n2 in [8, 16]:
            for block_size_k1 in [64, 128, 256]:
                for block_size_k2 in [64, 128, 256]:
                    configs.append(
                        {
                            "BLOCK_SIZE_M": 1,
                            "BLOCK_SIZE_N1": block_size_n1,
                            "BLOCK_SIZE_N2": block_size_n2,
                            "BLOCK_SIZE_K1": block_size_k1,
                            "BLOCK_SIZE_K2": block_size_k2,
                            "GROUP_SIZE_M1": 1,
                            "GROUP_SIZE_M2": 1,
                        }
                    )

    for tp_size in tp_size_grid:
        # for bs in bs_grid:
        for config in configs:
            print(f'{tp_size=} {bs=}')
            print(f'{config}')
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
                    config=config,
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
               method, config) -> float:
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

    gating_output = F.softmax(
        torch.rand(
            (num_calls, bs, num_total_experts),
            device=hidden_states.device,
            dtype=torch.float32,
        ),
    dim=-1)


    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for i in range(num_calls):
        hidden_states = method(
            hidden_states=hidden_states,
            w1=ws,
            w2=w2s,
            gating_output=gating_output[i],
            topk=2,
            renormalize=True,
            inplace=True,
            config=config,
        )
    end_event.record()
    end_event.synchronize()

    dur_ms = start_event.elapsed_time(end_event) / num_calls
    return dur_ms


if __name__ == "__main__":
    sys.exit(main())
