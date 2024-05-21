import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import triton
from tqdm import tqdm

from vllm.model_executor.layers.fused_silu.fused_silu import fused_silu

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(dtype: str):
    method = fused_silu
    for bs in [
            1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536,
            2048, 3072, 4096
    ]:
        run_grid(bs, method=method, dtype=dtype)

def run_grid(bs, method, dtype: str):
    d_model = 8192
    num_total_experts = 8
    top_k = 2
    tp_size = 4
    model_intermediate_size = 28672
    num_layers = 32
    num_calls = 100

    num_warmup_trials = 1
    num_trials = 1

    configs = []

    for block_size_n in [32, 64, 128, 256]:
        for block_size_m in [16, 32, 64]:
            for block_size_k in [32, 64, 128]:
                for group_size_m in [1, 16, 32, 64]:
                    for num_warps in [2, 4, 8]:
                        for num_stages in [2, 3, 4, 5, 6]:
                            configs.append({
                                "BLOCK_SIZE_M": block_size_m,
                                "BLOCK_SIZE_N": block_size_n,
                                "BLOCK_SIZE_K": block_size_k,
                                "GROUP_SIZE_M": group_size_m,
                                "num_warps": num_warps,
                                "num_stages": num_stages,
                            })

    best_config = None
    best_time_us = 1e20

    print(f'{tp_size=} {bs=}')

    for config in tqdm(configs):
        # warmup
        try:
            for _ in range(num_warmup_trials):
                run_timing(
                    num_calls=num_calls,
                    bs=bs,
                    d_model=d_model,
                    tp_size=tp_size,
                    model_intermediate_size=model_intermediate_size,
                    method=method,
                    config=config,
                    dtype=dtype,
                )
        except triton.runtime.autotuner.OutOfResources:
            continue

        # trial
        for _ in range(num_trials):
            kernel_dur_ms = run_timing(
                num_calls=num_calls,
                bs=bs,
                d_model=d_model,
                tp_size=tp_size,
                model_intermediate_size=model_intermediate_size,
                method=method,
                config=config,
                dtype=dtype,
            )

            kernel_dur_us = 1000 * kernel_dur_ms
            model_dur_ms = kernel_dur_ms * num_layers

            if kernel_dur_us < best_time_us:
                best_config = config
                best_time_us = kernel_dur_us

                tqdm.write(
                    f'{kernel_dur_us=:.1f} {model_dur_ms=:.1f}'
                    f' {bs=} {tp_size=} {top_k=} {num_total_experts=} '
                    f'{d_model=} {model_intermediate_size=} {num_layers=}')

    print("best_time_us", best_time_us)
    print("best_config", best_config)

    filename = "config.json"
    print(f"writing config to file {filename}")
    existing_content = {}
    if os.path.exists(filename):
        with open(filename, "r") as f:
            existing_content = json.load(f)
    existing_content[str(bs)] = best_config
    with open(filename, "w") as f:
        json.dump(existing_content, f, indent=4)
        f.write("\n")


def run_timing(num_calls: int, bs: int, d_model: int,
               tp_size: int, model_intermediate_size: int, method,
               config, dtype: str) -> float:
    shard_intermediate_size = model_intermediate_size // tp_size

    hidden_states = torch.rand(
        (bs, d_model),
        device="cuda:0",
        dtype=torch.float16,
    )

    w1 = torch.rand(
        (d_model, shard_intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    w2 = torch.rand(
        (d_model, shard_intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for i in range(num_calls):
        result = method(
            hidden_states,
            w1,
            w2,
            override_config=config,
        )
    end_event.record()
    end_event.synchronize()

    dur_ms = start_event.elapsed_time(end_event) / num_calls
    return dur_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='benchmark_mixtral_moe',
        description='Benchmark and tune the fused_silu kernel',
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['float8', 'float16'],
        help='Data type used for fused_silu kernel computations',
    )
    args = parser.parse_args()
    sys.exit(main(args.dtype))
