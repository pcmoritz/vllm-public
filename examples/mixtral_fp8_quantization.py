from safetensors import safe_open
from safetensors.torch import save_file
import torch

def fp8_quantize(weight, qdtype=torch.float8_e4m3fn):
    finfo = torch.finfo(qdtype)
    # Calculate the scale as dtype max divided by absmax
    scale = finfo.max / weight.abs().max().clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    qweight = (weight * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    qweight = qweight.to(qdtype)
    scale = scale.float().reciprocal()
    return qweight, scale

def activation_scale(activation, qdtype=torch.float8_e4m3fn):
    finfo = torch.finfo(qdtype)
    return weight.abs().max().clamp(min=1e-12) / finfo.max

activation_scales = torch.load("/home/ray/default/mixtral_scales.pth")

def rewrite_safetensors(name):
    tensors = {}
    with safe_open(name, framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
            if "w1" in k or "w2" in k or "w3" in k:
                qtensor, scale = fp8_quantize(tensors[k])
                name_parts = k.split(".")
                scale_name = "model.layers." + name_parts[2] + ".block_sparse_moe.scales." + name_parts[-2]
                print(f"scaling {k} with {scale_name}")
                tensors[scale_name] = scale
                tensors[k] = qtensor
                activation_scale_name = "model.layers." + name_parts[2] + ".block_sparse_moe.scales." + name_parts[-2].replace("w", "a")
                tensors[activation_scale_name] = activation_scale(activation_scales[k])
    save_file(tensors, name)

for i in range(1, 20):
    filename = f"model-{i:05}-of-00019.safetensors"
    print(f"rewriting {filename}")
    rewrite_safetensors(filename)