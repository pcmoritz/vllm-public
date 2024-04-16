import numpy
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

scales_file = numpy.load("/tmp/scales.npz")
activation_scales = {"a1": scales_file['arr_0'], "a2": scales_file['arr_1'], "a3": scales_file['arr_0']}

def rewrite_safetensors(name):
    tensors = {}
    with safe_open(name, framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
            if "w1" in k or "w2" in k or "w3" in k:
                qtensor, scale = fp8_quantize(tensors[k])
                scale_name = k.removesuffix(".weight").replace("experts", "scales")
                print(f"scaling {k} with {scale_name}")
                tensors[scale_name] = scale
                tensors[k] = qtensor
                activation_name_parts = k.removesuffix(".weight").replace("w", "a").split(".")
                del activation_name_parts[-2]
                activation_scale_name = ".".join(activation_name_parts)
                print(f"activation_scale_name = {activation_scale_name}")
                tensors[activation_scale_name] = torch.tensor(activation_scales[activation_name_parts[-1]][int(activation_name_parts[2])])
    save_file(tensors, name)

for i in range(1, 20):
    filename = f"model-{i:05}-of-00019.safetensors"
    print(f"rewriting {filename}")
    rewrite_safetensors(filename)