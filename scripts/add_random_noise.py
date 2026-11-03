import quiet_mode
import host_cpu
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import typer 
from transformers import GenerationConfig
from huggingface_hub import HfApi
from transformers import AutoConfig
import json 
import datetime
RELATIVE_NOISE = 0.01


def embedding_noise_std(weight: torch.Tensor, relative: float = RELATIVE_NOISE) -> float:
    rms = weight.detach().float().pow(2).mean().sqrt().item()
    if not math.isfinite(rms) or rms < 1e-6:
        rms = 1e-6
    return relative * rms


def add_embedding_noise(model, relative: float = RELATIVE_NOISE) -> float:
    with torch.no_grad():
        embeddings = model.get_input_embeddings()
        noise_std = embedding_noise_std(embeddings.weight, relative)
        noise = torch.randn_like(embeddings.weight) * noise_std
        embeddings.weight.add_(noise)
    return noise_std


def main(model_path: str, save_folder: str):
    host_cpu.apply("add_random_noise")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.generation_config = GenerationConfig(temperature=None, top_p=None)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Modifying input embeddings...", flush=True)
    noise_std = add_embedding_noise(model)
    print(f"Embedding noise std={noise_std:.3e} ({RELATIVE_NOISE} x rms)", flush=True)

    print(f"Saving modified model to {save_folder}...", flush=True)
    os.makedirs(save_folder, exist_ok=True)
    model.save_pretrained(save_folder)
    tokenizer.save_pretrained(save_folder)

if __name__ == "__main__":
    typer.run(main)
