import os
import shutil
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_model_and_tokenizer(model_id_or_dir: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager" if device=='cpu' else "flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    return model, tokenizer


def save_model_and_tokenizer(model, tokenizer, output_dir) -> Path:
    """Write the current weights + tokenizer to output_dir as a loadable checkpoint.

    The result is a plain HF model directory, so it round-trips through
    get_model_and_tokenizer() and can be passed straight to `vllm serve`.

    Files are staged in a sibling temp dir and moved in one at a time with
    os.replace, which is atomic per file on the same filesystem. Writing
    save_pretrained() output directly into output_dir would leave a torn
    checkpoint behind if the process died mid-write -- a real risk here, since
    model.safetensors is ~3 GB and the same directory is what vLLM serves from.
    Unrelated files already in output_dir (eval .jsonl, etc.) are left alone.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # torch.compile / DDP wrap the real module; save_pretrained lives on the inner one.
    model = getattr(model, "_orig_mod", model)
    model = getattr(model, "module", model)

    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        model.save_pretrained(staging)
        tokenizer.save_pretrained(staging)
        for staged in sorted(staging.iterdir()):
            os.replace(staged, output_dir / staged.name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return output_dir
