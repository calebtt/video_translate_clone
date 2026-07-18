"""Machine translation for transcript segments."""

from __future__ import annotations

import gc
import logging
from typing import Any, Dict, List

logger = logging.getLogger("vtclone")


def translate_segments(
    segments: List[Dict[str, Any]],
    mt_model: str,
    device: str = "cuda",
    batch_size: int = 8,
    use_pipeline: bool = False,
) -> None:
    texts = [s.get("text") or "" for s in segments]
    if not any(t.strip() for t in texts):
        logger.warning("No text to translate")
        for s in segments:
            s["translated_text"] = ""
        return

    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers/torch not installed. Install with: pip install torch transformers"
        ) from e

    logger.info("Loading translation model: %s", mt_model)
    torch_device = 0 if (device.startswith("cuda") and torch.cuda.is_available()) else -1
    if torch_device == -1 and device.startswith("cuda"):
        logger.warning("CUDA requested but not available, using CPU for MT")

    if use_pipeline:
        from transformers import pipeline

        translator = pipeline("translation", model=mt_model, device=torch_device)
        outs = []
        n_batches = max(1, (len(texts) + batch_size - 1) // batch_size)
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            outs.extend(translator(chunk))
            logger.info("Translated batch %s/%s", i // batch_size + 1, n_batches)
        for s, o in zip(segments, outs):
            s["translated_text"] = (o.get("translation_text") or "").strip()
        return

    tok = AutoTokenizer.from_pretrained(mt_model)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(mt_model)
    if torch_device >= 0:
        mdl = mdl.to("cuda")
    mdl.eval()

    outs: List[str] = []
    n_batches = max(1, (len(texts) + batch_size - 1) // batch_size)
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
            if torch_device >= 0:
                enc = {k: v.to("cuda") for k, v in enc.items()}
            gen = mdl.generate(**enc, max_new_tokens=256)
            dec = tok.batch_decode(gen, skip_special_tokens=True)
            outs.extend([d.strip() for d in dec])
            logger.info("Translated batch %s/%s", i // batch_size + 1, n_batches)

    for s, t in zip(segments, outs):
        s["translated_text"] = t

    del mdl
    del tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
