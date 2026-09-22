#!/usr/bin/env python3
"""Compare live SGLang generation evidence against the raw HF model."""
import argparse
import json
import math
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    messages = [{"role": "user", "content": "Say OK, thank you!"}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    response = requests.post(args.endpoint.rstrip("/") + "/generate", json={
        "input_ids": input_ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": args.max_new_tokens},
        "return_logprob": True,
        "stream": False,
    }, timeout=120)
    response.raise_for_status()
    output = response.json()
    meta = output.get("meta_info", {})
    rows = meta.get("output_token_logprobs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("SGLang did not return output_token_logprobs")
    output_ids = [int(row[1]) for row in rows]
    served = [float(row[0]) for row in rows]
    reported_ids = output.get("output_ids")
    if reported_ids is not None and reported_ids != output_ids:
        raise ValueError("output_ids disagrees with output_token_logprobs")
    decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
    text = output.get("text", "")
    if not isinstance(text, str) or not decoded.startswith(text):
        raise ValueError("text disagrees with decoded output token IDs")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True).to(args.device).eval()
    full_ids = torch.tensor([input_ids + output_ids], device=args.device)
    with torch.inference_mode():
        logits = model(full_ids).logits[0].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        positions = torch.arange(len(input_ids) - 1, len(full_ids[0]) - 1, device=args.device)
        raw = [float(logprobs[position, token_id])
               for position, token_id in zip(positions, output_ids)]
    deltas = [abs(float(raw_value) - served_value) for raw_value, served_value in zip(raw, served)]
    result = {
        "endpoint": args.endpoint,
        "model": args.model,
        "weight_version": meta.get("weight_version"),
        "input_tokens": len(input_ids),
        "output_tokens": len(output_ids),
        "output_token_ids": output_ids,
        "decoded": decoded,
        "serving_text": text,
        "finish_reason": meta.get("finish_reason"),
        "max_abs_logprob_delta": max(deltas, default=0.0),
        "mean_abs_logprob_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        "tolerance": args.tolerance,
        "passed": (meta.get("weight_version") is not None and all(math.isfinite(value) for value in raw)
                   and max(deltas, default=0.0) <= args.tolerance),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
