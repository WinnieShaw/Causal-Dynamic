import argparse
import math
import time
import json
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================
# Prompt
# =========================
def make_prompt(text: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a precise summarization assistant.\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{text}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# =========================
# Question Pool Loader
# =========================
def load_questions_from_levels(base_dir, num_each_level=10):
    all_questions = []

    for lvl in range(1, 6):
        path = os.path.join(base_dir, f"level{lvl}.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        key = f"difficulty_level{lvl}"
        if key not in data:
            raise KeyError(f"Key {key} not found in {path}")

        questions = data[key][:num_each_level]
        all_questions.extend(questions)

    return all_questions


# =========================
# Utilities
# =========================
def bits(x_nat: float) -> float:
    return x_nat / math.log(2)


def eval_indices(t_max: int):
    if t_max <= 0:
        return []
    return list(range(0, t_max, 2))


# =========================
# Runner
# =========================
class PrefixScoreRunner:

    def __init__(self, checkpoint_path: str, device: str, dtype_str: str = "bfloat16"):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto",
        ).eval()

    def safe_encode(self, text: str):
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        enc["input_ids"] = enc["input_ids"].long()
        return enc

    def generate_greedy_ids(self, prompt_text: str, max_new_tokens: int):
        enc = self.safe_encode(prompt_text)
        Lp = enc["input_ids"].size(1)

        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        out_ids = out[0]
        gen_ids = out_ids[Lp:]
        return gen_ids, enc["input_ids"][0]

    def compute_logp_token_all(self, prompt_ids_1d, gen_ids_1d):
        prompt_ids = prompt_ids_1d.unsqueeze(0).to(self.device)
        gen_ids = gen_ids_1d.to(self.device)
        T = gen_ids.size(0)

        if T == 0:
            return torch.empty(0)

        if T == 1:
            inp = prompt_ids
        else:
            inp = torch.cat([prompt_ids, gen_ids[:-1].unsqueeze(0)], dim=1)

        with torch.no_grad():
            logits = self.model(inp).logits

        Lp = prompt_ids.size(1)
        logits_slice = logits[0, Lp - 1 : Lp - 1 + T, :]

        logp = torch.log_softmax(logits_slice, dim=-1)
        tok_lp = logp.gather(1, gen_ids.unsqueeze(-1)).squeeze(-1)
        return tok_lp.detach().cpu()

    def compute_logp_prefix_all(self, logp_token_all_cpu):
        if logp_token_all_cpu.numel() == 0:
            return torch.empty(0)
        return torch.cumsum(logp_token_all_cpu, dim=0)


# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", type=str, default="../Qwen2.5-7B-Instruct")
    ap.add_argument("--gen_max", type=int, default=128)
    ap.add_argument("--max_tokens_eval", type=int, default=60)
    ap.add_argument("--out_csv", type=str, default="prefix_score_QPOOL_even.csv")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument(
        "--question_dir",
        type=str,
        default="question_pools/difficulty_levels",
        help="Directory containing level1.json ... level5.json",
    )
    ap.add_argument("--num_each_level", type=int, default=10)
    return ap.parse_args()


# =========================
# Main
# =========================
def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    runner = PrefixScoreRunner(
        checkpoint_path=args.checkpoint_path,
        device=device,
        dtype_str=args.dtype,
    )

    Q_POOL = load_questions_from_levels(
        args.question_dir, num_each_level=args.num_each_level
    )
    print(f"Loaded {len(Q_POOL)} questions from {args.question_dir}")

    records = []
    t0 = time.time()

    for i, Q in enumerate(Q_POOL):
        print(f"\n== Q{i} ==")

        prompt_text = make_prompt(Q)
        gen_ids, prompt_ids_1d = runner.generate_greedy_ids(prompt_text, args.gen_max)
        T = gen_ids.size(0)

        if T < 2:
            continue

        logp_token_all = runner.compute_logp_token_all(prompt_ids_1d, gen_ids)
        logp_prefix_all = runner.compute_logp_prefix_all(logp_token_all)

        t_max = min(args.max_tokens_eval, T)
        ts = eval_indices(t_max)

        for t in ts:
            eps_id = int(gen_ids[t].item())

            logp_token_nat = float(logp_token_all[t].item())
            logp_prefix_nat = float(logp_prefix_all[t].item())

            score_bits = bits(logp_token_nat - logp_prefix_nat)

            records.append({
                "q_index": i,
                "t": t,
                "score_prefix_bits": score_bits,
                "logp_token_bits": bits(logp_token_nat),
                "logp_prefix_bits": bits(logp_prefix_nat),
                "eps_id": eps_id,
            })

    df = pd.DataFrame(records)
    df.to_csv(args.out_csv, index=False)

    print(f"\nSaved: {args.out_csv} | n={len(df)}")
    print("Total time (min):", (time.time() - t0) / 60.0)


if __name__ == "__main__":
    main()