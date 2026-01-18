import math
import torch


def token_level_log_prob(model, input_ids, prompt_len):
    with torch.no_grad():
        logits = model(input_ids).logits

    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]

    ans_log_probs = log_probs[:, prompt_len - 1 :, :]
    ans_target_ids = target_ids[:, prompt_len - 1 :]

    return torch.gather(
        ans_log_probs, 2, ans_target_ids.unsqueeze(-1)
    ).sum().item()


def compute_qaci(model, prompt_input_ids, unconditional_input_ids):
    log2 = math.log(2)

    prompt_len = prompt_input_ids.size(1) - unconditional_input_ids.size(1)

    logP_cond = token_level_log_prob(
        model, prompt_input_ids, prompt_len
    )
    logP_uncond = token_level_log_prob(
        model, unconditional_input_ids, 0
    )

    return (logP_cond - logP_uncond) / log2
