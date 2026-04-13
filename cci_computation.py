import math
from collections import Counter

import torch


def logsumexp(log_values):
    """Compute log(sum(exp(log_values))) in a numerically stable way."""
    m = max(log_values)
    return m + math.log(sum(math.exp(v - m) for v in log_values))


def prefix_log_prob(model, prompt_ids, prefix_ids):
    """
    Compute the log-probability of an entire prefix conditioned on a prompt.

    Args:
        model: Autoregressive language model returning `.logits`.
        prompt_ids: Tensor of shape [1, prompt_len].
        prefix_ids: Tensor of shape [1, prefix_len].

    Returns:
        A float scalar:
            log P(prefix | prompt)
    """
    if prefix_ids.size(1) == 0:
        return 0.0

    inputs = torch.cat([prompt_ids, prefix_ids[:, :-1]], dim=1)

    with torch.no_grad():
        logits = model(inputs).logits

    prefix_len = prefix_ids.size(1)
    tail_logits = logits[:, -prefix_len:, :]
    log_probs = torch.log_softmax(tail_logits, dim=-1)

    token_log_probs = log_probs.gather(2, prefix_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum().item()


def build_sample_log_weights(prompt_items, mode="sample_uniform"):
    """
    Build log-weights over prompt samples.

    Args:
        prompt_items: A list of dictionaries, where each item has:
            {
                "question": <hashable question identifier or text>,
                "ctx": 1D tensor of token ids
            }
            Duplicate questions are allowed.
        mode: Weighting scheme.
            - "sample_uniform":
                Assign equal weight to each sample.
                If a question appears multiple times, it contributes multiple times.
            - "question_uniform":
                Assign equal total weight to each unique question.
                If a question appears k times, its total mass is split equally
                across its k associated samples.

    Returns:
        A list of log-weights, one per sample in prompt_items.
    """
    num_items = len(prompt_items)
    if num_items == 0:
        raise ValueError("prompt_items must not be empty.")

    if mode == "sample_uniform":
        return [math.log(1.0 / num_items)] * num_items

    if mode == "question_uniform":
        counts = Counter(item["question"] for item in prompt_items)
        num_unique_questions = len(counts)

        log_weights = []
        for item in prompt_items:
            weight = 1.0 / (num_unique_questions * counts[item["question"]])
            log_weights.append(math.log(weight))
        return log_weights

    raise ValueError(f"Unknown weighting mode: {mode}")


def compute_cci(model, token_id, prefix_ids, prompt_items, mode="sample_uniform"):
    """
    Compute the CCI value for a candidate next token.

    Args:
        model: Autoregressive language model returning `.logits`.
        token_id: Integer id of the candidate next token a_t.
        prefix_ids: Tensor of shape [1, t-1], representing a_<t.
        prompt_items: A list of dictionaries:
            [
                {"question": q1, "ctx": ctx1},
                {"question": q2, "ctx": ctx2},
                {"question": q1, "ctx": ctx3},
                ...
            ]
            Here, duplicate questions are allowed.
            Each `ctx` must be a 1D tensor of token ids.
        mode: Weighting scheme for P(Q).
            - "sample_uniform"
            - "question_uniform"

    Returns:
        A float scalar:
            CCI = log2 P(a_t | a_<t) - log2 P(a_<=t)
    """
    log_q_weights = build_sample_log_weights(prompt_items, mode=mode)

    denominator_terms = []
    for item, log_weight in zip(prompt_items, log_q_weights):
        ctx = item["ctx"]
        inputs = torch.cat([ctx, prefix_ids.squeeze(0)], dim=0)

        with torch.no_grad():
            logits = model(inputs.unsqueeze(0)).logits
            next_token_log_prob = torch.log_softmax(logits[:, -1, :], dim=-1)[0, token_id].item()

        denominator_terms.append(log_weight + next_token_log_prob)

    log_p_next_given_prefix = logsumexp(denominator_terms)

    full_prefix_ids = torch.cat(
        [prefix_ids, torch.tensor([[token_id]], device=prefix_ids.device)],
        dim=1,
    )

    numerator_terms = []
    for item, log_weight in zip(prompt_items, log_q_weights):
        log_prob_full_prefix = prefix_log_prob(
            model=model,
            prompt_ids=item["ctx"].unsqueeze(0),
            prefix_ids=full_prefix_ids,
        )
        numerator_terms.append(log_weight + log_prob_full_prefix)

    log_p_full_prefix = logsumexp(numerator_terms)

    return (log_p_next_given_prefix - log_p_full_prefix) / math.log(2)
