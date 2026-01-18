import math
import torch


def logsumexp(log_values):
    m = max(log_values)
    return m + math.log(sum(math.exp(v - m) for v in log_values))


def prefix_log_prob(model, prompt_ids, prefix_ids):
    if prefix_ids.size(1) == 0:
        return 0.0

    inp = torch.cat([prompt_ids, prefix_ids[:, :-1]], dim=1)

    with torch.no_grad():
        logits = model(inp).logits

    tail = logits[:, -prefix_ids.size(1):, :]
    log_probs = torch.log_softmax(tail, dim=-1)

    return log_probs.gather(2, prefix_ids.unsqueeze(-1)).sum().item()


def compute_cci(model, token_id, prefix_ids, prompt_id_map):
    questions = list(prompt_id_map.keys())
    q_prob = 1.0 / len(questions)

    # log P(a_t | a_<t)
    den_terms = []
    for q in questions:
        ctx = prompt_id_map[q]
        inp = torch.cat([ctx, prefix_ids.squeeze(0)], dim=0)

        with torch.no_grad():
            logits = model(inp.unsqueeze(0)).logits
            logp = torch.log_softmax(logits[:, -1, :], dim=-1)[0, token_id].item()

        den_terms.append(math.log(q_prob) + logp)

    log_next = logsumexp(den_terms)

    # log P(a_<=t)
    full_prefix_ids = torch.cat(
        [prefix_ids, torch.tensor([[token_id]], device=prefix_ids.device)],
        dim=1,
    )

    num_terms = []
    for q in questions:
        lp = prefix_log_prob(
            model,
            prompt_id_map[q].unsqueeze(0),
            full_prefix_ids,
        )
        num_terms.append(math.log(q_prob) + lp)

    log_prefix = logsumexp(num_terms)

    return (log_next - log_prefix) / math.log(2)
