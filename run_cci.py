import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from cci_computation import compute_cci


def make_prompt(question):
    return (
        "<|im_start|>system\n"
        "You are a precise and analytical assistant.\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{question}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def get_input_device(model):
    """Return the device where input tensors should be placed."""
    return next(model.parameters()).device


def tokenize_prompt(tokenizer, question, device):
    """Tokenize a formatted prompt and return a 1D tensor of token ids."""
    prompt = make_prompt(question)
    return tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0].to(device)


def generate_answer(model, tokenizer, question, max_new_tokens=128):
    """
    Generate an answer and return both the decoded text and the generated token ids
    excluding the prompt tokens.
    """
    input_device = get_input_device(model)
    prompt = make_prompt(question)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    prompt_len = inputs["input_ids"].size(1)

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_cfg)

    generated_ids = outputs[0, prompt_len:]
    answer_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return answer_text, generated_ids


if __name__ == "__main__":
    model_path = "/path/to/your/model"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    input_device = get_input_device(model)

    questions = [
        "If human sleep requirements were reduced to just two hours per day, how might this change patterns of productivity, social interaction, and long-term health?",
        "If Earth’s gravity were suddenly reduced by half, what effects might this have on human physiology, architectural design, and daily activities?",
    ]

    prompt_items = [
        {
            "question": q,
            "ctx": tokenize_prompt(tokenizer, q, input_device),
        }
        for q in questions
    ]

    target_question = questions[0]
    answer_text, generated_ids = generate_answer(model, tokenizer, target_question)

    print("Generated answer:")
    print(answer_text)
    print()

    for t in range(1, generated_ids.size(0)):
        prefix_ids = generated_ids[:t].unsqueeze(0)
        token_id = generated_ids[t].item()

        cci = compute_cci(
            model=model,
            token_id=token_id,
            prefix_ids=prefix_ids,
            prompt_items=prompt_items,
            mode="sample_uniform",
        )

        print(f"t={t:03d}  CCI={cci:+.4f}")
