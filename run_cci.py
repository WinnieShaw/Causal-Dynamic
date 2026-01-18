import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from cci_computation import compute_cci

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_prompt(question):
    return (
        "<|im_start|>system\n"
        "You are a precise and analytical assistant.\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{question}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def generate_answer(model, tokenizer, question, max_new_tokens=128):
    prompt = make_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gen_cfg)

    return tokenizer.decode(out[0], skip_special_tokens=True)


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

    questions = [
        "If human sleep requirements were reduced to just two hours per day, how might this change patterns of productivity, social interaction, and long-term health?",
        "If Earth’s gravity were suddenly reduced by half, what effects might this have on human physiology, architectural design, and daily activities?"
    ]

    prompt_id_map = {
        q: tokenizer(make_prompt(q), return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
        for q in questions
    }

    q = questions[0]
    answer = generate_answer(model, tokenizer, q)
    gen_ids = tokenizer(answer, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)[0]

    for t in range(1, len(gen_ids)):
        prefix_ids = gen_ids[:t].unsqueeze(0)
        token_id = gen_ids[t].item()

        cci = compute_cci(
            model=model,
            token_id=token_id,
            prefix_ids=prefix_ids,
            prompt_id_map=prompt_id_map,
        )

        print(f"t={t:03d}  CCI={cci:+.4f}")
