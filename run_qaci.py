import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from qaci_computation import compute_qaci

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_prompt(question):
    return (
        "<|im_start|>system\n"
        "You are a precise and analytical assistant.\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{question}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def generate_answer(model, tokenizer, question, max_new_tokens=256):
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

    question = "If human sleep requirements were reduced to just two hours per day, how might this change patterns of productivity, social interaction, and long-term health?"
    answer = generate_answer(model, tokenizer, question)

    prompt_text = make_prompt(question)

    prompt_input_ids = tokenizer(
        prompt_text + answer,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"].to(device)

    unconditional_input_ids = tokenizer(
        answer,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"].to(device)

    qaci = compute_qaci(
        model=model,
        prompt_input_ids=prompt_input_ids,
        unconditional_input_ids=unconditional_input_ids,
    )

    print(f"QACI = {qaci:+.4f}")
