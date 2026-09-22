import torch
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache, MinistralConfig


def load_model(name, device, dtype=torch.bfloat16):
    config = AutoConfig.from_pretrained(name)
    # The 2410 checkpoint still advertises Mistral, which ignores layer_types.
    # Ministral uses the same parameter names and honors hybrid attention.
    if config.model_type == "mistral" and getattr(config, "layer_types", None):
        revision = getattr(config, "_commit_hash", None)
        data = config.to_dict()
        data.pop("model_type", None)
        data["architectures"] = ["MinistralForCausalLM"]
        config = MinistralConfig.from_dict(data)
        config._commit_hash = revision
    return AutoModelForCausalLM.from_pretrained(
        name, config=config, dtype=dtype, device_map=device, attn_implementation="sdpa",
    ).eval().requires_grad_(False)


def tokenizer_template(tokenizer, sample, args):
    system_prompt = args.system_prompt
    context = sample.get("context", "")
    question = sample.get("question", "")
    user_content = f"{context}\n\n{question}\n\n"

    if tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

    return f"{system_prompt}\n\n{user_content}"

def build_kv_cache(
    model, 
    input_ids=None, 
    input_embeds=None, 
    use_cache=True,
    return_dict=True, 
    required_grad=False,
):
    assert (input_ids is None) != (input_embeds is None)
    with torch.set_grad_enabled(required_grad):
        outputs = model.model(
            input_ids=input_ids,
            inputs_embeds=input_embeds, 
            use_cache=use_cache,
            # Preserve all positions even for sliding-window models. The model's
            # attention mask still enforces the window; cache transforms need the
            # full sequence rather than a rolling, truncated cache.
            past_key_values=DynamicCache() if use_cache else None,
            return_dict=return_dict, 
        )
        return outputs.past_key_values


def greedy_decode(model, tokenizer, input_ids, cache, args):
    from src.kvcache import cache_clone

    cache = cache_clone(cache)
    eos_id = model.generation_config.eos_token_id
    stop_ids = set()
    tau = float(getattr(args, "tau", 0.0))

    if tau < 0:
        raise ValueError("tau must be non-negative.")
    
    if isinstance(eos_id, list):
        stop_ids.update(eos_id)
    elif eos_id is not None:
        stop_ids.add(eos_id)
    if tokenizer.pad_token_id is not None:
        stop_ids.add(tokenizer.pad_token_id)
        
    next_input_ids = input_ids.to(model.device)
    if next_input_ids.ndim == 1:
        next_input_ids = next_input_ids.unsqueeze(0)
    
    output_ids = torch.empty(args.max_new_tokens, dtype=torch.long, device=model.device)
    stop_tensor = torch.tensor(list(stop_ids), device=model.device) if stop_ids else None
    gen_len = 0
    
    with torch.inference_mode():
        for i in range(args.max_new_tokens):
            outputs = model(
                input_ids=next_input_ids,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            if tau == 0:
                next_input_ids = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_token_logits / tau, dim=-1)
                next_input_ids = torch.multinomial(probs, num_samples=1)
            if stop_tensor is not None and (next_input_ids[0, 0] == stop_tensor).any():
                break
            output_ids[i] = next_input_ids[0, 0]
            gen_len = i + 1

    if gen_len == 0:
        return ""

    return tokenizer.decode(output_ids[:gen_len].tolist(), skip_special_tokens=True)
