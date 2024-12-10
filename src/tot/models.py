import os
import backoff
import transformers
import torch

import sys

completion_tokens = prompt_tokens = 0

def gpt(prompt, model="llama", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def chatgpt(messages, model="llama", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        # log completion tokens
        completion_tokens += res["usage"]["completion_tokens"]
        prompt_tokens += res["usage"]["prompt_tokens"]
    return outputs

# @backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(model, messages, temperature, max_tokens, n, stop):
    if model == "llama":
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
    else:
        raise TypeError("Model type not accepted")

    if (stop is not None):
        print("Stopping criteria is not none")
        print(stop)
        print("------------------------------------------")
        sys.exit(0)

    return pipeline(
        messages,
        max_new_tokens = max_tokens,
        temperature = temperature,
        num_return_sequences = n,
    )
    # return openai.ChatCompletion.create(**kwargs)

def gpt_usage(backend="llama"):
    global completion_tokens, prompt_tokens
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": 0}
