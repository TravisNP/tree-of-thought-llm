import os
import backoff
import transformers
import torch

class StopOnNewline(transformers.StoppingCriteria):
    def __init__(self, tokenizer, stopping_tokens):
        self.tokenizer = tokenizer
        self.stopping_tokens = stopping_tokens

    def __call__(self, input_ids, scores, **kwargs):
        # Decode the generated tokens to text
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        if "Input" in self.stopping_tokens and generated_text.count("Input") == 3:
                return True

        return any(token in generated_text for token in self.stopping_tokens)
        # return '\n' in generated_text

def gpt(prompt, model="llama", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    return chatgpt(prompt, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def chatgpt(messages, model="llama", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    outputs = []
    print("---------------------------------")
    while n > 0:
        print("n: ", n)
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        outputs.append(res[0]["generated_text"])
        print("res: ", res)
        # log completion tokens
    print("Outputs: ", outputs)
    return outputs

# @backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(model, messages, temperature, max_tokens, n, stop):
    if model == "llama":
        model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
    else:
        raise TypeError("Model type not accepted")

    # If no stopping criteria
    if stop is None:
        return pipeline(
        messages,
        max_new_tokens = max_tokens,
        temperature = temperature,
        num_return_sequences = n
    )

    # If stopping criteria
    stop_criteria = StopOnNewline(tokenizer=pipeline.tokenizer, stopping_tokens = stop)
    return pipeline(
        messages,
        max_new_tokens = max_tokens,
        temperature = temperature,
        num_return_sequences = n,
        stopping_criteria = transformers.StoppingCriteriaList([stop_criteria])
    )
    # return openai.ChatCompletion.create(**kwargs)

def gpt_usage(backend="llama"):
    return {"completion_tokens": 0, "prompt_tokens": 0, "cost": 0}
