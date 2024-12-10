import os
import backoff
import transformers
import torch

class StopOnNewline(transformers.StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Decode the generated tokens to text
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # Check if the generated text contains a newline character
        return '\n' in generated_text

def gpt(prompt, model="llama", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def chatgpt(messages, model="llama", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    outputs = []
    print("---------------------------------")
    while n > 0:
        print("n: ", n)
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        print("res: ", res)
        outputs.append(res[0]["generated_text"][0]["content"])
        # log completion tokens
    print("Outputs: ", outputs)
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

    stop_criteria = StopOnNewline(tokenizer=pipeline.tokenizer)

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
