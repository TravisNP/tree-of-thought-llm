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

        # When running in propose prompt mode, "input" is passed as a stopping condition. Input is seen in the prompt twice.
        # When the llm tries generating a new input, stop
        if "Input" in self.stopping_tokens:
            return generated_text.count("Input") == 3

        return any(token in generated_text for token in self.stopping_tokens)

def gpt(prompt, pipeline, temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    return chatgpt(prompt, pipeline=pipeline, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def chatgpt(messages, pipeline, temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    outputs = []
    print("---------------------------------")
    while n > 0:
        print("n: ", n)
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(pipeline=pipeline, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)[0]["generated_text"].split("\n")[12:-1]
        outputs.extend(res)
        # log completion tokens
    return outputs

# @backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(pipeline, messages, temperature, max_tokens, n, stop):

    # If no stopping criteria
    if stop is None:
        return pipeline(
        messages,
        max_new_tokens = max_tokens,
        temperature = temperature,
        num_return_sequences = n
    )

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
