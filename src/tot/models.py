import os
import transformers
import torch

class StopOnXInput(transformers.StoppingCriteria):
    def __init__(self, tokenizer, inputAmount):
        self.tokenizer = tokenizer
        self.inputAmount = inputAmount

    def __call__(self, input_ids, scores, **kwargs):
        # Decode the generated tokens to text
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        # Input is seen in the prompt X. When the llm tries generating a new input, stop
        return generated_text.count("Input") == self.inputAmount

class StopOnEvaluation(transformers.StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.possibleEvaluationStops = {"sure": 5, "impossible": 5, "likely": 4}

    def __call__(self, input_ids, scores, **kwargs):
        # Decode the generated tokens to text
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        # When running with value prompt, sure, impossible, and likely are seen minus 1 as many times in possibleEvaluationStops
        # When see the threshold, stop generating
        return any(generated_text.count(word) == threshold for word, threshold in self.possibleEvaluationStops.items())

def gpt_24_proposal(prompt, pipeline, inputAmount, temperature=0.7, max_tokens=1000):
    return pipeline(
        prompt,
        max_new_tokens = max_tokens,
        temperature = temperature,
        num_return_sequences = 1,
        stopping_criteria = transformers.StoppingCriteriaList([StopOnXInput(tokenizer=pipeline.tokenizer, inputAmount=inputAmount)])
    )

def gpt_24_value(prompt, pipeline, lastStep, temperature=0.7, max_tokens=1000, n=1):
    return [gpt_24_value_query(prompt, pipeline, lastStep, temperature, max_tokens)[0]["generated_text"] for _ in range(n)]

def gpt_24_value_query(prompt, pipeline, lastStep, temperature, max_tokens):
    if lastStep:
        return pipeline(
            prompt,
            max_new_tokens = max_tokens,
            temperature = temperature,
            num_return_sequences = 1,
            stopping_criteria = transformers.StoppingCriteriaList([StopOnEvaluation(tokenizer=pipeline.tokenizer)])
        )

    return pipeline(
        prompt,
        max_new_tokens = max_tokens,
        temperature = temperature,
        num_return_sequences = 1,
        stopping_criteria = transformers.StoppingCriteriaList([StopOnEvaluation(tokenizer=pipeline.tokenizer)])
    )

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

def completions_with_backoff(pipeline, messages, temperature, max_tokens, n, stop):
    # If no stopping criteria
    if stop is None:
        return pipeline(
        messages,
        max_new_tokens = max_tokens,
        temperature = temperature,
        num_return_sequences = n
    )

    if stop == ["Input"]:
        return pipeline(
            messages,
            max_new_tokens = max_tokens,
            temperature = temperature,
            num_return_sequences = n,
            stopping_criteria = transformers.StoppingCriteriaList([StopOn3Input(tokenizer=pipeline.tokenizer)])
        )
    else:
        raise TypeError("Stopping condition not implemented")

def gpt_usage(backend="llama"):
    return {"completion_tokens": 0, "prompt_tokens": 0, "cost": 0}
