import os
import transformers
import torch
from transformers import StoppingCriteriaList, StopStringCriteria

class Model():
    def __init__(self, model_pipeline, model_id):
        self.model_pipeline = model_pipeline
        self.model_id = model_id
        self.stopLikelySureImp = StopStringCriteria(tokenizer=model_pipeline.tokenizer, stop_strings=["likely", "sure", "impossible"])
        self.stopInput = StopStringCriteria(tokenizer=model_pipeline.tokenizer, stop_strings=["Input"])

class StopOnXInput(transformers.StoppingCriteria):
    def __init__(self, tokenizer, inputAmount):
        self.tokenizer = tokenizer
        self.inputAmount = inputAmount

    def __call__(self, input_ids, scores, **kwargs):
        # Decode the generated tokens to text
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        # Input is seen in the prompt X many times. When the llm tries generating a new input, stop
        return generated_text.count("Input") == self.inputAmount

class StopOnEvaluation(transformers.StoppingCriteria):
    def __init__(self, tokenizer, possibleEvaluationStops, batchSize):
        self.tokenizer = tokenizer
        self.possibleEvaluationStops = possibleEvaluationStops
        self.stopWords = {"likely", "sure", "impossible"}
        self.stop = [False] * batchSize
        self.count = 0

    def __call__(self, input_ids, score, **kwargs):
        # Decode the generated tokens to text
        self.count += 1

        if self.count < 10:
            return False

        for i, input_ids_single in enumerate(input_ids):
            if self.stop[i]:
                continue

            generated_text_recent = self.tokenizer.decode(input_ids_single[-10:], skip_special_tokens=True)
            if any(word in generated_text_recent for word in self.stopWords):
                self.stop[i] = True

        return all(self.stop)

        # generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # print("------------------------------")
        # print(generated_text)
        # # When running with prompt, sure, impossible, and likely are seen minus 1 as many times in possibleEvaluationStops
        # # When see the threshold, stop generating
        # return any(generated_text.count(word) == threshold for word, threshold in self.possibleEvaluationStops.items())

def gpt_24_proposal(prompt, model, inputAmount, temperature=0.7, max_tokens=1000):
    return model.model_pipeline(
        prompt,
        max_new_tokens = max_tokens,
        temperature = temperature,
        num_return_sequences = 1,
        stopping_criteria = transformers.StoppingCriteriaList([StopOnXInput(tokenizer=model.model_pipeline.tokenizer, inputAmount=inputAmount)])
    )

def llama_propose(model, prompts, batch_size):
    return model.model_pipeline(
        prompts,
        max_new_tokens = 200,
        temperature = 0.7,
        num_return_sequences = 1,
        stopping_criteria = StoppingCriteriaList([model.stopInput]),
        batch_size = batch_size
    )

def llama_values(model, prompts, n_evaluate_sample, batch_size):
    return model.model_pipeline(
        prompts,
        max_new_tokens = 300, # The model sometimes never stops thinking about a sequence so this can't be too high
        temperature = 0.7,
        num_return_sequences = n_evaluate_sample,
        stopping_criteria = StoppingCriteriaList([model.stopLikelySureImp]),
        batch_size = batch_size
    )

def gpt_24_value(prompt, model, lastStep, temperature=0.7, max_tokens=1000, n=1):
    return [gpt_24_value_query(prompt, model, lastStep, temperature, max_tokens)[0]["generated_text"] for _ in range(n)]

def gpt_24_value_query(prompt, model, lastStep, temperature, max_tokens):
    if lastStep:
        stoppingCriteria = StopOnEvaluation(tokenizer=model.model_pipeline.tokenizer, possibleEvaluationStops={"sure": 5, "impossible": 5, "likely": -1}, batchSize=1)
    else:
        stoppingCriteria = StopOnEvaluation(tokenizer=model.model_pipeline.tokenizer, possibleEvaluationStops={"sure": 5, "impossible": 5, "likely": 4}, batchSize=1)

    return model.model_pipeline(
        prompt,
        max_new_tokens = max_tokens,
        temperature = temperature,
        num_return_sequences = 1,
        stopping_criteria = transformers.StoppingCriteriaList([stoppingCriteria])
    )

def gpt(prompt, model, temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    return chatgpt(prompt, pipeline=model.model_pipeline, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def chatgpt(messages, model, temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    outputs = []
    print("---------------------------------")
    while n > 0:
        print("n: ", n)
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(pipeline=model.model_pipeline, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)[0]["generated_text"].split("\n")[12:-1]
        outputs.extend(res)
        # log completion tokens
    return outputs

def completions_with_backoff(model, messages, temperature, max_tokens, n, stop):
    # If no stopping criteria
    if stop is None:
        return model.model_pipeline(
        messages,
        max_new_tokens = max_tokens,
        temperature = temperature,
        num_return_sequences = n
    )

    if stop == ["Input"]:
        return model.model_pipeline(
            messages,
            max_new_tokens = max_tokens,
            temperature = temperature,
            num_return_sequences = n,
            stopping_criteria = transformers.StoppingCriteriaList([StopOnXInput(tokenizer=model.model_pipeline.tokenizer, inputAmount=3)])
        )
    else:
        raise TypeError("Stopping condition not implemented")

def gpt_usage(backend="llama"):
    return {"completion_tokens": 0, "prompt_tokens": 0, "cost": 0}
