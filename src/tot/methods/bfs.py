import itertools
import numpy as np
from functools import partial
from tot.models import gpt, gpt_24_proposal, gpt_24_value, llama_values, llama_propose
from tot.tasks.game24 import get_current_numbers
from transformers import StoppingCriteriaList, StopStringCriteria

def get_value(task, x, y, n_evaluate_sample, model, lastStep, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y, lastStep)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    if value_prompt == "BAD":
        value = 0
    else:
        value_outputs = gpt_24_value(value_prompt, model, lastStep, n=n_evaluate_sample)
        value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, model, lastStep, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:
            value = get_value(task, x, y, n_evaluate_sample, model, lastStep, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_values_batch(task, x, ys, n_evaluate_sample, model, batch_size, lastStep, toPrint = False):
    prompts = [task.value_prompt_wrap(x, y, lastStep) for y in ys]
    if lastStep:
        goodPromptsYs = [(prompt, step) for prompt, step in zip(prompts, ys) if prompt != "BAD"]
        goodPrompts, goodys = map(list, zip(*goodPromptsYs))
    else:
        goodPrompts = prompts
        goodys = ys

    responses = llama_values(model, goodPrompts, n_evaluate_sample, batch_size)
    if toPrint:
        print(responses)
    values = [task.value_outputs_unwrap(x, step, [resultPrompt["generated_text"] for resultPrompt in valuesForStep]) for valuesForStep, step in zip(responses, goodys)]

    if lastStep:
        goodValues = []
        indxValues = 0
        for prompt in prompts:
            if prompt == "BAD":
                goodValues.append(0)
            else:
                goodValues.append(values[indxValues])
                indxValues += 1
    else:
        goodValues = values

    return goodValues

def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

def get_proposals_batch(task, x, ys, model, batch_size, lastInput):
    prompts = [task.propose_prompt_wrap(x, y, lastInput) for y in ys]
    if not lastInput:
        responses = llama_propose(model, prompts, batch_size)
        return [f"{y}{step}\n" for response, y in zip(responses, ys) for step in response[0]["generated_text"].split("\n")[12:-1]]

    good_prompts = [prompt for prompt in prompts if prompt != "BAD"]
    good_responses = llama_propose(model, good_prompts, batch_size)
    good_proposals = ["\n".join(proposal[0]["generated_text"].split("\n")[33:-1]) + '\n' for proposal in good_responses]

    proposals = []
    indxGoodProposal = 0
    for i, prompt in enumerate(prompts):
        if prompt == "BAD":
            proposals.append(ys[i] + "1 + 1 = 2 (left: 2)\n")
        else:
            proposals.append(good_proposals[indxGoodProposal])
            indxGoodProposal += 1
    return proposals

def get_proposals(task, x, y, model, lastInput):
    propose_prompt = task.propose_prompt_wrap(x, y)

    # 7 inputs to stop final prompt
    if lastInput:
        if get_current_numbers(y) != '24':
            return [y + "1 + 1 = 2 (left: 2)\n"]
        proposal = "\n".join(gpt_24_proposal(propose_prompt, model, 7)[0]["generated_text"].split("\n")[33:-1]) + '\n'
        return [proposal]

    # 3 inputs to stop regular prompt,
    proposals = gpt_24_proposal(propose_prompt, model, 3)[0]["generated_text"].split("\n")[12:-1]
    return [y + _ + '\n' for _ in proposals]

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]

def solve(args, task, idx, model, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = ['']  # current output candidates
    infos = []
    for step in range(task.steps):
        # generation
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
        elif args.method_generate == 'propose':
            # new_ys = [get_proposals(task, x, y, model, step == task.steps - 1) for y in ys]
            new_ys = get_proposals_batch(task, x, ys, model, 16, step == task.steps - 1)
        # new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))

        print("Proposals: ", new_ys)

        # evaluation
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            # values = get_values(task, x, new_ys, args.n_evaluate_sample, model_pipeline, step == task.steps - 1)
            values = get_values_batch(task, x, new_ys, args.n_evaluate_sample, model, 16, False)

        print("Values: ", values)

        # selection
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        print("Best proposals: ", select_new_ys)

        # log
        if to_print:
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')

        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys

    if to_print:
        print(ys)
    return ys, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}