import itertools
import numpy as np
from functools import partial
from tot.models import gpt, gpt_24_proposal, gpt_24_value, llama_value, llama_propose
from tot.tasks.game24 import get_current_numbers
from transformers import StoppingCriteriaList, StopStringCriteria
import time
from tot.prompts.game24 import value_prompt

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

    responses = llama_value(model, goodPrompts, n_evaluate_sample, batch_size)
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

def solve_together(args, task, model, cache_values=True, to_print=False):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    data_to_save = {}
    inputs = [task.get_input(i) for i in range(args.task_start_index, args.task_end_index)]
    all_ys = [[''] for _ in range(len(inputs))]
    for step in range(task.steps):
        old_ys = all_ys
        # Augment the steps with the prompt and remove previous bad thoughts
        if (step == task.steps - 1):
            proposal_prompts = [
                [result for result in (
                        task.propose_prompt_wrap(x, y, step == task.steps - 1)
                        for y in ys) if result != "BAD"]
                for x, ys in zip(inputs, all_ys)
            ]
        else:
            proposal_prompts = [[task.propose_prompt_wrap(x, y, step == task.steps - 1) for y in ys] for x, ys in zip(inputs, all_ys)]
        onelist_proposal_prompts = list(itertools.chain(*proposal_prompts))

        # Query the model for the proposals
        start_time = time.time()
        llama_proposals = llama_propose(model, onelist_proposal_prompts, args.batch_size_generate)
        end_time = time.time()

        # Extract the proposals and group them by task
        if (step == task.steps - 1):
            formatted_proposals_nottaskgrouped_prependedsteps = [["\n".join(proposal[0]["generated_text"].split("\n")[33:-1]) + "\n"] for proposal in llama_proposals]
        else:
            formatted_proposals_nottaskgrouped = [proposal[0]["generated_text"].split("\n")[12:-1] for proposal in llama_proposals]
            formatted_proposals_nottaskgrouped_prependedsteps = [[psstep + fsstep + '\n' for fsstep in fssteps] for psstep, fssteps
                                                                in zip(list(itertools.chain(*all_ys)), formatted_proposals_nottaskgrouped)]
        len_formatted_proposals_nottaskgrouped_prependedsteps = [len(sublist) for sublist in proposal_prompts]
        formatted_proposals = [sum(formatted_proposals_nottaskgrouped_prependedsteps[i:j], [])
                                for i, j in zip([0] + list(itertools.accumulate(len_formatted_proposals_nottaskgrouped_prependedsteps))[:-1], itertools.accumulate(len_formatted_proposals_nottaskgrouped_prependedsteps))]

        # Prune proposals if leftover amount is wrong
        if args.prune_bad_proposals and step != task.steps - 1:
            expected_number_left = 3 - step
            formatted_proposals = [[proposal for proposal in proposals if len(get_current_numbers(proposal)) != expected_number_left] for proposals in formatted_proposals]

        if to_print:
            print(f"Execution time: {end_time - start_time:.4f} seconds")
            print(formatted_proposals)

        # Evaluate the proposals
        if cache_values and step != task.steps - 1:
            all_current_numbers = [[get_current_numbers(proposal) for proposal in proposals] for proposals in formatted_proposals]
            all_current_numbers_uniquebytask = [list(set(current_numbers_task)) for current_numbers_task in all_current_numbers]
            values_prompt = [[value_prompt.format(input=current_numbers) for current_numbers in current_numbers_task] for current_numbers_task in all_current_numbers_uniquebytask]
            onelist_values_prompt = list(itertools.chain(*values_prompt))

            # Query the model for an evaluation of the proposals
            start_time = time.time()
            llama_values = llama_value(model, onelist_values_prompt, args.n_evaluate_sample, args.batch_size_evaluate)
            end_time = time.time()

            onelist_formatted_values = [task.value_outputs_unwrap_nox(step, [resultPrompt["generated_text"] for resultPrompt in valuesForStep]) for valuesForStep, step in zip(llama_values, list(itertools.chain(*formatted_proposals)))]

            length_formatted_proposals = [len(sublist) for sublist in values_prompt]
            formatted_values_unique = [onelist_formatted_values[i:j] for i, j in zip([0] + list(itertools.accumulate(length_formatted_proposals))[:-1], itertools.accumulate(length_formatted_proposals))]

            all_current_numbers_to_value = [dict(zip(current_numbers_task, formatted_values_task)) for current_numbers_task, formatted_values_task in zip(all_current_numbers_uniquebytask, formatted_values_unique)]
            formatted_values = [[all_current_numbers_task_to_value[current_numbers] for current_numbers in current_numbers_task] for current_numbers_task, all_current_numbers_task_to_value in zip(all_current_numbers, all_current_numbers_to_value)]
        else:
            values_prompt = [[task.value_prompt_wrap(input, proposal, step == task.steps - 1) for proposal in proposals] for input, proposals in zip(inputs, formatted_proposals)]
            onelist_values_prompt = list(itertools.chain(*values_prompt))

            start_time = time.time()
            llama_values = llama_value(model, onelist_values_prompt, args.n_evaluate_sample, args.batch_size_evaluate)
            end_time = time.time()

            onelist_formatted_values = [task.value_outputs_unwrap_nox(step, [resultPrompt["generated_text"] for resultPrompt in valuesForStep]) for valuesForStep, step in zip(llama_values, list(itertools.chain(*formatted_proposals)))]

            length_formatted_proposals = [len(sublist) for sublist in formatted_proposals]
            formatted_values = [onelist_formatted_values[i:j] for i, j in zip([0] + list(itertools.accumulate(length_formatted_proposals))[:-1], itertools.accumulate(length_formatted_proposals))]
        if to_print:
            print(f"Execution time: {end_time - start_time:.4f} seconds")
            print(formatted_values)

        # Select the best proposals
        mult_ids = [list(range(len(new_ys))) for new_ys in formatted_proposals]
        mult_select_ids = [sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample] for ids, values in zip(mult_ids, formatted_values)]
        all_ys = [[new_ys[select_id] for select_id in select_ids] for new_ys, select_ids in zip(formatted_proposals, mult_select_ids)]
        print(all_ys)

        # Save data to output to log file later
        for i, x, ys, new_ys, values, select_new_ys in zip(range(args.task_start_index, args.task_end_index), inputs, old_ys, formatted_proposals, formatted_values, all_ys):
            task_id = "task" + str(i)
            if step == 0:
                data_to_save[task_id] = {"steps": [{'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys}]}
            else:
                data_to_save[task_id]["steps"].append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})

    return all_ys, data_to_save


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