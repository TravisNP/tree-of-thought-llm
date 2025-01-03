import os
import json
import argparse
import time

from tot.tasks import get_task
from tot.methods.bfs import solve, naive_solve, solve_together
from tot.models import gpt_usage, Model
from transformers import pipeline, AutoTokenizer
import torch


def run(args):
    task = get_task(args.task)
    logs, cnt_avg, cnt_any = [], 0, 0

    # Set model
    if args.backend == "llama3-1_8":
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    elif args.backend == 'llama3-3_70':
        model_id = "meta-llama/Llama-3.3-70B-Instruct"
    else:
        raise TypeError("Model must be llama")

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model_pipeline = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    model = Model(model_pipeline, model_id)

    file = f'./logs/{args.task}/{args.backend}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}_propbatch{args.batch_size_generate}_valbatch{args.batch_size_evaluate}_{model_id}.json'
    start_time = time.time()
    all_ys, data_to_save = solve_together(args, task, model, True, True)
    end_time = time.time()
    for i, ys in zip(range(args.task_start_index, args.task_end_index), all_ys):
        correctness = [task.test_output(i, y) for y in ys]
        data_to_save["task" + str(i)].update({'idx': i, 'ys': ys, 'infos': correctness})
    data_to_save["time"] = f"{end_time - start_time:.2f}"

    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'w') as f:
        json.dump(data_to_save, f, indent=4)
    return

    if args.naive_run:
        file = f'./logs/{args.task}/{args.backend}_{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    else:
        file = f'./logs/{args.task}/{args.backend}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}_{model_id}.json'
    os.makedirs(os.path.dirname(file), exist_ok=True)

    for i in range(args.task_start_index, args.task_end_index):
        # solve
        if args.naive_run:
            ys, info = naive_solve(args, task, i)
        else:
            start_time = time.time()
            ys, info = solve(args, task, i, model)
            end_time = time.time()

        # log
        infos = [task.test_output(i, y) for y in ys]
        info.update({'idx': i, 'ys': ys, 'infos': infos, 'usage_so_far': end_time - start_time})
        logs.append(info)
        with open(file, 'w') as f:
            json.dump(logs, f, indent=4)

        # log main metric
        accs = [info['r'] for info in infos]
        cnt_avg += sum(accs) / len(accs)
        cnt_any += any(accs)
        print(i, 'sum(accs)', sum(accs), 'cnt_avg', cnt_avg, 'cnt_any', cnt_any, '\n')

    n = args.task_end_index - args.task_start_index
    print(cnt_avg / n, cnt_any / n)


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, choices=['llama3-1_8', 'llama3-3_70'], default='llama3-1_8')
    args.add_argument('--temperature', type=float, default=0.7)

    args.add_argument('--task', type=str, required=True, choices=['game24', 'text', 'crosswords'])
    args.add_argument('--task_start_index', type=int, default=900)
    args.add_argument('--task_end_index', type=int, default=1000)

    args.add_argument('--naive_run', action='store_true')
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])  # only used when method_generate = sample, or naive_run

    args.add_argument('--method_generate', type=str, choices=['sample', 'propose'])
    args.add_argument('--method_evaluate', type=str, choices=['value', 'vote'])
    args.add_argument('--method_select', type=str, choices=['sample', 'greedy'], default='greedy')
    args.add_argument('--n_generate_sample', type=int, default=1)  # only thing needed if naive_run
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--n_select_sample', type=int, default=1)

    args.add_argument('--batch_size_generate', type=int, default=1)
    args.add_argument('--batch_size_evaluate', type=int, default=1)

    args.add_argument('--prune_bad_proposals', type=bool, default=True)
    args.add_argument('--quick_last_valuation', type=bool, default=True)

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)