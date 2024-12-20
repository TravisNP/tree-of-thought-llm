python run.py \
    --backend llama \
    --task game24 \
    --task_start_index 1 \
    --task_end_index 3 \
    --method_generate propose \
    --method_evaluate value \
    --method_select greedy \
    --n_evaluate_sample 3 \
    --n_select_sample 5 \
    ${@}
