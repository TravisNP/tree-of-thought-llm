python run.py \
    --backend llama3-3_70 \
    --task game24 \
    --task_start_index 1 \
    --task_end_index 100 \
    --method_generate propose \
    --method_evaluate value \
    --method_select greedy \
    --n_evaluate_sample 3 \
    --n_select_sample 5 \
    --batch_size_generate 256 \
    --batch_size_evaluate 48 \
    ${@}
