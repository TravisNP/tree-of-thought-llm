python run.py \
    --backend llama3-3_70 \
    --task game24 \
    --task_start_index 18 \
    --task_end_index 20 \
    --method_generate propose \
    --method_evaluate value \
    --method_select greedy \
    --n_evaluate_sample 3 \
    --n_select_sample 5 \
    --batch_size_generate 216 \
    --batch_size_evaluate 40 \
    ${@} > output.log 2>&1
