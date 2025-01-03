i=400
while [ $i -le 1350 ]; do
    start_index=$i
    end_index=$((i + 50))

    python run.py \
        --backend llama3-3_70 \
        --task game24 \
        --task_start_index $start_index \
        --task_end_index $end_index \
        --method_generate propose \
        --method_evaluate value \
        --method_select greedy \
        --n_evaluate_sample 3 \
        --n_select_sample 5 \
        --batch_size_generate 60 \
        --batch_size_evaluate 25 \
        ${@} > "output_${start_index}_${end_index}_llama3.3-70B.log" 2>&1

    i=$((i + 50))
done
