```bash
# create a file with tasks to execute
DATA_BASE_DIR='./'
DATA_DIR='./data/data_'
# params below not changed very often
# CHANGEME - path to your environment python (when environment is activated write "which python")
python_exec=which python
JOB_SCRIPT="PYTHONPATH=. $python_exec run_multiple_choice_know.py \
	--model_type=roberta \
	--task_name=anli \ 
	--model_name_or_path=roberta-large \
        --save_steps=10000 \
        --do_lower_case \
        --seed=42 \
	--do_train \
	--do_eval \
        --do_test \
	--data_dir $DATA_DIR \
	--learning_rate=5e-6 \
	--num_train_epochs=5 \
	--max_seq_length=80 \
        --evaluate_during_training \
        --eval_all_checkpoints \
	--output_dir output_roberta_without_seed_42/ \
	--per_gpu_eval_batch_size=8 \
	--per_gpu_train_batch_size=8 \
	--gradient_accumulation_steps 1 \
        --local_rank=-1 \
	--overwrite_output"
```

