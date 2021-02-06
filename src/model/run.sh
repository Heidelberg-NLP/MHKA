# create a file with tasks to execute
DATA_BASE_DIR='./'
#ls ${DATA_BASE_DIR}/*.chunk.* > ${DATA_BASE_DIR}/chunk_files.txt
DATA_DIR='./data/data_without_k'
job_name_base=roberta_without

#while read input_file; do
#echo "$input_file"
JOB_NAME=${job_name_base}_${input_file##*/}
  
job_mem=10G
job_time=24:00:00   # HH:mm:ss
job_cpus_per_task=8
gpu_selector='gpu:mem24g:1'
# params below not changed very often
# CHANGEME - path to your environment python (when environment is activated write "which python")
python_exec=/home/mitarb/paul/anaconda3/envs/py38/bin/python
JOB_NAME=${JOB_NAME}_$(date +%y-%m-%d-%H-%M-%S)
# CHANGEME - your script for running one single file.
#JOB_SCRIPT="PYTHONPATH=. $python_exec ./experiment.py ./conf.txt"
JOB_SCRIPT="PYTHONPATH=. $python_exec run_multiple_choice_know.py \
	--model_type=roberta \
	--task_name=swag \ 
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
# DO NOT change anything here!
echo "bash ~/cluster_cmd/run_sbatch_gpulong.sh \"${JOB_SCRIPT}\" ${JOB_NAME} ${job_mem} ${job_time} ${job_cpus_per_task} ${gpu_selector}"
bash ~/cluster_cmd/run_sbatch.sh "${JOB_SCRIPT}" ${JOB_NAME} ${job_mem} ${job_time} ${job_cpus_per_task} ${gpu_selector}

