# create a file with tasks to execute
DATA_BASE_DIR='./'
#ls ${DATA_BASE_DIR}/*.chunk.* > ${DATA_BASE_DIR}/chunk_files.txt

job_name_base=counterfactual_

#while read input_file; do
#echo "$input_file"
JOB_NAME=${job_name_base}_${input_file##*/}
  
job_mem=60G
job_time=48:00:00   # HH:mm:ss
job_cpus_per_task=8
gpu_selector='gpu:mem24g:1'
# params below not changed very often
# CHANGEME - path to your environment python (when environment is activated write "which python")
python_exec=/home/mitarb/paul/anaconda3/envs/py38/bin/python
JOB_NAME=${JOB_NAME}_$(date +%y-%m-%d-%H-%M-%S)
# CHANGEME - your script for running one single file.
JOB_SCRIPT="PYTHONPATH=. $python_exec src/main.py --experiment_type atomic --experiment_num 0"
#JOB_SCRIPT="PYTHONPATH=. $python_exec scripts/generate/generate_conceptnet_beam_search.py --beam 3 --split test --model_name models/conceptnet-generation/iteration-500-100000/transformer/maxr_5-maxe2_200-rel_language-maxe1_200-devversion_12-trainsize_100/odpt_0.1-rdpt_0.1-nH_12-nL_12-model_transformer-init_pt-afn_gelu-edpt_0.1-pt_gpt-vSize_40616-hSize_768-adpt_0.1/e_1e-08-b1_0.9-l2_0.01-vl2_T-clip_1-lrwarm_0.002-loss_nll-lrsched_warmup_linear-b2_0.999-seed_123-exp_generation/numseq_3-bs_10-es_full-gs_full-smax_128-sample_beam/1e-05_adam_64_4000.pickle"
# DO NOT change anything here!
echo "bash ~/cluster_cmd/run_sbatch.sh \"${JOB_SCRIPT}\" ${JOB_NAME} ${job_mem} ${job_time} ${job_cpus_per_task} ${gpu_selector}"
bash ~/cluster_cmd/run_sbatch.sh "${JOB_SCRIPT}" ${JOB_NAME} ${job_mem} ${job_time} ${job_cpus_per_task} ${gpu_selector}

