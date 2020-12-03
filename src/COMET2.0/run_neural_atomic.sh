# create a file with tasks to execute
DATA_BASE_DIR='./'
#ls ${DATA_BASE_DIR}/*.chunk.* > ${DATA_BASE_DIR}/chunk_files.txt

job_name_base=atomic_generate_2.0_greedy_

#while read input_file; do
#echo "$input_file"
JOB_NAME=${job_name_base}_${input_file##*/}
  
job_mem=20G
job_time=10:00:00   # HH:mm:ss
job_cpus_per_task=8
gpu_selector='gpu:mem24g:1'
# params below not changed very often
# CHANGEME - path to your environment python (when environment is activated write "which python")
python_exec=/home/mitarb/paul/anaconda3/envs/py38/bin/python
JOB_NAME=${JOB_NAME}_$(date +%y-%m-%d-%H-%M-%S)
# CHANGEME - your script for running one single file.
#JOB_SCRIPT="PYTHONPATH=. $python_exec src/main.py --experiment_type conceptnet --experiment_num 0"
#JOB_SCRIPT="PYTHONPATH=. $python_exec scripts/generate/generate_atomic_beam_search.py --beam 3 --split test --model_name pretrained_models/atomic_pretrained_model.pickle"
JOB_SCRIPT="PYTHONPATH=. $python_exec scripts/generate/generate_atomic_beam_search.py --beam 4 --split test --model_name models/atomic-generation/iteration-500-50000/transformer/categories_oEffect#oReact#oWant#xAttr#xEffect#xIntent#xNeed#xReact#xWant-maxe1_17-maxe2_36-maxr_1/model_transformer-nL_12-nH_12-hSize_1024-edpt_0.1-adpt_0.1-rdpt_0.1-odpt_0.1-pt_gpt-afn_gelu-init_pt-vSize_50322/exp_generation-seed_123-l2_0.01-vl2_T-lrsched_warmup_linear-lrwarm_0.002-clip_1-loss_nll-b2_0.999-b1_0.9-e_1e-08/bs_1-smax_40-sample_greedy-numseq_1-gs_full-es_full-categories_oEffect#oReact#oWant#xAttr#xEffect#xIntent#xNeed#xReact#xWant/6.25e-05_adam_12_50000.pickle"
# DO NOT change anything here!
echo "bash ~/cluster_cmd/run_sbatch.sh \"${JOB_SCRIPT}\" ${JOB_NAME} ${job_mem} ${job_time} ${job_cpus_per_task} ${gpu_selector}"
bash ~/cluster_cmd/run_sbatch.sh "${JOB_SCRIPT}" ${JOB_NAME} ${job_mem} ${job_time} ${job_cpus_per_task} ${gpu_selector}

