#!/bin/bash
#SBATCH --array=1-8
#SBATCH -p 3090-gcondo
#SBATCH --gres=gpu:2
#SBATCH --gres-flags=enforce-binding
#SBATCH --exclude=gpu2262,gpu2112
#SBATCH -N 1
#SBATCH --cpus-per-gpu 2
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=grid_llama_70b.%j.%A.%a.out

# Load modules
module load anaconda/2023.09-0-7nso27y
module load cuda/12.1.1-ebglvvq
source .venv/bin/activate

# Set up experiment variables
id=$SLURM_ARRAY_TASK_ID

pt_rot_conds=('unrotated' 'rotated')
pt_conds=('aligned' 'misaligned' 'blocked' 'interleaved')

rot_id=$(( ((id - 1) / 1) % 2 ))
cond_id=$(( ((id - 1) / 2) % 4 ))

pt_rot_cond=${pt_rot_conds[$rot_id]}
pt_cond=${pt_conds[$cond_id]}

name="grid_llama_70b_${pt_rot_cond}_${pt_cond}"

# Run
python main.py \
--seed 0 \
--n_runs 1 \
--use_cuda \
--out_file ../results/$name \
--task grid \
--save_data_to ../data/$name \
--save_vocab ../vocabs/$name.json \
--n_train 12000 \
--n_val 100 \
--n_test 10 \
--n_symbols 5 \
--max_len 200 \
--lm_task causal \
--real_words \
--n_positions 1 \
--p_flip_xy 0.0 \
--generate_all \
--preface_id 7 \
--template_id 0 \
--sep_id 1 \
--test_preface_id 4 \
--pt_rot_cond $pt_rot_cond \
--pt_cond $pt_cond \
--n_epochs 0 \
--pretrain_bs 1 \
--n_blocks 0 \
--n_steps_per_block 0 \
--model_name llama2 \
--llama2_size 70b \
--cache_dir '/gpfs/data/superlab/models/llama2/llama/checkpoints/hf' \
--use_quant \
--num_beams 5 

