#!/bin/bash
#SBATCH --array=1-10
#SBATCH -p 3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --exclude=gpu2262,gpu2112
#SBATCH -N 1
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --output=grid_iwl_only.%j.%A.%a.out

# Load modules
module load anaconda/2023.09-0-7nso27y
module load cuda/12.1.1-ebglvvq
source .venv/bin/activate

# Set up experiment variables
id=$SLURM_ARRAY_TASK_ID

seeds=(1 2 3 4 5 6 7 8 9 10) # random seeds
seed_idx=$(( ((id - 1) / 1) % 10 ))
seed=${seeds[$seed_idx]}

name="grid_iwl_only_seed${seed}"

# Run
python main.py \
--seed $seed \
--n_runs 1 \
--use_cuda \
--out_file ../results/$name \
--task grid \
--save_data_to ../data/$name \
--save_vocab ../vocabs/vocab_$name.json \
--n_train 12000 \
--n_val 100 \
--n_test 10 \
--n_symbols 5 \
--max_len 200 \
--lm_task masked \
--real_words \
--n_positions 1 \
--p_flip_xy 0.5 \
--strict_train \
--strict_test \
--generate_all \
--preface_id 0 \
--template_id 0 \
--sep_id 0 \
--test_preface_id 0 \
--n_epochs 0 \
--n_blocks 4 \
--n_steps_per_block 1000 \
--finetune_lr 0.0001 \
--finetune_bs 5 \
--finetune_test_every 20 \
--model_name llama2 \
--n_layers 12 \
--n_heads 8 \
--d_model 64 \
--d_ff 128 \
--dropout_p 0.1 \
--model_path ../checkpoints/$name
