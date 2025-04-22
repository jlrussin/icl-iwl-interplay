#!/bin/bash
#SBATCH --array=1-80
#SBATCH -p 3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --exclude=gpu2262,gpu2112
#SBATCH -N 1
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --output=grid_iwl_icl.%j.%A.%a.out

# Load modules
module load anaconda/2023.09-0-7nso27y
module load cuda/12.1.1-ebglvvq
source .venv/bin/activate

# Set up experiment variables
id=$SLURM_ARRAY_TASK_ID

rot_conds=(unrotated rotated) # metalearning rotation condition
pt_conds=(aligned misaligned blocked interleaved) # metalearning curriculum
seeds=(1 2 3 4 5 6 7 8 9 10) # random seeds

rot_idx=$(( ((id - 1) / 1) % 2 ))
pt_idx=$(( ((id - 1) / 2) % 4 ))
seed_idx=$(( ((id - 1) / 8) % 10 ))

rot_cond=${rot_conds[$rot_idx]}
pt_cond=${pt_conds[$pt_idx]}
seed=${seeds[$seed_idx]}

name="grid_iwl_icl_${rot_cond}_${pt_cond}_seed${seed}"

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
--pt_rot_cond $rot_cond \
--pt_cond $pt_cond \
--n_epochs 500 \
--print_every 50 \
--test_every 1000 \
--pretrain_bs 256 \
--pretrain_lr 0.001 \
--stop_early \
--patience 200 \
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
