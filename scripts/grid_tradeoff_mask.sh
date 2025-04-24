#!/bin/bash
#SBATCH --array=1-9
#SBATCH -p 3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --exclude=gpu2262,gpu2112
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=grid_tradeoff_mask.%j.%A.%a.out

# Load modules
module load anaconda/2023.09-0-7nso27y
module load cuda/12.1.1-ebglvvq
source .venv/bin/activate

# Set up experiment variables
id=$SLURM_ARRAY_TASK_ID

ab_levels=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
ab_level_strs=("0p0" "0p1" "0p2" "0p3" "0p4" "0p5" "0p6" "0p7" "0p8")

ab_level_idx=$(( ((id - 1) / 1) % 9 ))
ab_level=${ab_levels[$ab_level_idx]}
ab_level_str=${ab_level_strs[$ab_level_idx]}

data_name="grid_iwl_icl_unrotated_interleaved_seed3"
model_name="${data_name}_run0.pt"
name="grid_tradeoff_mask_ab_level${ab_level_str}"

# Run
python main.py \
--seed 3 \
--n_runs 1 \
--use_cuda \
--out_file ../results/$name \
--task grid \
--load_data_from ../data/$data_name \
--load_vocab ../vocabs/vocab_$data_name.json \
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
--pt_rot_cond unrotated \
--pt_cond interleaved \
--n_epochs 0 \
--n_blocks 4 \
--n_steps_per_block 2000 \
--finetune_lr 0.0001 \
--finetune_bs 5 \
--finetune_test_every 200 \
--test_one_rot_cond \
--congruent_only \
--ablation mask \
--ablation_level $ab_level \
--test_ablation_level_min 0.0 \
--test_ablation_level_max 1.0 \
--test_ablation_level_num 11 \
--model_name llama2 \
--n_layers 12 \
--n_heads 8 \
--d_model 64 \
--d_ff 128 \
--dropout_p 0.1 \
--load_model_from ../checkpoints/$model_name \
--model_path ../checkpoints/$name
