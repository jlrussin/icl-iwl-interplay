#!/bin/bash
#SBATCH --array=1-9
#SBATCH -p 3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --exclude=gpu2262,gpu2112
#SBATCH -N 1
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --output=cat_tradeoff_noise.%j.%A.%a.out

# Load modules
module load anaconda/2023.09-0-7nso27y
module load cuda/12.1.1-ebglvvq
source .venv/bin/activate

# Set up experiment variables
id=$SLURM_ARRAY_TASK_ID

ab_levels=(0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
ab_level_strs=("0p0" "1p0" "2p0" "3p0" "4p0" "5p0" "6p0" "7p0" "8p0")

ab_level_idx=$(( ((id - 1) / 1) % 9 ))
ab_level=${ab_levels[$ab_level_idx]}
ab_level_str=${ab_level_strs[$ab_level_idx]}

data_name="cat_iwl_icl_unrotated_interleaved_seed1"
model_name="${data_name}_run0.pt"
name="cat_tradeoff_noise_ab_level${ab_level_str}"

# Run
python main.py \
--seed 1 \
--n_runs 1 \
--use_cuda \
--out_file ../results/$name \
--task category \
--load_data_from ../data/$data_name \
--load_vocab ../vocabs/vocab_$data_name.json \
--n_train 12000 \
--n_val 100 \
--n_test 10 \
--n_symbols 8 \
--max_len 320 \
--lm_task masked \
--n_task_dims 200 \
--n_rel_dims 1 \
--n_labels 2 \
--n_labels_per_task 2 \
--n_distractors 0 \
--n_in_context 32 \
--format 'space' \
--preface_id 0 \
--template_id 0 \
--sep_id 0 \
--test_preface_id 0 \
--pt_rot_cond unrotated \
--pt_cond interleaved \
--n_epochs 0 \
--n_blocks 4 \
--n_steps_per_block 2000 \
--finetune_lr 0.00001 \
--finetune_bs 32 \
--finetune_test_every 20 \
--test_one_rot_cond \
--congruent_only \
--ablation gaussian \
--ablation_level $ab_level \
--test_ablation_level_min 0.0 \
--test_ablation_level_max 8.0 \
--test_ablation_level_num 9 \
--model_name llama2 \
--n_layers 4 \
--n_heads 8 \
--d_model 64 \
--d_ff 128 \
--dropout_p 0.0 \
--load_model_from ../checkpoints/$model_name \
--model_path ../checkpoints/$name
