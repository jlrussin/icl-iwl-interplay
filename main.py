import os
import json
import argparse
import random
import torch
import torch.nn as nn
import numpy as np 

from data import generate_datasets, get_pretrain_loaders, get_finetune_loaders
from prompts import seps
from models import get_model
from pretrain import pretrain
from finetune import finetune

parser = argparse.ArgumentParser()
# Setup
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed')
parser.add_argument('--n_runs', type=int, default=1,
                    help='Number of runs')
parser.add_argument('--use_cuda', action='store_true',
                    help='Use GPU, if available') 
parser.add_argument('--out_file', default='../results/results',
                    help='Path to write results to')
# Dataset
parser.add_argument('--task', default='grid', choices=['grid', 'category'],
                    help='Type of task')
parser.add_argument('--save_data_to', default=None,
                    help='If generating data, will use save to this path')
parser.add_argument('--load_data_from', default=None,
                    help='Path to load dataset from. If None, generates it')
parser.add_argument('--save_vocab', default=None,
                    help='Path to vocab file to save to')
parser.add_argument('--load_vocab', default=None,
                    help='Path to vocab file to load from')
parser.add_argument('--n_train', type=int, default=12000,
                    help='Number of episodes in meta-train set')
parser.add_argument('--n_val', type=int, default=100,
                    help='Number of episodes in meta-test set')
parser.add_argument('--n_test', type=int, default=10,
                    help='Number of episodes in each finetuning set')
parser.add_argument('--n_symbols', type=int, default=5,
                    help='Number of symbols in alphabet')
parser.add_argument('--max_len', type=int, default=200,
                    help='Maximum length in the dataset (for model pos. emb.)')
parser.add_argument('--lm_task', default='masked', 
                    choices=['masked', 'causal'], 
                    help='Train with causal or masked language modeling task')
# Grid task-specific parameters
parser.add_argument('--real_words', action='store_true',
                    help='Whether to use real words for colors and animals')
parser.add_argument('--n_positions', type=int, default=1,
                    help='Number of symbols per item in complex task')
parser.add_argument('--p_flip_xy', type=float, default=0.5,
                    help='Probability of flipping (x,y) order in targets')
parser.add_argument('--strict_train', action='store_true',
                    help='Ensure no repeats in training set')
parser.add_argument('--strict_test', action='store_true',
                    help='Ensure no training samples in test set')
parser.add_argument('--generate_all', action='store_true',
                    help='Generate all examples and sample without replacement')
# Category-learning task-specific parameters
parser.add_argument('--n_task_dims', type=int, default=200,
                    help="Number of dimensions in task space for category task")
parser.add_argument('--n_rel_dims', type=int, default=1, choices=[1,2],
                    help="N relevant dims per task: 2 or 1")
parser.add_argument('--n_labels', type=int, default=2,
                    help="Number of total labels in category task")
parser.add_argument('--n_labels_per_task', type=int, default=2, choices=[2,4],
                    help="Number of labels per task in category task")
parser.add_argument('--n_distractors', type=int, default=0,
                    help="Number of distractor dimensions in category task")
parser.add_argument('--n_in_context', type=int, default=32,
                    help="Number of in-context training items in category task")
parser.add_argument('--format', type=str, default='space',
                    choices=['space', 'comma', 'paren', 'paren_comma'],
                    help='Format/template for category task')
# Prompt formatting
parser.add_argument('--preface_id', type=int, default=0,
                    help='ID of preface to use from prompts.py')
parser.add_argument('--template_id', type=int, default=0,
                    help='ID of template to use from prompts.py')
parser.add_argument('--sep_id', type=int, default=0,
                    help='ID of <sep> string to use from prompts.py')
parser.add_argument('--test_preface_id', type=int, default=0,
                    help='ID of test preface to use from prompts.py')
# Metalearning
parser.add_argument('--pt_rot_cond', default='unrotated', 
                    choices=['unrotated', 'rotated'],
                    help='Rotation condition for all episodes in pretraining')
parser.add_argument('--pt_cond', default='blocked', 
                    choices=['aligned', 'misaligned', 'blocked', 'interleaved'],
                    help='Choice of train samples in each pretraining episode')
parser.add_argument('--n_epochs', type=int, default=500,
                    help='Number of epochs') 
parser.add_argument('--print_every', type=int, default=500,
                    help='Number of steps before printing average loss')
parser.add_argument('--test_every', type=int, default=1000,
                    help='Number of steps before testing on held-out episodes') 
parser.add_argument('--pretrain_bs', type=int, default=256,
                    help='Batch size during pretraining')
parser.add_argument('--pretrain_lr', type=float, default=0.001,
                    help='Learning rate for pretraining')
parser.add_argument('--stop_early', action='store_true',
                    help="Use early stopping (based on validation accuracy)")
parser.add_argument('--patience', type=int, default=200,
                    help='Number of tests without improvement before stopping')
# Task-specific finetuning
parser.add_argument('--n_blocks', type=int, default=4,
                    help='Number of blocks of for finetuning (cycle: A, B)') 
parser.add_argument('--n_steps_per_block', type=int, default=1000,
                    help='Number of steps per block of finetuning') 
parser.add_argument('--finetune_lr', type=float, default=0.0001,
                    help='Learning rate for finetuning')
parser.add_argument('--finetune_bs', type=int, default=5,
                    help='Batch size during pretraining')
parser.add_argument('--finetune_test_every', type=int, default=20,
                    help='Number of steps of finetuning before testing')
parser.add_argument('--test_one_rot_cond', action='store_true',
                    help='Only finetune on rotation condition used in training')
parser.add_argument('--congruent_only', action='store_true',
                    help='Only use congruent curricula for finetuning')
# Tradeoff experiments
parser.add_argument('--ablation', default=None,
                    choices=['mask', 'bernoulli', 'gaussian'],
                    help='Type of ablation to apply during finetuning')
parser.add_argument('--ablation_level', type=float, default=0.1,
                    help='Level of ablation to apply during finetuning')
parser.add_argument('--test_ablation_level_min', type=float, default=0.0,
                    help='Minimum ablation level to test')
parser.add_argument('--test_ablation_level_max', type=float, default=1.0,
                    help='Minimum ablation level to test')
parser.add_argument('--test_ablation_level_num', type=int, default=11,
                    help='Number of levels of ablation to test')
# Model
parser.add_argument('--model_name', default='llama2',
                    help='Type of model to use')
parser.add_argument('--llama2_size', default=None,
                    choices=['7b', '7b-chat', 
                             '13b', '13b-chat', 
                             '70b', '70b-chat'], 
                    help='Llama2 model size. None = train from scratch')
parser.add_argument('--n_layers', type=int, default=12,
                    help='Number of layers')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Number of heads')
parser.add_argument('--d_model', type=int, default=64,
                    help='Dimension of internal representations')
parser.add_argument('--d_ff', type=int, default=128,
                    help='Dimension of feedforward layers')
parser.add_argument('--dropout_p', type=float, default=0.1,
                    help='Dropout probability')
parser.add_argument('--model_path', default=None,
                    help="Path to save weights to")
parser.add_argument('--model_dtype', default='fp32', 
                    choices=['fp32', 'fp16'],
                    help="Data type for model weights (fp32 or fp16)")
parser.add_argument('--load_model_from', default=None,
                    help="Path to load model weights from")
parser.add_argument('--cache_dir', default=None,
                    help="Path to Llama 2 pretrained weights")
parser.add_argument('--use_quant', action='store_true',
                    help='Use bitsandbytes quantization')
# Generation
parser.add_argument('--test_gen', action='store_true',
                    help='Test with generate()')
parser.add_argument('--num_beams', type=int, default=5,
                    help='Number of beams to use during generation')

def main(args):

    # CUDA
    if args.use_cuda:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
    else:
        use_cuda = False
        device = "cpu"
    args.device = device
    print("Using CUDA: ", use_cuda, flush=True)

    # Random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Generate dataset
    if args.load_data_from is None:
        assert args.save_data_to is not None
        # If save_data_to already exists, delete all contents
        if os.path.isdir(args.save_data_to):
            print(f"Directory {args.save_data_to} already exists!", flush=True)
            print(f"Deleting contents of {args.save_data_to}", flush=True)
            os.system(f'rm -r {args.save_data_to}')
        # Create fresh directory
        os.mkdir(args.save_data_to)
        generate_datasets(args) # from scratch

    # Vocab
    if args.load_vocab is None:
        args.load_vocab = args.save_vocab
    print(f"Loading vocab from {args.load_vocab}", flush=True)
    with open(args.load_vocab, 'r') as f:
        args.vocab = json.load(f)
    print(f"Vocab size: {len(args.vocab)}", flush=True)
    args.d_vocab = len(args.vocab)
    sep_token = seps[args.sep_id].strip()
    args.sep_idx = args.vocab[sep_token] # vocab index for separator token

    # Loop over runs
    for run_i in range(args.n_runs):
        # Get pretrain data
        pretrain_data = get_pretrain_loaders(args)
        
        # Initialize model
        model = get_model(args)
        if args.load_model_from is not None:
            model.load_state_dict(torch.load(args.load_model_from))
        if args.model_name in ['lstm', 'gru']:
            model.to(args.device)
        print(model, flush=True)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {n_params} learnable parameters", flush=True)

        # Set token embeddings for <mask>
        tokenizer = pretrain_data[0].dataset.tokenizer
        if tokenizer is not None:
            if args.lm_task == 'masked':
                assert '<mask>' in tokenizer.additional_special_tokens
                print("Before resizing embeddings:", model)
                print("Device_map:", model.model.hf_device_map)
                model.model.resize_token_embeddings(len(tokenizer),
                                                    pad_to_multiple_of=128)
                print("After resizing embeddings:", model)
                print("Device_map:", model.model.hf_device_map)

        # Loss function
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        loss_fn.to(args.device)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr)

        # Pretrain
        print(f"Starting pretraining run {run_i}", flush=True)
        pretrain(run_i, pretrain_data, model, loss_fn, optimizer, args)
        
        # Finetune (all conditions)
        if args.n_blocks > 0 and args.n_steps_per_block > 0:
            rotated_loaders, unrotated_loaders = get_finetune_loaders(args)
            for ft_cond_step in ['blocked', 'interleaved']:
                if args.task == 'grid':
                    ft_cond_ctxs = ['aligned', 'misaligned', 
                                    'blocked', 'interleaved']
                elif args.task == 'category':
                    ft_cond_ctxs = ['blocked', 'interleaved']
                else:
                    raise ValueError("Task not recognized")
                for ft_cond_ctx in ft_cond_ctxs:
                    if args.congruent_only:
                        if ft_cond_step != ft_cond_ctx:
                            print(f"Skipping {ft_cond_step}, {ft_cond_ctx}",
                                  flush=True)
                            continue
                    
                    for rot_cond in ['unrotated', 'rotated']:
                        if args.test_one_rot_cond:
                            if rot_cond != args.pt_rot_cond:
                                print(f"Skipping {rot_cond}", flush=True)
                                continue
                        if rot_cond == 'unrotated':
                            loader = unrotated_loaders[ft_cond_ctx]
                        elif rot_cond == 'rotated':
                            loader = rotated_loaders[ft_cond_ctx]
                        msg = f"Finetuning: {rot_cond}, {ft_cond_step}, "
                        msg += f"{ft_cond_ctx}, run {run_i}"
                        print(msg, flush=True)
                        finetune(run_i, ft_cond_ctx, ft_cond_step, rot_cond, 
                                 loader, loss_fn, args)
        else:
            print("Skipped finetuning!")

if __name__ == '__main__':
    args = parser.parse_args()
    print(args, flush=True)
    main(args)