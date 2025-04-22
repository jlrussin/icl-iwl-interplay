import torch
import pickle
import numpy as np

from itertools import cycle
from models import get_model
from utils import run_forward
from test import evaluate
from ablate import ablate

def finetune(run_i, ft_cond_ctx, ft_cond_step, rot_cond, 
             loaders, loss_fn, args):
    
    # Initialize data structures for storing results
    loss_data = [] # records losses, one per episode
    block_name_data = [] # list of block names, one per episode
    acc_data = [] # accuracy data, one per episode
    init_test_data = [] # initial test data, one per episode
    ab_acc_data = [] # ablation accuracy data, one per episode
    ab_init_test_data = [] # initial test data for ablation, one per episode

    # Loop through episodes
    for episode_i, episode_loaders in enumerate(loaders):
        # Data
        train_all_loader = episode_loaders['train_all'] # all train data
        test_all_loader = episode_loaders['test_all'] # all test data
        if ft_cond_step == 'blocked':
            train_loaders = []
            train_names = []
            for k in episode_loaders.keys():
                if 'train' in k and k != 'train_all':
                    train_loaders.append(episode_loaders[k])
                    train_names.append(k)
            block_cycler = cycle(train_loaders)
            name_cycler = cycle(train_names)
        elif ft_cond_step == 'interleaved':
            # Repeat interleaved blocks
            block_cycler = cycle([train_all_loader]) 
            name_cycler = cycle(['train_all'])

        # Load model
        model = get_model(args)
        if args.load_model_from is not None:
            model.load_state_dict(torch.load(args.load_model_from))
        if args.model_name in ['gru', 'lstm']:
            model.to(args.device)
        model.train()

        # Add special token embedding for <pad> and <mask>
        tokenizer = episode_loaders['train_all'].dataset.tokenizer
        if tokenizer is not None:
            assert tokenizer.pad_token == '<pad>'
            if args.lm_task == 'masked':
                assert '<mask>' in tokenizer.additional_special_tokens
            model.model.resize_token_embeddings(len(tokenizer),
                                                pad_to_multiple_of=128)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.finetune_lr)

        # Determine ablation levels to test
        if args.ablation is not None:
            ab_levels = np.linspace(args.test_ablation_level_min,
                                    args.test_ablation_level_max,
                                    args.test_ablation_level_num).tolist()
            if 0.0 not in ab_levels:
                ab_levels.append(0.0)
        else:
            ab_levels = [0.0]

        # Initial test before finetuning
        init_test_accs = {} # for recording initial test accuracies
        ab_init_test_accs = {} # for recording initial ablation test accuracies
        for ab_level in ab_levels:
            # Initial test
            train_acc_dict = evaluate(model, train_all_loader, args,
                                        ab_type=args.ablation, 
                                        ab_level=ab_level)
            test_acc_dict = evaluate(model, test_all_loader, args,
                                        ab_type=args.ablation, 
                                        ab_level=ab_level)
            # Record testing results (no ablation)
            if ab_level == 0.0:
                for k,v in train_acc_dict.items():
                    if k not in init_test_accs:
                        init_test_accs[k] = []
                    init_test_accs[k].append(v)
                for k,v in test_acc_dict.items():
                    if k not in init_test_accs:
                        init_test_accs[k] = []
                    init_test_accs[k].append(v)
            # Record ablation results
            if ab_level not in ab_init_test_accs:
                ab_init_test_accs[ab_level] = {}
            for k,v in train_acc_dict.items():
                if k not in ab_init_test_accs[ab_level]:
                    ab_init_test_accs[ab_level][k] = []
                ab_init_test_accs[ab_level][k].append(v)
            for k,v in test_acc_dict.items():
                if k not in ab_init_test_accs[ab_level]:
                    ab_init_test_accs[ab_level][k] = []
                ab_init_test_accs[ab_level][k].append(v)
        
        # Record initial test accuracies
        init_test_data.append(init_test_accs)
        ab_init_test_data.append(ab_init_test_accs)
        
        # Set up for recording loss and accuracy over finetuning episode
        total_steps = 0
        losses = []
        block_names = []
        test_accs = []
        ab_test_accs = []

        # Training loop
        for block_i in range(args.n_blocks):
            step = 0
            block = next(block_cycler)
            block_name = next(name_cycler)
            block_losses = []
            ave_loss = []
            block_accs = {}
            ab_block_accs = {}
            while step < args.n_steps_per_block:
                for batch in block:
                    optimizer.zero_grad()

                    # Data
                    src, tgt, info = batch
                    src = {k: v.to(args.device) for k, v in src.items()}
                    tgt = {k: v.to(args.device) for k, v in tgt.items()}
                    bs, tgt_len = tgt['input_ids'].shape

                    # Finetuning with ablation
                    if args.ablation is not None:
                        src = ablate(args.ablation, args.ablation_level, 
                                     src, model, args)

                    # Model forward pass
                    prd = run_forward(model, src, tgt, args)
                    # prd: [bs, tgt_len, d_vocab]    

                    # Loss
                    tgt_flat = tgt['input_ids'].reshape(-1) # [bs*tgt_len]
                    prd_flat = prd.reshape(-1, prd.shape[2]) 
                    # prd_flat = [bs*tgt_len, d_vocab]
                    loss = loss_fn(prd_flat, tgt_flat) # [bs*tgt_len]
                    mean_loss = torch.mean(loss)
                    
                    # Backward
                    mean_loss.backward()

                    # Take optimizer step
                    optimizer.step()

                    # Record loss
                    block_losses.append(mean_loss.item())
                    ave_loss.append(mean_loss.item())

                    # Test
                    last_step = step >= args.n_steps_per_block
                    if (step % args.finetune_test_every == 0) or last_step:
                        for ab_level in ab_levels:
                            # Evaluate
                            train_acc_dict = evaluate(model, train_all_loader, 
                                                      args,
                                                      ab_type=args.ablation, 
                                                      ab_level=ab_level)
                            test_acc_dict = evaluate(model, test_all_loader, 
                                                     args, 
                                                     ab_type=args.ablation, 
                                                     ab_level=ab_level)
                            # Record testing results (no ablation)
                            if ab_level == 0.0:
                                for k,v in train_acc_dict.items():
                                    if k not in block_accs:
                                        block_accs[k] = []
                                    block_accs[k].append(v)
                                for k,v in test_acc_dict.items():
                                    if k not in block_accs:
                                        block_accs[k] = []
                                    block_accs[k].append(v)
                            # Record ablation results
                            if ab_level not in ab_block_accs:
                                ab_block_accs[ab_level] = {}
                            for k,v in train_acc_dict.items():
                                if k not in ab_block_accs[ab_level]:
                                    ab_block_accs[ab_level][k] = []
                                ab_block_accs[ab_level][k].append(v)
                            for k,v in test_acc_dict.items():
                                if k not in ab_block_accs[ab_level]:
                                    ab_block_accs[ab_level][k] = []
                                ab_block_accs[ab_level][k].append(v)
                        ave_loss = [] # reset
                    step += 1
                    total_steps += 1
            losses.append(block_losses)
            block_names.append(block_name)
            test_accs.append(block_accs)
            ab_test_accs.append(ab_block_accs)
        loss_data.append(losses)
        block_name_data.append(block_names)
        acc_data.append(test_accs)
        ab_acc_data.append(ab_test_accs)
    
    # Save results
    results = {'loss_data': loss_data,
               'block_name_data': block_name_data,
               'acc_data': acc_data,
               'ab_acc_data': ab_acc_data,
               'initial_test': init_test_data,
               'ab_initial_test': ab_init_test_data}
    out_fn = f'{args.out_file}_ft_{rot_cond}_{ft_cond_ctx}_{ft_cond_step}'
    out_fn += f'_run{run_i}.pickle'
    print(f"Saving results to {out_fn}", flush=True)
    with open(out_fn, 'wb') as f:
        pickle.dump(results, f)
    print('Done.', flush=True)

