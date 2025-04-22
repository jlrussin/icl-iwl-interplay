import torch
import numpy as np

from utils import run_forward
from ablate import ablate
    
def evaluate(model, loader, args, ab_type=None, ab_level=None):
    model.eval()

    acc = {} # dictionary to store accuracies

    # Forward pass to get predictions
    with torch.no_grad():
        for batch_i, batch in enumerate(loader):
            # Data
            src, tgt, info = batch
            src = {k: v.to(args.device) for k, v in src.items()}
            tgt = {k: v.to(args.device) for k, v in tgt.items()}

            # Retention test
            if ab_type is not None:
                assert ab_level is not None
                src = ablate(ab_type, ab_level, src, model, args)

            # Model
            prd = run_forward(model, src, tgt, args)
            # prd: [bs, tgt_len, d_vocab]  

            # Calculate correct
            prd = torch.argmax(prd, dim=2) # [bs, tgt_len]
            tok_correct = (prd == tgt['input_ids']) # [bs, tgt_len]
            seq_correct = torch.all(tok_correct, dim=1) # [bs]
            correct = seq_correct.cpu().numpy() # [bs]

            for i, c in enumerate(correct):
                # Track by split (train vs. test)
                split = info['split'][i]
                if split not in acc:
                    acc[split] = []
                acc[split].append(c)

                # Track by episode_id
                episode_id = info['episode_id'][i]
                episode_str = f"episode_{episode_id}"
                if episode_str not in acc:
                    acc[episode_str] = {}
                if split not in acc[episode_str]:
                    acc[episode_str][split] = []
                acc[episode_str][split].append(c)

                # Track task dimensions used in each episode
                if 'task_dims' in info:
                    task_dims = info['task_dims'][i]
                    dims_tuple = tuple(task_dims)
                    if 'task_dims' not in acc[episode_str]:
                        acc[episode_str]['task_dims'] = dims_tuple
                    else:
                        # All samples in episode should use the same task_dims
                        assert acc[episode_str]['task_dims'] == dims_tuple

    model.train()

    # Average to get accuracy
    accuracies = {}
    for k,v in acc.items():
        if isinstance(v, list): # split
            if len(v) > 0: # avoid warnings
                accuracies[k] = np.mean(v)
            else:
                accuracies[k] = np.nan
        elif isinstance(v, dict): # episode_id or task_dims
            accuracies[k] = {}
            for k2,v2 in v.items():
                if isinstance(v2, list):
                    if len(v2) > 0:
                        accuracies[k][k2] = np.mean(v2)
                    else:
                        accuracies[k][k2] = np.nan
                else: # task_dims
                    accuracies[k][k2] = v2

    return accuracies