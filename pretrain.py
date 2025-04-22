import torch
import pickle
import numpy as np

from utils import run_forward
from test import evaluate

class TrainingState:
    """
    For maintaining results and tracking validation for early stopping
    """
    def __init__(self, run_i):
        self.run_i = run_i # run
        self.step = 0 # step of training (total, across all epochs)
        self.losses = {'train': [], 'test': []} # one per print_every steps
        self.ave_loss = {'train': [], 'test': []} # record all, will be averaged
        self.test_accs = {'train': [], 'test': []} # one per test_every steps
        self.best_val = 0.0 # track best validation accuracy for early stopping
        self.n_wo_improve = 0 # n tests without improving (early stopping)

        # Add tracking by episode_id
        self.test_accs_by_episode = {}
    
    def log(self, epoch):
        # Group losses by variables of interest
        ave_train_loss = np.mean(self.ave_loss['train'])
        ave_test_loss = np.mean(self.ave_loss['test'])
        self.losses['train'].append(ave_train_loss)
        self.losses['test'].append(ave_test_loss)
        
        print(f'Run: {self.run_i}, Epoch: {epoch}, Step: {self.step},',
              f'Loss (train): {ave_train_loss},', 
              f'Loss (test): {ave_test_loss}', flush=True)

        # Reset lists used for averaging
        self.reset_running_aves()

    def reset_running_aves(self):
        self.ave_loss = {'train': [], 'test': []} # reset

def pretrain(run_i, data, model, loss_fn, optimizer, args):
    # Training state for tracking results, validation
    state = TrainingState(run_i)

    # Data
    train_loader, test_loader = data

    # Model
    model.train()

    # Training loop
    done = False
    for epoch in range(args.n_epochs):
        print(f"Starting Epoch {epoch}", flush=True)
        for batch_i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            # Data
            src, tgt, info = batch
            src = {k: v.to(args.device) for k, v in src.items()}
            tgt = {k: v.to(args.device) for k, v in tgt.items()}
            bs, tgt_len = tgt['input_ids'].shape

            # Model
            prd = run_forward(model, src, tgt, args)
            # prd: [bs, tgt_len, d_vocab] 

            # Loss
            tgt_flat = tgt['input_ids'].reshape(-1) # [bs*tgt_len]
            prd_flat = prd.reshape(-1, prd.shape[2]) # [bs*tgt_len, d_vocab]
            loss = loss_fn(prd_flat, tgt_flat) # [bs*tgt_len]
            mean_loss = torch.mean(loss)
            
            # Backward
            mean_loss.backward()
                    
            # Take optimizer step
            optimizer.step()

            # Sort and record losses
            all_loss = loss.reshape(bs, tgt_len).detach().cpu().numpy()
            for sample_i, l in enumerate(all_loss):
                split = info['split'][sample_i]
                state.ave_loss[split].append(np.mean(l)) # ave masked positions

            # Log
            last = batch_i == (len(train_loader) - 1)
            if (state.step > 0 and state.step % args.print_every == 0) or last:
                state.log(epoch)

            # Checkpoint
            if (state.step % args.test_every == 0) or last:
                done = checkpoint(run_i, epoch, state, test_loader, model, args)
                if done:
                    break
            state.step += 1
        if done:
            break
    # Final checkpoint (this will also test when n_epochs=0)
    epoch = args.n_epochs
    checkpoint(run_i, epoch, state, test_loader, model, args)


def checkpoint(run_i, epoch, state, test_loader, model, args):
    # Test on held-out episodes
    print("Testing on test set...", flush=True)
    acc_dict = evaluate(model, test_loader, args)
    train_acc, test_acc = acc_dict['train'], acc_dict['test']
    state.test_accs['train'].append(train_acc)
    state.test_accs['test'].append(test_acc)
    for k,v in acc_dict.items():
        if 'episode' in k:
            if k not in state.test_accs_by_episode:
                state.test_accs_by_episode[k] = []
            state.test_accs_by_episode[k].append(v)
    print(f"Accuracy: Train = {train_acc}, Test = {test_acc}", flush=True)

    # Save results
    results = {'losses': state.losses,
               'test_accs': state.test_accs,
               'test_accs_by_episode': state.test_accs_by_episode}
    out_fn = args.out_file + '_pt' + f'_run{run_i}' + '.pickle'
    print(f"Saving results to {out_fn}", flush=True)
    with open(out_fn, 'wb') as f:
        pickle.dump(results, f)
    print('Done.')
    
    # Checkpoint model
    model.train()
    if test_acc > state.best_val or state.step == 0: # always save first step
        state.n_wo_improve = 0
        state.best_val = test_acc
        if args.model_path is not None:
            model_path = args.model_path + f'_run{run_i}' + '.pt'
            print(f"Saving model to {model_path}", flush=True)
            torch.save(model.state_dict(), model_path)
            print('Done.', flush=True)
            args.load_model_from = model_path
    else:
        state.n_wo_improve += 1
        if args.stop_early and state.n_wo_improve >= args.patience:
            print("Exceeded patience, stopping early", flush=True)
            return True
    return False
    
