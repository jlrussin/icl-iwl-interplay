import torch

def run_forward(model, src, tgt, args):
    tgt_len = tgt['input_ids'].shape[1]
    if args.lm_task == 'masked':
        prd = model(src) # [bs, src_len, d_vocab]
        bs, src_len, d_vocab = prd.shape
        mask_pos = (src['input_ids'] == args.mask_idx) # pos of mask
        n_tgts = torch.unique(mask_pos.sum(dim=1))
        assert n_tgts.size(0) == 1 # ensure same n_tgts in each row
        assert n_tgts.item() == tgt_len # ensure n_tgts == tgt_len
        indices = torch.where(mask_pos)
        prd = prd[indices[0],indices[1]].view(bs, tgt_len, d_vocab) 
        # prd: [bs, tgt_len, d_vocab]
    elif args.lm_task == 'causal':
        src_end = src['input_ids'][:,-tgt_len:]
        two_equal = src_end == tgt['input_ids']
        assert torch.all(two_equal)
        src = {k:v[:,:-1] for k,v in src.items()} # drop last token
        prd = model(src) # [bs, src_len - 1, d_vocab]
        prd = prd[:,-tgt_len:,:] # [bs, tgt_len, d_vocab]  
    
    return prd