import torch

def ablate(ab_type, ab_level, src, model, args):
    """
    Apply ablation to the model based on the specified type and level.
    Args:
        ab_type: 'mask', 'bernoulli', 'gaussian')
        ab_level: Level of ablation to apply (e.g., 0.1, 0.2, etc.)
        src: Input data
        model: Model to ablate
        args: Additional arguments
    Returns:
        new_src: ablated input data (includes past keys/values if applicable)
    """
    # Create a new source dictionary to avoid modifying the original
    new_src = {'input_ids': src['input_ids'].clone().detach(),
               'attention_mask': src['attention_mask'].clone().detach()}

    # Apply ablation
    if ab_type == 'mask':
        # Apply attention masking
        new_src = mask_attention(ab_level, new_src, args)
    elif ab_type in ['bernoulli', 'gaussian']:
        # Run forward pass, saving past key values
        out_pre = model.model(**src)
        past_kv = out_pre['past_key_values']

        # Inject noise
        new_src = inject_noise(new_src, past_kv, ab_level, ab_type, args)
    else:
        raise ValueError(f"Unrecognized ablation type: {ab_type}")
    return new_src

def mask_attention(ab_level, src, args):
    """
    Mask each in-context example with probability = ablation_level
    Args:
        ablation_level: Probability of masking example (e.g., 0.1, 0.2, etc.)   
        src: Input data
        args: Additional arguments
    Returns:
        src: Modified input data with new attention mask
    """

    # Get inputs and attention mask
    input_ids = src['input_ids']       # [bs, seq_len]
    attn_mask = src['attention_mask']  # [bs, seq_len]
    bs = input_ids.shape[0]  # batch size

    # Create a new mask
    new_attn_mask = attn_mask.clone().detach()
    
    # Find positions of separator tokens for all samples
    is_sep = (input_ids == args.sep_idx)  # [bs, seq_len]

    # Loop through batch
    for i in range(bs):
        # Get all separator positions for this sample
        sample_seps = (is_sep[i]).nonzero(as_tuple=True)[0].tolist()

        # Make sure there are some separators
        assert len(sample_seps) > 0, "No separators found in the input."
        
        # Define example spans based on separator positions
        example_spans = []
        
        # First example: from start to first separator
        example_spans.append((0, sample_seps[0]))
        
        # Middle examples: after previous separator to current separator
        for j in range(1, len(sample_seps)):
            example_spans.append((sample_seps[j-1] + 1, sample_seps[j]))
        
        # Number of in-context examples
        num_examples = len(example_spans)
        
        # Generate example-level attention masks
        if ab_level > 0:
            keep_prob = 1 - ab_level
            keep_attn = torch.full((num_examples,), keep_prob, 
                                    device=attn_mask.device)
            keep_attn = torch.bernoulli(keep_attn).bool()
            
            # Apply attention masking to examples that weren't kept
            for j, keep in enumerate(keep_attn):
                if not keep:
                    start, end = example_spans[j]
                    # Zero out attention mask
                    new_attn_mask[i, start:end] = 0
    
        # Update source dictionary with modified attention mask
        src['input_ids'] = input_ids
        src['attention_mask'] = new_attn_mask
    return src

def inject_noise(src, past_kv, ab_level, ab_type, args):
    """
    Inject noise into the model's past values.
    Args:
        src: Input data
        past_kv: Past key values from the model
        ablation_level: Level of noise to apply (e.g., 0.1, 0.2, etc.)
        ablation_type: Type of noise to apply ('bernoulli', 'gaussian')
    Returns:
        new_src: Modified input data with injected noise
    """

    # Get index of last sep token in the sequence
    is_sep = (src['input_ids'][0] == args.sep_idx)
    last_sep = is_sep.nonzero()[-1].item()
    assert torch.all(src['input_ids'][:, last_sep] == args.sep_idx)

    # Inject noise into past values
    noisy_past_kv = []
    for layer in past_kv:
        # Get keys and values corresponding to context (before last_sep)
        keys, values = layer
        keys = keys[:, :, :last_sep, :]
        values = values[:, :, :last_sep, :]

        # Inject noise  into values
        if ab_type == 'bernoulli':
            # Generate Bernoulli noise
            keep_prob = 1 - ab_level
            noise = (torch.rand_like(values) < keep_prob)
            noise = noise.to(device=values.device, dtype=values.dtype)
            values = values * noise
        elif ab_type == 'gaussian':
            # Generate Gaussian noise
            noise = torch.randn_like(values) * ab_level
            noise = noise.to(device=values.device, dtype=values.dtype)
            values = values + noise
        else:
            raise ValueError(f"Unrecognized noise type: {ab_type}")
        
        # Append modified keys and values to the new past key values
        noisy_past_kv.append((keys, values))
    
    src = {'input_ids': src['input_ids'][:, last_sep:],
           'attention_mask': src['attention_mask'], 
           'past_key_values': noisy_past_kv}
    return src
        