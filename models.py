import os
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM
from transformers import BitsAndBytesConfig

def get_model(args):
    if args.model_name == 'llama2':
        dev_map = 'auto' if args.use_cuda else 'cpu'
        model = HuggingFaceModelWrapper(args.model_name, args.n_layers, 
                                        args.n_heads, args.d_model, args.d_ff, 
                                        args.d_vocab, args.dropout_p, 
                                        args.max_len, args.pad_idx,
                                        args.llama2_size, args.use_quant,
                                        args.model_dtype, dev_map,
                                        args.device, args.cache_dir)
    else:
        print("Model name not recognized", flush=True)
    return model

class HuggingFaceModelWrapper(nn.Module):
    def __init__(self, model_name, n_layers, n_heads, 
                 d_model, d_ff, d_vocab, 
                 dropout_p, max_len, pad_idx, 
                 llama2_size, use_quant, model_dtype,
                 dev_map, device, cache_dir):
        super().__init__()
        self.model_name = model_name
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_vocab = d_vocab
        self.dropout_p = dropout_p
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.llama2_size = llama2_size
        self.use_quant = use_quant
        self.model_dtype = model_dtype
        self.dev_map = dev_map # device map
        self.device = device # for models without dev_map
        self.cache_dir = cache_dir 
    
        if model_name == 'llama2':
            if self.llama2_size is None:
                config = LlamaConfig(vocab_size=d_vocab, # vocab size
                    max_position_embeddings=max_len, # n positional embeddings
                    embedding_size=d_model, # embedding size
                    hidden_size=d_model, # hidden size
                    num_hidden_layers=n_layers, # number of layers
                    num_attention_heads=n_heads, # number of attention heads
                    intermediate_size=d_ff, # feedforward size
                    hidden_dropout_prob=dropout_p, # dropout in embeddings
                    attention_probs_dropout_prob=dropout_p, # dropout attention 
                    pad_token_id=pad_idx,
                    bos_token_id=None,
                    eos_token_id=None) 
                self.model = LlamaForCausalLM(config)
                self.model = self.model.to(device)
            else:
                name = f'Llama-2-{self.llama2_size}-hf'
                path = os.path.join(cache_dir, name)
                if self.use_quant:
                    print("Using quantization to load model...")
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type='nf4',
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.float16)
                    self.model = LlamaForCausalLM.from_pretrained(path,
                                device_map=self.dev_map,
                                quantization_config=bnb_config)
                else:
                    if self.model_dtype == 'fp16':
                        model_dtype = torch.float16 
                    else:
                        model_dtype = torch.float32
                    self.model = LlamaForCausalLM.from_pretrained(path,
                                    device_map=self.dev_map,
                                    torch_dtype=model_dtype)
                print("Device map:", self.model.hf_device_map)
                print("Torch dtypes:")
                for name, param in self.model.named_parameters():
                    print(name, param.dtype)
        else:
            raise ValueError(f"model_name not recognized: {model_name}")
        
    def forward(self, src):
        out = self.model(**src)
        out = out.logits
        return out