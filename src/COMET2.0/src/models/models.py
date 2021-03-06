from src.models.gpt2 import (GPT2DoubleHeadsModel, GPT2LMHeadModel)
from .configuration_gpt2 import GPT2Config 
import torch.nn as nn


def make_model(opt, n_vocab, n_ctx, n_special, load=True,
               return_acts=True, return_probs=False,
               clf_token="<CLASS>", answer_size=None):
    if opt.exp == "generation":
        
        model = GPT2LMHeadModel.from_pretrained('gpt2-large')
        
    DEFAULT_CONFIG = GPT2Config 
    
    return model


def multi_gpu(model, devices):
    return nn.DataParallel(model, device_ids=devices)


def load_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        new_state_dict = {i[len("module."):]: j for i, j in state_dict.items()}
        model.load_state_dict(new_state_dict)
