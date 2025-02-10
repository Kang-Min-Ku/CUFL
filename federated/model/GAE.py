from typing import Optional
import torch
import torch.nn as nn
from torch_geometric.nn.inits import reset
from .decoder import InnerProductDecoder

class GAE(nn.Module):
    def __init__(self, encoder,
                 decoder:Optional[nn.Module]=None,
                 encoder_forward_default_args:Optional[dict]=None,
                 decoder_forward_default_args:Optional[dict]=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        self.encoder_forward_default_args = {} if encoder_forward_default_args is None else encoder_forward_default_args
        self.decoder_forward_default_args = {} if decoder_forward_default_args is None else decoder_forward_default_args
        GAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        kwargs = {**self.encoder_forward_default_args, **kwargs}
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        kwargs = {**self.decoder_forward_default_args, **kwargs}
        return self.decoder(*args, **kwargs)

    def encode_decode(self, *args, **kwargs):
        kwargs = {**self.encoder_forward_default_args, **kwargs}
        z = self.encoder(*args, **kwargs)
        return self.decoder(z)