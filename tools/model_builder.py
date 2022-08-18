from typing import Tuple
import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from astropy.modeling import ParameterError
from tools.vocab import Vocabulary
from tools.models.conformer import Comformer

def build_conformer(
        config,
        vocab: Vocabulary,
        device: torch.device,
) -> nn.DataParallel:
    if input_dropout_p < 0.0:
        raise ParameterError("dropout probability should be positive")
    if feed_forward_dropout_p < 0.0:
        raise ParameterError("dropout probability should be positive")
    if attention_dropout_p < 0.0:
        raise ParameterError("dropout probability should be positive")
    if conv_dropout_p < 0.0:
        raise ParameterError("dropout probability should be positive")
    if input_size < 0:
        raise ParameterError("input_size should be greater than 0")
    assert conv_expansion_factor == 2, "currently, conformer conv expansion factor only supports 2"

    return nn.DataParallel(Conformer(
        num_classes=len(vocab),
        input_dim=config.n_mels,
        encoder_dim=config.encoder_dim,
        decoder_dim=config.decoder_dim,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        decoder_rnn_type=config.decoder_rnn_type,
        num_attention_heads=config.num_attention_heads,
        feed_forward_expansion_factor=config.feed_forward_expansion_factor,
        conv_expansion_factor=config.conv_expansion_factor,
        input_dropout_p=config.input_dropout_p,
        feed_forward_dropout_p=config.feed_forward_dropout_p,
        attention_dropout_p=config.attention_dropout_p,
        conv_dropout_p=config.conv_dropout_p,
        decoder_dropout_p=config.decoder_dropout_p,
        conv_kernel_size=config.conv_kernel_size,
        half_step_residual=config.half_step_residual,
        device=device,
        decoder=config.decoder,
    )).to(device)

