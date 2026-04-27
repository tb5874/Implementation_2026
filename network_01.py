import os, sys, pathlib

import time, datetime, random, math

import numpy

import torch

from . import network_sub_01

from . import network_sub_02

class initial_variant_model(torch.nn.Module):

    def __init__(self, seq_option, seq_dim, input_dim, hidden_dim):

        super().__init__()
        mid_dim = 128
        if ( input_dim >= mid_dim ):
            mid_dim = input_dim * 4
        if ( seq_option == "tsh" ) :
            self.seq_start = network_sub_01.tsh_block(input_dim, mid_dim)
        elif ( seq_option == "tmh" ) :
            self.seq_start = network_sub_01.tmh_block(input_dim, mid_dim)
        elif ( seq_option == "gru" ) :
            self.seq_start = network_sub_02.gru_block(input_dim, mid_dim)
        elif ( seq_option == "mamba" ) :
            import mamba_ssm
            self.seq_start = mamba_ssm.Mamba(mid_dim, 1)
        else:
            raise RuntimeError("need to set seq_trs option")
        self.seq_encoder = network_sub_01.gqm_block(seq_dim, mid_dim, mid_dim)
        self.seq_head = torch.nn.Sequential( torch.nn.Linear(mid_dim, input_dim) )
        self.seq_mlp = torch.nn.Linear(input_dim, mid_dim)
        self.seq_add = torch.nn.Linear(mid_dim, mid_dim)
        self.seq_norm1 = torch.nn.LayerNorm(mid_dim)
        self.seq_ffn = torch.nn.Sequential(
            torch.nn.Linear(mid_dim, mid_dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(mid_dim * 4, mid_dim),
        )
        self.seq_norm2 = torch.nn.LayerNorm(mid_dim)
        self.seq_last = torch.nn.Linear(mid_dim, hidden_dim)

    def forward(self, input_x):

        start_x = self.seq_start(input_x)
        encoder_x = self.seq_encoder(start_x)
        get_head = self.seq_head(encoder_x)
        get_softplus = 1.0 + torch.nn.functional.softplus(get_head)
        get_w = get_softplus / (1.0 + get_softplus/100.0)
        get_round = torch.round(get_w)
        get_int = get_w + (get_round - get_w).detach()
        get_xw = input_x[:,-1,:] * get_int
        get_mlp = self.seq_mlp(get_xw)
        get_add = self.seq_add(get_mlp)
        get_res1 = ( get_mlp + get_add )
        get_norm1 = self.seq_norm1(get_res1)
        get_ffn = self.seq_ffn(get_norm1)
        get_res2 = get_res1 + get_ffn
        get_norm2 = self.seq_norm2(get_res2)
        get_last = self.seq_last(get_norm2)

        return get_last, get_int

class enhanced_variant_model(torch.nn.Module):

    def __init__(self, seq_option, seq_dim, input_dim, hidden_dim):

        super().__init__()
        mid_dim = 128
        if ( input_dim >= mid_dim ):
            mid_dim = input_dim * 4
        if ( seq_option == "tsh" ) :
            self.seq_start = network_sub_01.tsh_mid_block(input_dim, mid_dim)
        elif ( seq_option == "tmh" ) :
            self.seq_start = network_sub_01.tmh_mid_block(input_dim, mid_dim)
        elif ( seq_option == "gru" ) :
            self.seq_start = network_sub_02.gru_block(input_dim, mid_dim)
        elif ( seq_option == "mamba" ) :
            import mamba_ssm
            self.seq_start = mamba_ssm.Mamba(mid_dim, 1)
        else:
            raise RuntimeError("need to set seq_trs option")
        self.seq_encoder = network_sub_01.gqm_mid_block(seq_dim, mid_dim, mid_dim)
        self.seq_head = torch.nn.Sequential( torch.nn.Linear(mid_dim, input_dim) )
        self.seq_mlp = torch.nn.Linear(input_dim, mid_dim)
        self.seq_add = torch.nn.Linear(mid_dim, mid_dim)
        self.seq_norm1 = torch.nn.LayerNorm(mid_dim)
        self.seq_ffn = torch.nn.Sequential(
            torch.nn.Linear(mid_dim, mid_dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(mid_dim * 4, mid_dim),
        )
        self.seq_norm2 = torch.nn.LayerNorm(mid_dim)
        self.seq_last = torch.nn.Linear(mid_dim, hidden_dim)

    def forward(self, input_x):

        start_x = self.seq_start(input_x)
        encoder_x = self.seq_encoder(start_x)
        get_head = self.seq_head(encoder_x)
        get_softplus = torch.nn.functional.softplus(get_head)
        get_w = get_softplus / (1.0 + get_softplus/100.0)
        get_xw = input_x[:,-1,:] * get_w
        get_mlp = self.seq_mlp(get_xw)
        get_add = self.seq_add(get_mlp)
        get_res1 = ( get_mlp + get_add )
        get_norm1 = self.seq_norm1(get_res1)
        get_ffn = self.seq_ffn(get_norm1)
        get_res2 = get_res1 + get_ffn
        get_norm2 = self.seq_norm2(get_res2)
        get_last = self.seq_last(get_norm2)

        return get_last, get_w
