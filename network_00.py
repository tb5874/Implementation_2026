import os, sys, pathlib

import time, datetime, random, math

import numpy

import torch

class initial_model(torch.nn.Module):

    def __init__(self, seq_option, seq_dim, input_dim, hidden_dim):

        super().__init__()
        mid_dim = 128
        if ( input_dim >= mid_dim ):
            mid_dim = input_dim * 4
        self.seq_start = torch.nn.Sequential(
            torch.nn.Linear(input_dim, mid_dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(mid_dim * 4, mid_dim),
        )
        self.seq_encoder = torch.nn.Linear(mid_dim, mid_dim)
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

        start_x = self.seq_start(input_x[:,-1,:])
        encoder_x = self.seq_encoder(start_x)
        get_head = self.seq_head(encoder_x)
        get_mlp = self.seq_mlp(get_head)
        get_add = self.seq_add(get_mlp)
        get_res1 = ( get_mlp + get_add )
        get_norm1 = self.seq_norm1(get_res1)
        get_ffn = self.seq_ffn(get_norm1)
        get_res2 = get_res1 + get_ffn
        get_norm2 = self.seq_norm2(get_res2)
        get_last = self.seq_last(get_norm2)

        return get_last, 0
