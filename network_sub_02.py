import os, sys, pathlib

import time, datetime, random, math

import numpy

import torch

class gru_block(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.block = torch.nn.GRU(
            input_size = input_dim,
            hidden_size = hidden_dim,
            num_layers = 1,
            batch_first = True
        )
        for name, p in self.block.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(p)
            elif "bias" in name:
                torch.nn.init.zeros_(p)

    def forward(self, input_x):

        # [ input ] : -1 ~ 1 ±«¿Â 

        gru_t_all, gru_t_last = self.block(input_x)
            # gru_t_all : [ batch, seq, dim ]
            # gru_t_last : [ 1, batch, dim ]

        return gru_t_all
