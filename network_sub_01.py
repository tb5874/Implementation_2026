import os, sys, pathlib

import time, datetime, random, math

import numpy

import torch

class tsh_block(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        mid_dim = 128
        if ( input_dim >= mid_dim ):
            mid_dim = input_dim * 4
        self.embed_mlp = torch.nn.Linear(input_dim, mid_dim)
        self.projection_q = torch.nn.Linear(mid_dim, mid_dim)
        self.projection_k = torch.nn.Linear(mid_dim, mid_dim)
        self.projection_v = torch.nn.Linear(mid_dim, mid_dim)
        self.normalize_x1 = torch.nn.LayerNorm(mid_dim)        
        self.feedforwardnetwork = torch.nn.Sequential(
            torch.nn.Linear(mid_dim, mid_dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(mid_dim * 4, mid_dim),
        )
        self.normalize_x2 = torch.nn.LayerNorm(mid_dim)
        self.projection_head = torch.nn.Linear(mid_dim, hidden_dim)

    def forward(self, input_x):

        embed_x = self.embed_mlp(input_x)
        e_dim = embed_x.shape[-1]
        proj_q = self.projection_q(embed_x)
        proj_k = self.projection_k(embed_x)
        proj_v = self.projection_v(embed_x)
        score_qk = torch.matmul(proj_q, proj_k.transpose(-2, -1)) / (e_dim ** 0.5)
        prob_qk = torch.softmax(score_qk, dim=-1)
        self_atn = torch.matmul(prob_qk, proj_v)
        res1 = embed_x + self_atn
        norm1 = self.normalize_x1(res1)
        ffn1 = self.feedforwardnetwork(norm1)
        res2 = res1 + ffn1
        norm2 = self.normalize_x2(res2)
        proj_h = self.projection_head(norm2)

        return proj_h

def best_head(hidden_dim):
    best_a, best_b = 1, hidden_dim
    min_diff = float('inf')
    for a in range(1, int(math.sqrt(hidden_dim)) + 1):
        if hidden_dim % a == 0:
            b = hidden_dim // a
            current_ratio = b / a
            diff = abs(current_ratio - 4)
            if diff < min_diff:
                min_diff = diff
                best_a, best_b = a, b
    return best_a, best_b

class tsh_mid_block(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        mid_dim = 128
        if ( input_dim >= mid_dim ):
            mid_dim = input_dim * 4
        self.embed_mlp = torch.nn.Linear(input_dim, mid_dim)
        self.projection_q = torch.nn.Linear(mid_dim, mid_dim)
        self.projection_k = torch.nn.Linear(mid_dim, mid_dim)
        self.projection_v = torch.nn.Linear(mid_dim, mid_dim)
        self.normalize_x1 = torch.nn.LayerNorm(mid_dim)        
        self.feedforwardnetwork = torch.nn.Sequential(
            torch.nn.Linear(mid_dim, mid_dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(mid_dim * 4, mid_dim),
        )
        self.normalize_x2 = torch.nn.LayerNorm(mid_dim)
        self.projection_head = torch.nn.Linear(mid_dim, hidden_dim)

    def forward(self, input_x):

        embed_x = self.embed_mlp(input_x)
        e_dim = embed_x.shape[-1]
        proj_q = self.projection_q(embed_x)
        proj_k = self.projection_k(embed_x)
        proj_v = self.projection_v(embed_x)
        norm_q = torch.nn.functional.normalize(proj_q, dim=-1)
        norm_k = torch.nn.functional.normalize(proj_k, dim=-1)
        score_qk = torch.matmul(norm_q, norm_k.transpose(-2, -1))
        prob_qk = torch.softmax(score_qk, dim=-1)
        self_atn = torch.matmul(prob_qk, proj_v)
        res1 = embed_x + self_atn
        norm1 = self.normalize_x1(res1)
        ffn1 = self.feedforwardnetwork(norm1)
        res2 = res1 + ffn1
        norm2 = self.normalize_x2(res2)
        proj_h = self.projection_head(norm2)

        return proj_h

class tmh_block(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        mid_dim = 128
        if ( input_dim >= mid_dim ):
            mid_dim = input_dim * 4
        self.embed_mlp = torch.nn.Linear(input_dim, mid_dim)
        self.projection_q = torch.nn.Linear(mid_dim, mid_dim)
        self.projection_k = torch.nn.Linear(mid_dim, mid_dim)
        self.projection_v = torch.nn.Linear(mid_dim, mid_dim)
        self.head_num, self.head_dim = best_head(mid_dim)
        self.multi_gate = torch.nn.Linear(mid_dim, mid_dim)
        self.normalize_x1 = torch.nn.LayerNorm(mid_dim)        
        self.feedforwardnetwork = torch.nn.Sequential(
            torch.nn.Linear(mid_dim, mid_dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(mid_dim * 4, mid_dim),
        )
        self.normalize_x2 = torch.nn.LayerNorm(mid_dim)
        self.projection_head = torch.nn.Linear(mid_dim, hidden_dim)

    def forward(self, input_x):

        embed_x = self.embed_mlp(input_x)
        rb_dim, s_dim, e_dim = embed_x.shape
        proj_q = self.projection_q(embed_x)
        proj_k = self.projection_k(embed_x)
        proj_v = self.projection_v(embed_x)
        multi_q = proj_q.reshape(rb_dim, s_dim, self.head_num, self.head_dim)
        multi_k = proj_k.reshape(rb_dim, s_dim, self.head_num, self.head_dim)
        multi_v = proj_v.reshape(rb_dim, s_dim, self.head_num, self.head_dim)
        trans_q = multi_q.transpose(1, 2)
        trans_k = multi_k.transpose(1, 2)
        trans_v = multi_v.transpose(1, 2)
        score_qk = torch.matmul(trans_q, trans_k.transpose(-2, -1)) / ( self.head_dim ** 0.5 )
        prob_qk = torch.softmax(score_qk, dim=-1)
        self_atn = torch.matmul(prob_qk, trans_v)
        trans_atn = self_atn.transpose(1, 2)
        concat_atn = trans_atn.reshape(rb_dim, s_dim, e_dim)
        gate_atn = self.multi_gate(concat_atn)
        res1 = embed_x + gate_atn
        norm1 = self.normalize_x1(res1)
        ffn1 = self.feedforwardnetwork(norm1)
        res2 = res1 + ffn1
        norm2 = self.normalize_x2(res2)
        proj_h = self.projection_head(norm2)

        return proj_h

class tmh_mid_block(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        mid_dim = 128
        if ( input_dim >= mid_dim ):
            mid_dim = input_dim * 4
        self.embed_mlp = torch.nn.Linear(input_dim, mid_dim)
        self.projection_q = torch.nn.Linear(mid_dim, mid_dim)
        self.projection_k = torch.nn.Linear(mid_dim, mid_dim)
        self.projection_v = torch.nn.Linear(mid_dim, mid_dim)
        self.head_num, self.head_dim = best_head(mid_dim)
        self.multi_gate = torch.nn.Linear(mid_dim, mid_dim)
        self.normalize_x1 = torch.nn.LayerNorm(mid_dim)        
        self.feedforwardnetwork = torch.nn.Sequential(
            torch.nn.Linear(mid_dim, mid_dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(mid_dim * 4, mid_dim),
        )
        self.normalize_x2 = torch.nn.LayerNorm(mid_dim)
        self.projection_head = torch.nn.Linear(mid_dim, hidden_dim)

    def forward(self, input_x):

        embed_x = self.embed_mlp(input_x)
        rb_dim, s_dim, e_dim = embed_x.shape
        proj_q = self.projection_q(embed_x)
        proj_k = self.projection_k(embed_x)
        proj_v = self.projection_v(embed_x)
        multi_q = proj_q.reshape(rb_dim, s_dim, self.head_num, self.head_dim)
        multi_k = proj_k.reshape(rb_dim, s_dim, self.head_num, self.head_dim)
        multi_v = proj_v.reshape(rb_dim, s_dim, self.head_num, self.head_dim)
        norm_q = torch.nn.functional.normalize(multi_q, dim=-1)
        norm_k = torch.nn.functional.normalize(multi_k, dim=-1)
        trans_q = norm_q.transpose(1, 2)
        trans_k = norm_k.transpose(1, 2)
        trans_v = multi_v.transpose(1, 2)
        score_qk = torch.matmul(trans_q, trans_k.transpose(-2, -1)) 
        prob_qk = torch.softmax(score_qk, dim=-1)
        self_atn = torch.matmul(prob_qk, trans_v)
        trans_atn = self_atn.transpose(1, 2)
        concat_atn = trans_atn.reshape(rb_dim, s_dim, e_dim)
        gate_atn = self.multi_gate(concat_atn)
        res1 = embed_x + gate_atn
        norm1 = self.normalize_x1(res1)
        ffn1 = self.feedforwardnetwork(norm1)
        res2 = res1 + ffn1
        norm2 = self.normalize_x2(res2)
        proj_h = self.projection_head(norm2)

        return proj_h

class gqm_block(torch.nn.Module):

    def __init__(self, seq_dim, input_dim, hidden_dim):
        super().__init__()
        mid_dim = 128
        if ( input_dim >= mid_dim ):
            mid_dim = input_dim * 4
        self.seq_dim = seq_dim
        self.embed_mlp = torch.nn.Linear(input_dim, mid_dim)
        self.query_mlp = torch.nn.Linear(mid_dim, mid_dim)
        self.projection_q = torch.nn.Linear(mid_dim, mid_dim)
        self.projection_k = torch.nn.Linear(mid_dim, mid_dim)
        self.projection_v = torch.nn.Linear(mid_dim, mid_dim)
        self.normalize_x1 = torch.nn.LayerNorm(mid_dim)        
        self.feedforwardnetwork = torch.nn.Sequential(
            torch.nn.Linear(mid_dim, mid_dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(mid_dim * 4, mid_dim),
        )
        self.normalize_x2 = torch.nn.LayerNorm(mid_dim)
        self.projection_head = torch.nn.Linear(mid_dim, hidden_dim)

    def forward(self, input_x):
        embed_x = self.embed_mlp(input_x)
        e_dim = embed_x.shape[-1]
        g_query = self.query_mlp(embed_x[:,-1,:]).unsqueeze(-2)
        proj_q = self.projection_q(g_query)
        proj_k = self.projection_k(embed_x)
        proj_v = self.projection_v(embed_x)
        score_qk = torch.matmul(proj_q, proj_k.transpose(-2, -1)) / (e_dim ** 0.5)
        prob_qk = torch.softmax(score_qk, dim=-1)
        no_100 = 0.98 * prob_qk + 0.02 / self.seq_dim
        global_atn = torch.matmul(no_100, proj_v)
        res1 = ( g_query + global_atn )
        norm1 = self.normalize_x1(res1)
        ffn1 = self.feedforwardnetwork(norm1)
        res2 = res1 + ffn1
        norm2 = self.normalize_x2(res2).squeeze(-2)
        proj_h = self.projection_head(norm2)

        return proj_h

class gqm_mid_block(torch.nn.Module):

    def __init__(self, seq_dim, input_dim, hidden_dim):
        super().__init__()
        mid_dim = 128
        if ( input_dim >= mid_dim ):
            mid_dim = input_dim * 4
        self.seq_dim = seq_dim
        self.embed_mlp = torch.nn.Linear(input_dim, mid_dim)
        self.query_mlp = torch.nn.Linear(mid_dim, mid_dim)
        self.projection_q = torch.nn.Linear(mid_dim, mid_dim)
        self.projection_k = torch.nn.Linear(mid_dim, mid_dim)
        self.projection_v = torch.nn.Linear(mid_dim, mid_dim)
        self.normalize_x1 = torch.nn.LayerNorm(mid_dim)        
        self.feedforwardnetwork = torch.nn.Sequential(
            torch.nn.Linear(mid_dim, mid_dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(mid_dim * 4, mid_dim),
        )
        self.normalize_x2 = torch.nn.LayerNorm(mid_dim)
        self.projection_head = torch.nn.Linear(mid_dim, hidden_dim)

    def forward(self, input_x):

        embed_x = self.embed_mlp(input_x)
        e_dim = embed_x.shape[-1]
        g_query = self.query_mlp(embed_x[:,-1,:]).unsqueeze(-2)
        proj_q = self.projection_q(g_query)
        proj_k = self.projection_k(embed_x)
        proj_v = self.projection_v(embed_x)
        norm_q = torch.nn.functional.normalize(proj_q, dim=-1)
        norm_k = torch.nn.functional.normalize(proj_k, dim=-1)
        score_qk = torch.matmul(norm_q, norm_k.transpose(-2, -1))
        prob_qk = torch.softmax(score_qk, dim=-1)
        no_100 = 0.98 * prob_qk + 0.02 / self.seq_dim
        global_atn = torch.matmul(no_100, proj_v)
        res1 = ( g_query + global_atn )
        norm1 = self.normalize_x1(res1)
        ffn1 = self.feedforwardnetwork(norm1)
        res2 = res1 + ffn1
        norm2 = self.normalize_x2(res2).squeeze(-2)
        proj_h = self.projection_head(norm2)

        return proj_h


