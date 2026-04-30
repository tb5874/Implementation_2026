# Implementation_2026

For review purposes, anonymous project repository.

This repository contains the implementation for a paper currently under review.  
The code provides the model architecture and core components used in the experiments.

---

## Information

- `network_00.py`,`network_01.py` and `network_02.py` contain the implementations of the models presented in the paper.  
- `subnetwork_01.py` and `subnetwork_02.py` contain the implementations of the sequence models.

The models used in the main text of the paper are `enhanced_model` and `enhanced_variant_model`,  
which are implemented in `network_01.py` and `network_02.py`.

The description of `initial_model` is provided in the appendix.

`network_00.py` serves as the baseline model.

---

## Argument

For `enhanced_model` and `enhanced_variant_model` classes:

### seq_option
- `tsh` (transformer-single-head)
- `tmh` (transformer-multi-head)
- `gru` (gated recurrent unit)
- `mamba` (not yet supported)

### seq_dim
- Represents the sequence length to be input into the sequence model.

### input_dim
- Represents the dimensionality of the input.

### hidden_dim
- Represents the dimensionality of the output.

---

## Input

- Shape: `[batch, sequence_length, input_dim]`

---

## Output

- Shape: `([batch, hidden_dim], [input_dim])`  
- `[batch, hidden_dim]`: learned representation  
- `[input_dim]`: weight
