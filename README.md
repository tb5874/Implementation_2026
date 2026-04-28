# Implementation_2026

For review purposes, anonymous project repository.

This repository contains the implementation for a paper currently under review.

The code provides the model architecture and core components used in the experiments.

# Information
network_01.py and network_02.py contain the implementations of the models presented in the paper,
while subnetwork_01.py and subnetwork_02.py contain the implementations of the sequence models.

The models used in the main text of the paper are 'enhanced_model' and 'enhanced_variant_model',
which are implemented in the files network_01.py and network_02.py.

The description of initial_model is provided in the appendix.

# Argument
For enhanced_model, enhanced_variant_model calss

seq_option
	tsh ( transformer-single-head )
	tmh ( transformer-multi-head )
	gru ( gated-recurrent-unit )
	mamba ( not yet )
seq_dim
	This argument represents the sequence length to be input into the sequence model.
input_dim
	This argument represents the dimensionality of the input.
hidden_dim
	This argument represents the dimensionality of the output.

# Input 
Input shape is [batch, sequence length, input shape]

# Output
The output has shape ([batch,hidden_dim],[input_dim]).
The [batch,hidden_dim] component represents the learned representation,
while the [input_dim] component corresponds to the weight.