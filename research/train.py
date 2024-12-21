#!/usr/bin/env python3
import argparse
import json
import os
import sys
from time import time
from custom_loader import get_loaders

import torch
from torch.cuda import is_available as cuda_available

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/Users/shreya/Desktop/steg/SteganoGAN')))

from steganogan import SteganoGAN
from steganogan.critics import BasicCritic
from steganogan.decoders import DenseDecoder
from steganogan.encoders import BasicEncoder, DenseEncoder, ResidualEncoder

def main():
    # Check if GPU is available
    device = torch.device("cuda" if cuda_available() else "cpu")
    torch.manual_seed(42)
    timestamp = int(time())

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=4, type=int)
    parser.add_argument('--encoder', default="dense", type=str)
    parser.add_argument('--data_depth', default=1, type=int)
    parser.add_argument('--hidden_size', default=32, type=int)
    parser.add_argument('--dataset', default="div2k", type=str)
    parser.add_argument('--output', default=False, type=str)
    parser.add_argument('--lr', default=0.001, type=float)

    args = parser.parse_args()

    # Loaders
    train = get_loaders(
        "datasets/combined-dataset", 
        "datasets/messages", 
        "datasets/metadata.csv", 
        batch_size=8, 
        shuffle=True
    )

    validation = get_loaders(
        "datasets/combined-dataset", 
        "datasets/messages", 
        "datasets/metadata.csv", 
        batch_size=8, 
        shuffle=False
    )

    encoder = {
        "basic": BasicEncoder,
        "residual": ResidualEncoder,
        "dense": DenseEncoder,
    }[args.encoder]
    
    # Initialize SteganoGAN model with GPU if available
    steganogan = SteganoGAN(
        data_depth=args.data_depth,
        encoder=encoder,
        decoder=DenseDecoder,
        critic=BasicCritic,
        hidden_size=args.hidden_size,
        lr=args.lr,
        cuda=cuda_available(),  # Check if GPU is available
        verbose=True,
        log_dir=os.path.join('models', str(timestamp))
    )
    
    # Save config
    with open(os.path.join("models", str(timestamp), "config.json"), "wt") as fout:
        fout.write(json.dumps(args.__dict__, indent=2, default=lambda o: str(o)))

    # Move each part of the SteganoGAN to the correct device (GPU or CPU)
    steganogan.encoder.to(device)
    steganogan.decoder.to(device)
    steganogan.critic.to(device)

    # Training
    steganogan.fit(train, validation, epochs=args.epochs)
    steganogan.save(os.path.join("models", str(timestamp), "weights.steg"))
    if args.output:
        steganogan.save(args.output)

if __name__ == '__main__':
    main()
