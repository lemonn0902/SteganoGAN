#!/usr/bin/env python3
import argparse
import json
import os
import sys
from time import time
from custom_loader import get_loaders

import torch

# Check for MPS (Metal Performance Shaders) support on Apple Silicon
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/Users/shreya/Desktop/steg/SteganoGAN')))

from steganogan import SteganoGAN
from steganogan.critics import BasicCritic
from steganogan.decoders import DenseDecoder
from steganogan.encoders import BasicEncoder, DenseEncoder, ResidualEncoder

def main():
    # Set seed and timestamp
    torch.manual_seed(42)
    timestamp = int(time())

    # Parse command line arguments
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
    
    # Initialize SteganoGAN model
    steganogan = SteganoGAN(
        data_depth=args.data_depth,
        encoder=encoder,
        decoder=DenseDecoder,
        critic=BasicCritic,
        hidden_size=args.hidden_size,
        lr=args.lr,
        cuda=False,  # Do not use CUDA
        verbose=True,
        log_dir=os.path.join('models', str(timestamp))
    )
    
    # Ensure the output directory exists
    os.makedirs(os.path.join("models", str(timestamp)), exist_ok=True)

    # Save config
    with open(os.path.join("models", str(timestamp), "config.json"), "wt") as fout:
        fout.write(json.dumps(args.__dict__, indent=2, default=lambda o: str(o)))

    # Move each part of the SteganoGAN to the correct device (MPS or CPU)
    steganogan.encoder.to(device)
    steganogan.decoder.to(device)
    steganogan.critic.to(device)

    # Training
    for batch in train:
        cover, message = batch  # Unpack the tuple
        print("Cover shape:", cover.shape)
        print("Message shape:", message.shape)
        cover = cover.to(device)
        message = message.to(device)
        # Example: Forward pass (modify as needed)
        encoded = steganogan.encoder(cover, message)

    steganogan.fit(train, validation, epochs=args.epochs)
    steganogan.save(os.path.join("models", str(timestamp), "weights.steg"))
    if args.output:
        steganogan.save(args.output)

if __name__ == '__main__':
    main()
