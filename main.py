import model
import simulate
import dataset

import torch
from torch.utils.data import Dataset, DataLoader

def main():
    print("generating noise model")
    noise_model = simulate.get_noise(0.01, 0.01)

    # Create a dataset
    print("creating dataset")
    ds = dataset.SurfaceCodeDataset(noise_model)

    # Create a model
    print("creating model")
    m = model.SurfaceCodeDecoder(
        n_attn_dims=8,
        n_heads=4,
        n_attn_layers=2,
        n_ff_layers=2,
        n_ff_dims=64,
        dropout=0.1,
        max_seq_len=511,
    )
    
    # Dataloader
    print("Creating DataLoader")
    dl = DataLoader(ds, batch_size=32, num_workers=2)

    print("Iterating over DataLoader")
    for batch in dl:
        break

    # Train the model
    # m.train(ds, s, 100, 100, 0.001, 0.9, 0.1, 10, 10)


if __name__ == '__main__':
    main()