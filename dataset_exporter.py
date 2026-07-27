import numpy as np
import os

def save_dataset(h_tensor, R_matrix, output_dir="data"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    np.save(os.path.join(output_dir, "h_tensor.npy"), h_tensor.numpy())
    np.save(os.path.join(output_dir, "R_matrix.npy"), R_matrix.numpy())
    print(f"Saved dataset to {output_dir}")
