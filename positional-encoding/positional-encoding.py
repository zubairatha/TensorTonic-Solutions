import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    positions = np.arange(seq_len)[:, None] # [seq_len, 1]
    pair_indices = np.arange((d_model+1)//2) # 1D, shape ((d_model+1)//2,)
    # (d_model+1)//2 lets us handle the odd d_model scenario
    # if d_model = 33 - we need to calculate 17 pairs. (33+1)//2 = 17

    divisors = base ** (2 * pair_indices/d_model) # 1D, shape ((d_model+1)//2,)

    angles = positions/divisors # [seq_len, (d_model+1)//2]

    pe = np.zeros((seq_len, d_model), dtype=float) # [seq_len, d_model]

    pe[:,0::2] = np.sin(angles)
    pe[:,1::2] = np.cos(angles[:,:d_model//2]) 
    # For odd d_model, skip the cosine for the last angle.
    # Its sine still fills the final column.

    return pe