import numpy as np

def rotate_2d(pair, theta):
    """
    Rotate a 2D vector by theta radians.
    Args:
        pair: np.array of shape (2,)
        theta: rotation angle in radians
    Returns:import numpy as np

def rotate_2d(pair, theta):

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rot_matrix = np.array([
        [cos_t, -sin_t],
        [sin_t,  cos_t]
    ])
    return rot_matrix @ pair


def apply_rope_numpy(x, base_theta=10000.0):

    seq_len, dim = x.shape
    assert dim % 2 == 0, "Embedding dimension must be even"

    # Compute frequencies
    dim_half = dim // 2
    idx = np.arange(0, dim_half, 1)
    inv_freq = 1.0 / (base_theta ** (idx / dim_half))

    output = np.zeros_like(x)

    for pos in range(seq_len):
        theta_pos = pos * inv_freq  # shape: (dim/2,)
        for i in range(dim_half):
            pair = x[pos, 2 * i : 2 * i + 2]
            theta = theta_pos[i]
            output[pos, 2 * i : 2 * i + 2] = rotate_2d(pair, theta)

    return output

  # 4 tokens, 4-dim embeddings
X = np.array([
    [1.0, 0.0, 0.5, 0.5],  # "The"
    [0.0, 1.0, 0.5, -0.5], # "quick"
    [1.0, 1.0, 0.0, 1.0],  # "brown"
    [0.0, 0.0, 1.0, 1.0]   # "fox"
])
ans=apply_rope_numpy(X)
print(ans)

        Rotated vector of shape (2,)
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rot_matrix = np.array([
        [cos_t, -sin_t],
        [sin_t,  cos_t]
    ])
    return rot_matrix @ pair


def apply_rope_numpy(x, base_theta=10000.0):

    seq_len, dim = x.shape
    assert dim % 2 == 0, "Embedding dimension must be even"

    # Compute frequencies
    dim_half = dim // 2
    idx = np.arange(0, dim_half, 1)
    inv_freq = 1.0 / (base_theta ** (idx / dim_half))

    output = np.zeros_like(x)

    for pos in range(seq_len):
        theta_pos = pos * inv_freq  # shape: (dim/2,)
        for i in range(dim_half):
            pair = x[pos, 2 * i : 2 * i + 2]
            theta = theta_pos[i]
            output[pos, 2 * i : 2 * i + 2] = rotate_2d(pair, theta)

    return output

  # 4 tokens, 4-dim embeddings
X = np.array([
    [1.0, 0.0, 0.5, 0.5],  # "The"
    [0.0, 1.0, 0.5, -0.5], # "quick"
    [1.0, 1.0, 0.0, 1.0],  # "brown"
    [0.0, 0.0, 1.0, 1.0]   # "fox"
])
ans=apply_rope_numpy(X)
print(ans)
