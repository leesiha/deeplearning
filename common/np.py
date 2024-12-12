# coding: utf-8
from common.config import GPU

if GPU:
    import cupy as np

    # scatter_add 대체 구현 함수
    def scatter_add(a, indices, updates):
        """
        Custom scatter_add implementation for CuPy
        Args:
            a: Target array (cupy.ndarray)
            indices: Indices to update (1D array)
            updates: Values to add at the specified indices (same shape as indices)
        """
        if not GPU:
            raise ValueError("scatter_add should only be used with GPU enabled.")

        # Validate input shapes
        assert len(indices) == len(
            updates), "Indices and updates must have the same length"

        for i, idx in enumerate(indices):
            a[idx] += updates[i]

    np.cuda.set_allocator(np.cuda.MemoryPool().malloc)

    print('\033[92m' + '-' * 60 + '\033[0m')
    print(' ' * 23 + '\033[92mGPU Mode (cupy)\033[0m')
    print('\033[92m' + '-' * 60 + '\033[0m\n')
else:
    import numpy as np
