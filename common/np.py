# coding: utf-8
from common.config import GPU

if GPU:
    import cupy as cp

    # scatter_add 대체 구현 함수
    def scatter_add(a, indices, updates):
        """
        scatter_add 대체 구현
        a[indices] += updates를 수행
        """
        cp.scatter_add(a, indices, updates)

    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    np = cp  # np를 cupy로 매핑

    print('\033[92m' + '-' * 60 + '\033[0m')
    print(' ' * 23 + '\033[92mGPU Mode (cupy)\033[0m')
    print('\033[92m' + '-' * 60 + '\033[0m\n')
else:
    import numpy as np
