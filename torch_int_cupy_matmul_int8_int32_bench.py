import torch
import singleton_timer as st
import csv
from torch_int._CUDA import bmm_s8t_s8n_s32t
import cupy as cp
import numpy as np


def torch_square_matmul_bench():
    timer = st.SingletonTimer()

    batch_size = 10
    dims = [2**i for i in range(4, 13)]

    for i in range(5):
        for dim in dims:
            print(f"Iteration {i}, Dim={dim}")
            x = torch.randint(-128, 127, (batch_size,
                              dim, dim), dtype=torch.int8)
            y = torch.randint(-128, 127, (batch_size,
                              dim, dim), dtype=torch.int8)

            x_np = x.clone().detach().numpy().astype(np.int32)
            y_np = np.moveaxis(y.clone().detach().numpy(), -1, -2).astype(np.int32)

            t = timer.start(
                tag=f'Torch GPU HtoD (M=K=N={dim})', category=f'Torch GPU HtoD (M=K=N={dim})', exclude=(i == 0))
            x = x.to(torch.device('cuda:0'))
            y = y.to(torch.device('cuda:0'))
            timer.end(t)

            t = timer.start(
                tag=f'Torch GPU Computation (M=K=N={dim})', category=f'Torch GPU Computation (M=K=N={dim})', exclude=(i == 0))
            z = bmm_s8t_s8n_s32t(x, y)
            timer.end(t)

            t = timer.start(
                tag=f'Torch GPU DtoH (M=K=N={dim})', category=f'Torch GPU DtoH (M=K=N={dim})', exclude=(i == 0))
            z = z.to(torch.device("cpu"))
            timer.end(t)

            t = timer.start(
                tag=f'Cupy GPU HtoD (M=K=N={dim})', category=f'Cupy GPU HtoD (M=K=N={dim})', exclude=(i == 0))
            x_cp = cp.asarray(x_np)
            y_cp = cp.asarray(y_np)
            timer.end(t)

            t = timer.start(
                tag=f'Cupy GPU Computation (M=K=N={dim})', category=f'Cupy GPU Computation (M=K=N={dim})', exclude=(i == 0))
            z_cp = cp.matmul(x_cp, y_cp)
            timer.end(t)

            t = timer.start(
                tag=f'Cupy GPU DtoH (M=K=N={dim})', category=f'Cupy GPU DtoH (M=K=N={dim})', exclude=(i == 0))
            z_np = cp.asnumpy(z_cp)
            timer.end(t)

            z_np = torch.tensor(z_np)

            assert torch.allclose(z, z_np, atol=1e-2)

    raw_data = timer.display_summary()

    data = []

    data_torch_htod = ['Torch GPU htod ']
    for dim in dims:
        key = f'Torch GPU HtoD (M=K=N={dim})'
        data_torch_htod.append(raw_data[key])
    data.append(data_torch_htod)

    data_torch_gpu = ['Torch GPU comp ']
    for dim in dims:
        key = f'Torch GPU Computation (M=K=N={dim})'
        data_torch_gpu.append(raw_data[key])
    data.append(data_torch_gpu)

    data_torch_dtoh = ['Torch GPU dtoh ']
    for dim in dims:
        key = f'Torch GPU DtoH (M=K=N={dim})'
        data_torch_dtoh.append(raw_data[key])
    data.append(data_torch_dtoh)

    data_cupy_htod = ['Cupy GPU htod ']
    for dim in dims:
        key = f'Cupy GPU HtoD (M=K=N={dim})'
        data_cupy_htod.append(raw_data[key])
    data.append(data_cupy_htod)

    data_cupy_gpu = ['Cupy GPU comp ']
    for dim in dims:
        key = f'Cupy GPU Computation (M=K=N={dim})'
        data_cupy_gpu.append(raw_data[key])
    data.append(data_cupy_gpu)

    data_cupy_dtoh = ['Cupy GPU dtoh ']
    for dim in dims:
        key = f'Cupy GPU DtoH (M=K=N={dim})'
        data_cupy_dtoh.append(raw_data[key])
    data.append(data_cupy_dtoh)

    f = open('torch_int_cupy_matmul_square_int8_int32.csv', 'w')
    writer = csv.writer(f)
    writer.writerows(data)
    f.close()


if __name__ == '__main__':
    torch_square_matmul_bench()
