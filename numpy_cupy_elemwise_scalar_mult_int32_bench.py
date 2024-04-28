import numpy as np
import cupy as cp
import singleton_timer as st
import csv


def npcp_elemwise_scalar_mult_bench():
    timer = st.SingletonTimer()

    batch_size = 10
    dims = [2**i for i in range(0, 14)]

    for i in range(5):
        for dim in dims:
            print(f"Iteration {i}, Dim={dim}")
            x = np.random.randint(-128, 127, (batch_size,
                                  dim, dim), dtype=np.int32)

            x_gpu = x.copy()

            t = timer.start(
                tag=f'CPU Computation (M=K=N={dim})', category=f'CPU Computation (M=K=N={dim})', exclude=(i == 0))
            z = np.multiply(x, 88)
            timer.end(t)

            assert (z[0][0][0] == x[0][0][0] * 88)

            t = timer.start(
                tag=f'GPU HtoD (M=K=N={dim})', category=f'GPU HtoD (M=K=N={dim})', exclude=(i == 0))
            x_gpu = cp.asarray(x_gpu)
            timer.end(t)

            t = timer.start(
                tag=f'GPU Computation (M=K=N={dim})', category=f'GPU Computation (M=K=N={dim})', exclude=(i == 0))
            z_gpu = cp.multiply(x_gpu, 88)
            timer.end(t)
            assert (z_gpu[0][0][0] == x_gpu[0][0][0] * 88)

            t = timer.start(
                tag=f'GPU DtoH (M=K=N={dim})', category=f'GPU DtoH (M=K=N={dim})', exclude=(i == 0))
            z_gpu = cp.asnumpy(z_gpu)
            timer.end(t)
            assert np.array_equal(z, z_gpu)

    raw_data = timer.display_summary()

    data = []

    data_cpu = ['CPU comp. ']
    for dim in dims:
        key = f'CPU Computation (M=K=N={dim})'
        data_cpu.append(raw_data[key])
    data.append(data_cpu)

    data_gpu = ['GPU comp. ']
    for dim in dims:
        key = f'GPU Computation (M=K=N={dim})'
        data_gpu.append(raw_data[key])
    data.append(data_gpu)

    data_gpu_htod = ['GPU htod. ']
    for dim in dims:
        key = f'GPU HtoD (M=K=N={dim})'
        data_gpu_htod.append(raw_data[key])
    data.append(data_gpu_htod)

    data_gpu_dtoh = ['GPU dtoh. ']
    for dim in dims:
        key = f'GPU DtoH (M=K=N={dim})'
        data_gpu_dtoh.append(raw_data[key])
    data.append(data_gpu_dtoh)

    f = open('numpy_cupy_elemwise_scalar_mult_cpu_vs_gpu_int32.csv', 'w')
    writer = csv.writer(f)
    writer.writerows(data)
    f.close()


if __name__ == '__main__':
    npcp_elemwise_scalar_mult_bench()
