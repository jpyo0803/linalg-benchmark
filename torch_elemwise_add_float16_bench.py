import torch
import singleton_timer as st
import csv

def torch_elemwise_add_bench():
  timer = st.SingletonTimer()

  batch_size = 10
  dims = [2**i for i in range(0, 13)]

  for i in range(5):
    for dim in dims:
      print(f"Iteration {i}, Dim={dim}")
      x = torch.randn((batch_size, dim, dim), dtype=torch.float16)
      y = torch.randn((batch_size, dim, dim), dtype=torch.float16)

      x_gpu = x.clone().detach()
      y_gpu = y.clone().detach()

      t = timer.start(tag=f'CPU Computation (M=K=N={dim})', category=f'CPU Computation (M=K=N={dim})', exclude=(i == 0))
      z = torch.add(x, y)
      timer.end(t)

      assert (z[0][0][0] == x[0][0][0] + y[0][0][0])

      t = timer.start(tag=f'GPU HtoD (M=K=N={dim})', category=f'GPU HtoD (M=K=N={dim})', exclude=(i == 0))
      x_gpu = x_gpu.to(torch.device("cuda:0"))
      y_gpu = y_gpu.to(torch.device("cuda:0"))
      timer.end(t)

      t = timer.start(tag=f'GPU Computation (M=K=N={dim})', category=f'GPU Computation (M=K=N={dim})', exclude=(i == 0))
      z_gpu = torch.add(x_gpu, y_gpu)
      timer.end(t)

      assert (z_gpu[0][0][0] == x_gpu[0][0][0] + y_gpu[0][0][0])

      t = timer.start(tag=f'GPU DtoH (M=K=N={dim})', category=f'GPU DtoH (M=K=N={dim})', exclude=(i == 0))
      z_gpu = z_gpu.to(torch.device("cpu"))
      timer.end(t)
      
      assert torch.allclose(z, z_gpu, atol=1e-2)
  
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

  f = open('torch_elemwise_add_cpu_vs_gpu_float16.csv', 'w')
  writer = csv.writer(f)
  writer.writerows(data)
  f.close()

if __name__=='__main__':
  torch_elemwise_add_bench()