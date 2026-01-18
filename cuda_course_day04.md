# 第 4 天：共享内存基础

## 学习目标

- 理解共享内存的作用与特点
- 掌握 Bank Conflicts 的概念及避免方法
- 学习 CUB 库的 BlockLoad/BlockStore
- 实现简单的 block-level 操作
- 分析 `quantization.cuh` 中的共享内存使用

## 1. 共享内存（Shared Memory）基础

### 1.1 什么是共享内存？

**共享内存**是位于 GPU 芯片上的快速存储器：

| 特性 | 描述 |
|------|------|
| **位置** | 每个 SM（Streaming Multiprocessor）上 |
| **容量** | ~48KB-164KB（取决于架构） |
| **延迟** | ~20-30 cycles（比全局内存快 10-20 倍） |
| **作用域** | 同一 block 内的所有线程可见 |
| **生命周期** | Block 启动到结束 |

### 1.2 为什么需要共享内存？

**问题**：全局内存太慢（300-600 cycles）

**解决方案**：
1. 从全局内存加载数据到共享内存（一次性成本）
2. 在共享内存中进行多次读写（快速）
3. 将结果写回全局内存

**典型应用**：
- 矩阵乘法中的 tile
- Reduction 操作中的中间结果
- 需要线程间通信的算法

### 1.3 声明共享内存

#### 静态分配（编译时大小已知）

```cuda
__global__ void static_shared_kernel() {
    __shared__ float smem[256];  // 256 个 float
    // 使用 smem...
}
```

#### 动态分配（运行时确定大小）

```cuda
__global__ void dynamic_shared_kernel() {
    extern __shared__ float smem[];  // 大小由启动时指定
    // 使用 smem...
}

// 启动时指定大小
kernel<<<grid, block, 256 * sizeof(float)>>>();
//                     ^^^^^^^^^^^^^^^^^^^^
//                     动态共享内存大小（字节）
```

#### 多个动态共享内存数组

```cuda
__global__ void multi_array_kernel() {
    extern __shared__ char smem[];  // 使用 char* 作为基础

    // 手动分配不同类型的数组
    float* smem_float = (float*)smem;
    int* smem_int = (int*)(smem + 256 * sizeof(float));

    // 使用...
}
```

## 2. Bank Conflicts

### 2.1 什么是 Memory Banks？

共享内存被划分为 **32 个 banks**（通道）：

```
Bank 0: 地址 0, 32, 64, 96, ...
Bank 1: 地址 1, 33, 65, 97, ...
...
Bank 31: 地址 31, 63, 95, 127, ...
```

**每个 bank 的宽度**：4 字节（32 位）

### 2.2 Bank Conflict 的发生

当 warp 中的**多个线程访问同一 bank 的不同地址**时，发生 bank conflict。

#### ✅ 无冲突访问（最优）

```cuda
__shared__ float smem[32];

// 每个线程访问不同的 bank
float value = smem[threadIdx.x];  // Thread 0→Bank 0, Thread 1→Bank 1, ...
```

#### ⚠️ 广播（Broadcast，无惩罚）

```cuda
// 所有线程访问同一地址 → 广播
float value = smem[0];  // 所有线程读相同地址，一次广播完成
```

#### ❌ 2-way Bank Conflict（性能损失 50%）

```cuda
__shared__ float smem[64];

// 线程 0 和线程 1 都访问 Bank 0 的不同地址
float value = smem[threadIdx.x * 2];
// Thread 0: smem[0] → Bank 0
// Thread 1: smem[2] → Bank 0 (冲突！)
// Thread 2: smem[4] → Bank 0 (冲突！)
// 需要 2 次事务
```

#### ❌ 32-way Bank Conflict（性能损失 97%）

```cuda
__shared__ float smem[32][32];

// 最坏情况：所有线程访问同一列
float value = smem[threadIdx.x][0];
// 所有 32 个线程都访问 Bank 0 → 需要 32 次事务！
```

### 2.3 避免 Bank Conflicts 的方法

#### 方法 1：Padding（填充）

```cuda
// ❌ 有冲突
__shared__ float smem[32][32];

// ✅ 无冲突（添加 1 列 padding）
__shared__ float smem[32][33];  // 每行多 1 个元素
// 现在列访问不会冲突：
// smem[0][0] → Bank 0
// smem[1][0] → Bank 33 % 32 = Bank 1
// smem[2][0] → Bank 66 % 32 = Bank 2
```

#### 方法 2：使用不同的访问模式

```cuda
// ❌ 列访问（冲突）
for (int i = 0; i < 32; ++i) {
    sum += smem[i][threadIdx.x];
}

// ✅ 行访问（无冲突）
for (int i = 0; i < 32; ++i) {
    sum += smem[threadIdx.x][i];
}
```

#### 方法 3：使用 Swizzle（交错）

复杂的访问模式下使用位运算重新排列索引（Attention kernel 中常见）。

## 3. CUB 库简介

### 3.1 什么是 CUB？

**CUB（CUDA Unbound）** 是 NVIDIA 提供的高性能 CUDA 原语库：

- Block-level 操作：BlockLoad, BlockStore, BlockReduce, BlockScan
- Warp-level 操作：WarpReduce, WarpScan
- Device-level 操作：DeviceReduce, DeviceSort

**优势**：
- 自动处理 bank conflicts
- 高度优化的实现
- 类型安全的模板

### 3.2 BlockLoad 基础

`BlockLoad` 提供高效的 block-level 数据加载。

#### 基本用法

```cuda
#include <cub/cub.cuh>

__global__ void load_example_kernel(float* input, float* output) {
    // 1. 定义 BlockLoad 类型
    constexpr int BLOCK_THREADS = 256;
    constexpr int ITEMS_PER_THREAD = 4;
    typedef cub::BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD,
                           cub::BLOCK_LOAD_VECTORIZE> BlockLoad;

    // 2. 分配共享内存
    __shared__ typename BlockLoad::TempStorage temp_storage;

    // 3. 每个线程的本地数组
    float thread_data[ITEMS_PER_THREAD];

    // 4. 执行 BlockLoad
    int block_offset = blockIdx.x * BLOCK_THREADS * ITEMS_PER_THREAD;
    BlockLoad(temp_storage).Load(input + block_offset, thread_data);

    // 5. 使用加载的数据...
}
```

**加载模式**：
- `BLOCK_LOAD_DIRECT`：直接加载（可能有冲突）
- `BLOCK_LOAD_VECTORIZE`：向量化加载（推荐）
- `BLOCK_LOAD_TRANSPOSE`：转置加载（避免冲突）
- `BLOCK_LOAD_WARP_TRANSPOSE`：Warp 级转置

## 4. 案例分析：`quantization.cuh`

### 4.1 完整代码

文件位置：`include/flashinfer/quantization.cuh`

```cuda
template <BitOrder BITORDER>
__global__ void PackBitsKernel(bool* input, uint8_t* output, int64_t num_elements) {
  // 1. 计算起始偏移
  int64_t start_offset = static_cast<int64_t>(blockIdx.x) * blockDim.x * 8;
  int64_t tx = threadIdx.x;

  uint8_t ret = 0;
  bool input_vec[8];  // 每个线程处理 8 个 bool

  // 2. 定义 CUB BlockLoad 类型
  typedef cub::BlockLoad<bool, 256, 8, cub::BLOCK_LOAD_VECTORIZE> BlockLoad;

  // 3. 共享内存用于 BlockLoad
  __shared__ typename BlockLoad::TempStorage temp_storage;

  // 4. 计算有效元素数量（处理边界）
  int block_items_end =
      (num_elements - start_offset > INT32_MAX) ? INT32_MAX : num_elements - start_offset;

  // 5. 使用 BlockLoad 加载数据
  BlockLoad(temp_storage).Load(input + start_offset, input_vec, block_items_end, /*default=*/0);

  // 6. 位打包：8 个 bool → 1 个 uint8_t
  if constexpr (BITORDER == BitOrder::kBig) {
    ret = (input_vec[0] << 7) | (input_vec[1] << 6) | (input_vec[2] << 5) | (input_vec[3] << 4) |
          (input_vec[4] << 3) | (input_vec[5] << 2) | (input_vec[6] << 1) | input_vec[7];
  } else {
    ret = (input_vec[7] << 7) | (input_vec[6] << 6) | (input_vec[5] << 5) | (input_vec[4] << 4) |
          (input_vec[3] << 3) | (input_vec[2] << 2) | (input_vec[1] << 1) | input_vec[0];
  }

  // 7. 写回结果
  if (start_offset + tx * 8 < num_elements)
    output[start_offset / 8 + tx] = ret;
}
```

### 4.2 代码分解讲解

#### **Step 1-3：CUB BlockLoad 设置**

```cuda
typedef cub::BlockLoad<bool, 256, 8, cub::BLOCK_LOAD_VECTORIZE> BlockLoad;
__shared__ typename BlockLoad::TempStorage temp_storage;
```

- `BlockLoad<bool, 256, 8, ...>`：
  - 类型：`bool`
  - 线程数：256
  - 每线程项数：8
  - 加载模式：向量化

- `TempStorage`：CUB 需要的共享内存

#### **Step 4：边界处理**

```cuda
int block_items_end = (num_elements - start_offset > INT32_MAX)
                      ? INT32_MAX
                      : num_elements - start_offset;
```

**为什么需要？**
- CUB 的某些内部实现使用 `int32_t` 索引
- 对于超大数组（如视频模型的 128K 序列），避免溢出

#### **Step 5：BlockLoad 执行**

```cuda
BlockLoad(temp_storage).Load(input + start_offset, input_vec, block_items_end, 0);
```

**参数**：
- `input + start_offset`：起始地址
- `input_vec`：输出数组（每线程 8 个元素）
- `block_items_end`：有效元素数
- `0`：默认值（填充超出范围的元素）

**内部过程**：
1. 协作式加载到共享内存（避免 bank conflicts）
2. 从共享内存分发到各线程的寄存器
3. 自动向量化优化

#### **Step 6：位打包**

```cuda
ret = (input_vec[0] << 7) | (input_vec[1] << 6) | ... | input_vec[7];
```

将 8 个 `bool`（每个 1 字节）打包为 1 个 `uint8_t`：

```
Bool:    [T, F, T, F, T, F, T, F]
         ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
Bits:    [1, 0, 1, 0, 1, 0, 1, 0]
         ↓
uint8_t: 0b10101010 = 170
```

**BitOrder**：
- `kBig`：MSB first（最高位在前）
- `kLittle`：LSB first（最低位在前）

### 4.3 为什么使用 CUB？

**对比手动实现**：

```cuda
// ❌ 手动实现（可能有 bank conflicts）
__global__ void manual_load_kernel(bool* input, ...) {
    __shared__ bool smem[256 * 8];

    // 每个线程加载 8 个元素
    for (int i = 0; i < 8; ++i) {
        smem[threadIdx.x * 8 + i] = input[...];
    }
    __syncthreads();

    // 从共享内存读取
    bool input_vec[8];
    for (int i = 0; i < 8; ++i) {
        input_vec[i] = smem[threadIdx.x * 8 + i];
    }
}
```

**问题**：
- 手动管理共享内存布局
- 可能有 bank conflicts
- 代码复杂

**✅ 使用 CUB（自动优化）**：
```cuda
BlockLoad(temp_storage).Load(input + offset, input_vec, count, 0);
```

- 一行代码搞定
- 自动避免 bank conflicts
- 自动向量化

## 5. 实验：对比 CUB vs 手动实现

### 5.1 创建测试文件

创建 `test_cub_vs_manual.cu`：

```cuda
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <stdio.h>

// 手动实现
__global__ void manual_load_kernel(float* input, float* output, int n) {
    __shared__ float smem[256 * 4];
    int tid = threadIdx.x;
    int idx = blockIdx.x * 256 * 4 + tid;

    // 加载到共享内存
    for (int i = 0; i < 4; ++i) {
        if (idx + i * 256 < n) {
            smem[tid * 4 + i] = input[idx + i * 256];
        }
    }
    __syncthreads();

    // 处理并写回
    for (int i = 0; i < 4; ++i) {
        if (idx + i * 256 < n) {
            output[idx + i * 256] = smem[tid * 4 + i] * 2.0f;
        }
    }
}

// CUB 实现
__global__ void cub_load_kernel(float* input, float* output, int n) {
    typedef cub::BlockLoad<float, 256, 4, cub::BLOCK_LOAD_VECTORIZE> BlockLoad;
    __shared__ typename BlockLoad::TempStorage temp_storage;

    float thread_data[4];
    int block_offset = blockIdx.x * 256 * 4;

    BlockLoad(temp_storage).Load(input + block_offset, thread_data, n - block_offset, 0.0f);

    // 处理
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        thread_data[i] *= 2.0f;
    }

    // 写回（这里简化，实际应该用 BlockStore）
    int idx = block_offset + threadIdx.x;
    for (int i = 0; i < 4; ++i) {
        if (idx + i * 256 < n) {
            output[idx + i * 256] = thread_data[i];
        }
    }
}

int main() {
    const int n = 1 << 20;  // 1M elements
    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));

    int blocks = (n + 256 * 4 - 1) / (256 * 4);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Benchmark manual
    cudaEventRecord(start);
    for (int i = 0; i < 1000; ++i) {
        manual_load_kernel<<<blocks, 256>>>(d_in, d_out, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_manual;
    cudaEventElapsedTime(&time_manual, start, stop);

    // Benchmark CUB
    cudaEventRecord(start);
    for (int i = 0; i < 1000; ++i) {
        cub_load_kernel<<<blocks, 256>>>(d_in, d_out, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_cub;
    cudaEventElapsedTime(&time_cub, start, stop);

    printf("Manual: %.3f ms\n", time_manual / 1000);
    printf("CUB:    %.3f ms\n", time_cub / 1000);
    printf("Speedup: %.2fx\n", time_manual / time_cub);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
```

### 5.2 编译运行

```bash
nvcc -O3 -arch=sm_80 test_cub_vs_manual.cu -o test_cub
./test_cub
```

## 6. 共享内存的最佳实践

### 6.1 容量管理

查询设备的共享内存容量：

```cuda
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
```

**现代 GPU**：
- Volta/Turing/Ampere：48KB-96KB
- Hopper：100KB-164KB

### 6.2 L1 Cache 与 Shared Memory 的权衡

某些架构允许配置 L1/Shared 比例：

```cuda
// 优先共享内存（更大的 Shared Memory）
cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

// 优先 L1 Cache
cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
```

## 7. 今日作业

1. **代码阅读**：
   - 完整阅读 `include/flashinfer/quantization.cuh`
   - 理解 CUB BlockLoad 的使用

2. **实验**：
   - 运行 `test_cub_vs_manual.cu`
   - 使用 `nsight-compute` 分析 bank conflicts

3. **思考问题**：
   - 为什么共享内存有 32 个 banks？（提示：warp size）
   - Padding 如何解决 bank conflicts？
   - CUB 的 `BLOCK_LOAD_VECTORIZE` 内部是如何优化的？

4. **进阶挑战**：
   - 实现一个使用共享内存的矩阵转置 kernel
   - 对比有无 padding 的性能差异
   - 使用 `nsight-compute` 查看 bank conflicts 指标

## 8. 参考资料

- [CUDA C++ Programming Guide - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [CUB Documentation](https://nvlabs.github.io/cub/)
- [Bank Conflicts Explained](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
- FlashInfer 源码：`include/flashinfer/quantization.cuh`

## 下一天

[第 5 天：Warp 级编程](./cuda_course_day05.md)

---

**重点回顾**：
- ✅ 共享内存：芯片上的快速存储（20-30 cycles）
- ✅ Bank Conflicts：多线程访问同一 bank 不同地址时发生
- ✅ 避免方法：Padding、访问模式调整、Swizzle
- ✅ CUB 库：自动优化的 block-level 原语
- ✅ BlockLoad：高效加载，自动避免 conflicts
