# 第 5 天：Warp 级编程

## 学习目标

- 深入理解 Warp 作为 GPU 的基本执行单元
- 掌握 Warp-level reduction 的实现
- 学习 Warp shuffle 的高级应用
- 分析 `norm.cuh` 中的 Warp reduction 代码
- 理解 Warp 同步与线程发散

## 1. Warp 架构深入

### 1.1 什么是 Warp？

**Warp** 是 GPU 的基本调度和执行单元：

| 属性 | 值 |
|------|------|
| **Warp 大小** | 32 个线程（固定） |
| **调度方式** | SIMT（Single Instruction, Multiple Threads） |
| **执行模式** | Lock-step（所有线程执行相同指令） |
| **寄存器** | 每个线程独立的寄存器 |
| **同步** | Warp 内隐式同步 |

### 1.2 SIMT 执行模型

```
时钟周期 1: 所有 32 个线程执行: ADD
时钟周期 2: 所有 32 个线程执行: MUL
时钟周期 3: 所有 32 个线程执行: STORE
```

**关键特性**：
- Warp 内的线程自动同步（无需 `__syncthreads()`）
- 所有线程执行相同的指令路径
- 不同 warp 之间独立执行

### 1.3 线程发散（Thread Divergence）

当 warp 内线程走不同分支时，发生**线程发散**：

```cuda
__global__ void divergent_kernel(int* data) {
    int idx = threadIdx.x;

    if (idx % 2 == 0) {
        // 偶数线程：分支 A
        data[idx] = idx * 2;
    } else {
        // 奇数线程：分支 B
        data[idx] = idx * 3;
    }
}
```

**执行过程**：
```
步骤 1: 执行分支 A（偶数线程活跃，奇数线程等待）
步骤 2: 执行分支 B（奇数线程活跃，偶数线程等待）
```

**性能影响**：
- 执行时间 = 最长分支的时间
- 尽量避免或最小化线程发散

### 1.4 Warp 内的隐式同步

```cuda
__device__ float warp_sum(float val) {
    // 不需要 __syncthreads()！
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;  // 所有线程自动同步
}
```

**为什么不需要同步？**
- Warp 内所有线程 lock-step 执行
- Shuffle 指令在同一时钟周期完成

## 2. Warp Shuffle 指令详解

### 2.1 Shuffle 指令家族

CUDA 提供多种 shuffle 模式：

| 函数 | 功能 | 模式 |
|------|------|------|
| `__shfl_sync` | 直接交换 | `dst_lane = src_lane` |
| `__shfl_up_sync` | 向上移动 | `dst = src - delta` |
| `__shfl_down_sync` | 向下移动 | `dst = src + delta` |
| `__shfl_xor_sync` | 异或交换 | `dst = src XOR mask` |

### 2.2 `__shfl_xor_sync` 详解

**函数签名**：
```cuda
T __shfl_xor_sync(unsigned mask, T var, int laneMask);
```

**参数**：
- `mask`：参与的线程掩码（通常 `0xffffffff` 全员参与）
- `var`：要交换的值
- `laneMask`：XOR 掩码

**工作原理**：
```
线程 i 接收 线程 (i XOR laneMask) 的值
```

#### 示例：laneMask = 1

```
Lane ID:  0   1   2   3   4   5   6   7
原始值:   a0  a1  a2  a3  a4  a5  a6  a7
          ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
XOR 1:    a1  a0  a3  a2  a5  a4  a7  a6
         (0^1)(1^1)(2^1)(3^1)...
```

#### 示例：laneMask = 16

```
Lane ID:  0   1   2  ...  15  16  17  ...  31
原始值:   a0  a1  a2 ... a15 a16 a17 ... a31
          ↓   ↓   ↓       ↓   ↓   ↓       ↓
XOR 16:  a16 a17 a18    a31  a0  a1     a15
        (0^16=16)      (15^16=31) (16^16=0)
```

### 2.3 其他 Shuffle 指令

#### `__shfl_down_sync`

```cuda
float val = __shfl_down_sync(0xffffffff, value, 1);
// 线程 i 接收线程 i+1 的值
// 线程 31 接收线程 31 的值（边界处理）
```

```
Lane:  0   1   2   3   ...  30  31
原始:  a0  a1  a2  a3  ...  a30 a31
Down1: a1  a2  a3  a4  ...  a31 a31
```

#### `__shfl_up_sync`

```cuda
float val = __shfl_up_sync(0xffffffff, value, 1);
// 线程 i 接收线程 i-1 的值
// 线程 0 接收线程 0 的值
```

## 3. Warp Reduction 实现

### 3.1 基本 Warp Reduction

使用 `__shfl_xor_sync` 实现 warp 内求和：

```cuda
__device__ float warp_reduce_sum(float val) {
    constexpr uint32_t warp_size = 32;

    // 二分归约
    #pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }

    return val;  // 所有线程都有相同的总和
}
```

**迭代过程**（以 8 个线程为例）：

```
初始值（8个线程）:
Lane:  0   1   2   3   4   5   6   7
Val:   a0  a1  a2  a3  a4  a5  a6  a7

Offset=4 (XOR with lane 4):
Lane:  0       1       2       3       4       5       6       7
Val:   a0+a4   a1+a5   a2+a6   a3+a7   a4+a0   a5+a1   a6+a2   a7+a3

Offset=2 (XOR with lane 2):
Lane:  0               1               2               3
Val:   a0+a4+a2+a6     a1+a5+a3+a7     a2+a6+a0+a4     a3+a7+a1+a5
       ...

Offset=1 (XOR with lane 1):
Lane:  0
Val:   sum(a0...a7)  // 所有线程都有总和
```

### 3.2 为什么使用 XOR？

**优势**：
- **对称性**：`i XOR mask == j` 则 `j XOR mask == i`
- **所有线程参与**：每次迭代都有数据交换
- **高效**：固定的迭代次数（log₂(warp_size) = 5）

### 3.3 其他 Reduction 操作

#### Warp Max

```cuda
__device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}
```

#### Warp Min

```cuda
__device__ float warp_reduce_min(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}
```

## 4. 案例分析：`norm.cuh` 中的 Warp Reduction

### 4.1 相关代码

文件位置：`include/flashinfer/norm.cuh` (行 68-72)

```cuda
// Step 1: 计算平方和（每个线程的局部和）
float sum_sq = 0.f;
for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, VEC_SIZE> input_vec;
    input_vec.load(input + ...);
    #pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
        sum_sq += float(input_vec[j]) * float(input_vec[j]);
    }
}

// Step 2: Warp 内 reduction
#pragma unroll
for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq += math::shfl_xor_sync(sum_sq, offset);
}
```

### 4.2 完整上下文

```cuda
template <uint32_t VEC_SIZE, typename T>
__global__ void RMSNormKernel(...) {
    const uint32_t tx = threadIdx.x, ty = threadIdx.y;
    constexpr uint32_t warp_size = 32;
    const uint32_t num_warps = blockDim.y;

    float sum_sq = 0.f;

    // 每个线程计算部分数据的平方和
    for (uint32_t i = 0; i < rounds; i++) {
        vec_t<T, VEC_SIZE> input_vec;
        input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
        #pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; j++) {
            sum_sq += float(input_vec[j]) * float(input_vec[j]);
        }
    }

    // Warp-level reduction：每个 warp 内求和
    #pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
        sum_sq += math::shfl_xor_sync(sum_sq, offset);
    }

    // 现在每个 warp 的第 0 个线程有该 warp 的总和
    // （其他线程也有，但我们只用第 0 个）
}
```

### 4.3 为什么这样设计？

**Warp reduction 的优势**：
1. **无需共享内存**：纯寄存器操作
2. **无需同步**：Warp 内隐式同步
3. **延迟低**：1-2 个时钟周期每次 shuffle
4. **代码简洁**：几行代码完成

**对比共享内存版本**：
```cuda
// ❌ 使用共享内存（更慢）
__shared__ float smem[32];
smem[threadIdx.x] = sum_sq;
__syncthreads();

if (threadIdx.x < 16) smem[threadIdx.x] += smem[threadIdx.x + 16];
__syncthreads();
if (threadIdx.x < 8) smem[threadIdx.x] += smem[threadIdx.x + 8];
__syncthreads();
// ... 繁琐且需要多次同步
```

## 5. 多 Warp 场景

### 5.1 问题

一个 block 通常有多个 warp（如 256 threads = 8 warps）。

如何在多个 warp 之间做 reduction？

### 5.2 两级 Reduction 模式

```
Level 1: Warp 内 reduction（shuffle）
         ↓
     每个 warp 得到一个部分和
         ↓
Level 2: Warp 间 reduction（共享内存）
         ↓
     最终的全局和
```

**代码框架**：

```cuda
__global__ void multi_warp_reduction_kernel(...) {
    const uint32_t lane_id = threadIdx.x % 32;  // 在 warp 中的位置
    const uint32_t warp_id = threadIdx.x / 32;  // warp 编号

    float val = ...;  // 每个线程的值

    // Level 1: Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    // 现在 val 在每个 warp 的第 0 个线程中是该 warp 的和

    // Level 2: 跨 warp reduction（使用共享内存）
    __shared__ float warp_sums[num_warps];

    if (lane_id == 0) {
        warp_sums[warp_id] = val;  // 每个 warp 的代表存储结果
    }
    __syncthreads();

    // 由第一个 warp 完成最终 reduction
    if (warp_id == 0) {
        val = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_xor_sync(0xffffffff, val, offset);
        }
        // 现在 val 是全局和（在第一个 warp 中）
    }
}
```

**我们将在第 6 天详细学习这个模式！**

## 6. 实验：Warp Reduction 性能

### 6.1 创建测试文件

创建 `test_warp_reduction.cu`：

```cuda
#include <cuda_runtime.h>
#include <stdio.h>

// Warp reduction using shuffle
__device__ float warp_reduce_sum_shuffle(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp reduction using shared memory (for comparison)
__device__ float warp_reduce_sum_smem(float val, int lane) {
    __shared__ float smem[32];
    smem[lane] = val;
    __syncthreads();

    if (lane < 16) smem[lane] += smem[lane + 16]; __syncthreads();
    if (lane < 8)  smem[lane] += smem[lane + 8];  __syncthreads();
    if (lane < 4)  smem[lane] += smem[lane + 4];  __syncthreads();
    if (lane < 2)  smem[lane] += smem[lane + 2];  __syncthreads();
    if (lane < 1)  smem[lane] += smem[lane + 1];  __syncthreads();

    return smem[0];
}

__global__ void test_shuffle_reduction(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? input[idx] : 0.0f;

    val = warp_reduce_sum_shuffle(val);

    if (threadIdx.x % 32 == 0) {
        atomicAdd(output, val);
    }
}

__global__ void test_smem_reduction(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? input[idx] : 0.0f;

    val = warp_reduce_sum_smem(val, threadIdx.x % 32);

    if (threadIdx.x % 32 == 0) {
        atomicAdd(output, val);
    }
}

int main() {
    const int n = 1 << 20;  // 1M elements
    float *d_input, *d_output;

    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Benchmark shuffle
    cudaEventRecord(start);
    for (int i = 0; i < 1000; ++i) {
        cudaMemset(d_output, 0, sizeof(float));
        test_shuffle_reduction<<<blocks, threads>>>(d_input, d_output, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_shuffle;
    cudaEventElapsedTime(&time_shuffle, start, stop);

    // Benchmark shared memory
    cudaEventRecord(start);
    for (int i = 0; i < 1000; ++i) {
        cudaMemset(d_output, 0, sizeof(float));
        test_smem_reduction<<<blocks, threads>>>(d_input, d_output, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_smem;
    cudaEventElapsedTime(&time_smem, start, stop);

    printf("Shuffle: %.3f ms\n", time_shuffle / 1000);
    printf("Shared:  %.3f ms\n", time_smem / 1000);
    printf("Speedup: %.2fx\n", time_smem / time_shuffle);

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
```

### 6.2 编译运行

```bash
nvcc -O3 -arch=sm_80 test_warp_reduction.cu -o test_warp
./test_warp
```

预期结果：Shuffle 版本应该快 1.5-2 倍。

## 7. 今日作业

1. **代码阅读**：
   - 阅读 `include/flashinfer/norm.cuh` 的 warp reduction 部分
   - 理解为什么使用 `shfl_xor_sync`

2. **实验**：
   - 运行 `test_warp_reduction.cu`
   - 使用 `nsight-compute` 分析性能差异

3. **思考问题**：
   - 为什么 warp reduction 不需要 `__syncthreads()`？
   - 如果 warp size 不是 32 会怎样？
   - `#pragma unroll` 对性能有什么影响？

4. **进阶挑战**：
   - 实现 warp-level max reduction
   - 实现 warp-level argmax（找最大值的索引）
   - 对比 `__shfl_xor_sync` 和 `__shfl_down_sync` 的性能

## 8. 参考资料

- [CUDA C++ Programming Guide - Warp Shuffle](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions)
- [Using Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
- FlashInfer 源码：`include/flashinfer/norm.cuh`

## 下一天

[第 6 天：多级 Reduction](./cuda_course_day06.md)

---

**重点回顾**：
- ✅ Warp：32 个线程的基本执行单元
- ✅ SIMT 模型：所有线程执行相同指令
- ✅ Warp shuffle：寄存器间直接通信
- ✅ Warp reduction：高效的 warp 内聚合
- ✅ `__shfl_xor_sync`：对称的交换模式
