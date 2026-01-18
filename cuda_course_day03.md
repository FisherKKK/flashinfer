# 第 3 天：PTX 汇编与底层指令

## 学习目标

- 理解 PTX（Parallel Thread Execution）汇编
- 学习内联汇编（inline assembly）的语法
- 掌握常用数学指令（exp2, log2, rsqrt, tanh）
- 理解 Warp shuffle 指令及其应用
- 分析 `math.cuh` 中的底层优化

## 1. PTX 汇编简介

### 1.1 什么是 PTX？

**PTX（Parallel Thread Execution）** 是 NVIDIA GPU 的中间表示（IR）：

```
CUDA C++ 代码
     ↓ (nvcc 编译)
PTX 汇编代码
     ↓ (GPU 驱动编译)
SASS 机器码
     ↓ (GPU 执行)
```

**特点**：
- **虚拟 ISA**：不直接对应硬件，由驱动编译为实际机器码（SASS）
- **可读性**：比机器码易读，但比 C++ 底层
- **前向兼容**：旧 PTX 可在新 GPU 上运行
- **手动优化**：直接控制硬件功能

### 1.2 为什么使用 PTX？

**使用场景**：
1. 访问 CUDA C++ 未暴露的硬件功能
2. 精确控制指令选择和优化
3. 实现超低延迟操作（如 shuffle）
4. 使用特殊指令（如 `tanh.approx`）

## 2. 内联汇编基础

### 2.1 基本语法

CUDA 支持在 C++ 代码中嵌入 PTX 汇编：

```cuda
asm volatile("ptx_instruction;" : outputs : inputs : clobbers);
```

**组成部分**：
- `asm volatile`：内联汇编关键字（`volatile` 防止编译器优化掉）
- `"ptx_instruction;"`：PTX 指令字符串
- `outputs`：输出操作数列表
- `inputs`：输入操作数列表
- `clobbers`：被修改的寄存器（可选）

### 2.2 操作数约束

**约束字符**指定变量与寄存器的对应关系：

| 约束 | 含义 | 示例 |
|------|------|------|
| `"=f"` | 输出，单精度浮点寄存器 | `float` |
| `"f"` | 输入，单精度浮点寄存器 | `float` |
| `"=r"` | 输出，32位整型寄存器 | `uint32_t`, `int` |
| `"r"` | 输入，32位整型寄存器 | `uint32_t`, `int` |
| `"=h"` | 输出，16位整型寄存器 | `uint16_t`, `half` (as ushort) |
| `"h"` | 输入，16位整型寄存器 | `uint16_t`, `half` (as ushort) |
| `"=l"` | 输出，64位整型寄存器 | `uint64_t`, `long long` |
| `"l"` | 输入，64位整型寄存器 | `uint64_t`, `long long` |
| `"=d"` | 输出，双精度浮点寄存器 | `double` |
| `"d"` | 输入，双精度浮点寄存器 | `double` |

**`=` 前缀**：表示输出（write-only）
**无 `=`**：表示输入（read-only）

### 2.3 示例：平方根倒数

```cuda
__forceinline__ __device__ float rsqrt(float x) {
  float y;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;"
               : "=f"(y)      // 输出：y，单精度浮点
               : "f"(x));     // 输入：x，单精度浮点
  return y;
}
```

**PTX 指令分解**：
```
rsqrt.approx.ftz.f32 %0, %1
  ^      ^     ^    ^   ^  ^
  |      |     |    |   |  输入寄存器1（x）
  |      |     |    |   输出寄存器0（y）
  |      |     |    数据类型（32位浮点）
  |      |     Flush-To-Zero（小数变0）
  |      近似计算（更快但精度略低）
  指令名（reciprocal square root）
```

## 3. 常用数学指令

### 3.1 指数函数：`exp2.approx`

计算 $2^x$。

#### **单精度版本**

```cuda
__forceinline__ __device__ float ptx_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
```

**为什么不用 `exp()`？**
- `exp(x)` = $e^x$ 在 GPU 上需要多条指令
- `exp2(x)` = $2^x$ 是单条硬件指令，更快
- 转换：$e^x = 2^{x \cdot \log_2(e)}$

#### **Half2 版本（SIMD）**

```cuda
__forceinline__ __device__ half2 ptx_exp2(half2 x) {
  uint32_t y_u32;
  uint32_t x_u32 = half2_as_uint32(x);  // 类型双关（type punning）
  asm volatile("ex2.approx.f16x2 %0, %1;" : "=r"(y_u32) : "r"(x_u32));
  return uint32_as_half2(y_u32);
}
```

**关键点**：
- `half2`：打包了两个 `half`（fp16）值
- `f16x2`：SIMD 指令，同时计算两个值
- 类型双关：将 `half2` 当作 `uint32_t` 传递（因为 PTX 约束限制）

### 3.2 对数函数：`lg2.approx`

计算 $\log_2(x)$。

```cuda
__forceinline__ __device__ float ptx_log2(float x) {
  float y;
  asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
```

**应用**：
- Softmax 中的 log-sum-exp 技巧
- Entropy 计算
- 数值稳定性优化

### 3.3 平方根倒数：`rsqrt.approx`

计算 $\frac{1}{\sqrt{x}}$。

```cuda
__forceinline__ __device__ float rsqrt(float x) {
  float y;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
```

**用途**：
- RMSNorm：$\frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}}$
- 向量归一化：$\frac{\vec{v}}{||\vec{v}||}$

**为什么不用 `1.0f / sqrt(x)`？**
- `rsqrt` 是单条指令
- `sqrt` + `rcp`（倒数）需要两条指令

### 3.4 双曲正切：`tanh.approx`

计算 $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$。

```cuda
__forceinline__ __device__ float tanh(float x) {
  float y;
  asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
```

**应用**：
- GELU 激活函数的近似
- 老式 RNN/LSTM 的激活

## 4. Warp Shuffle 指令

### 4.1 什么是 Shuffle？

**Shuffle** 允许 warp 内的线程直接交换寄存器值，**无需共享内存**。

**优势**：
- ⚡ **极快**：寄存器到寄存器，1-2 cycles
- 💾 **节省内存**：不使用共享内存
- 🔄 **灵活**：支持多种模式（butterfly, up, down, indexed）

### 4.2 Butterfly Shuffle

**模式**：`y[i] = x[i XOR lane_mask]`

```cuda
__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) {
  float y;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
               : "=f"(y)
               : "f"(x), "r"(lane_mask));
  return y;
}
```

**参数解释**：
- `%0`：输出 `y`
- `%1`：输入 `x`（要交换的值）
- `%2`：`lane_mask`（XOR 掩码）
- `0x1f`：warp 大小 - 1（31，表示 32 线程）
- `0xffffffff`：参与的线程掩码（全部线程）

### 4.3 Butterfly Shuffle 图解

假设 `lane_mask = 1`（offset = 1）：

```
线程 ID: 0  1  2  3  4  5  6  7
原始值:  a0 a1 a2 a3 a4 a5 a6 a7
          ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
交换后:  a1 a0 a3 a2 a5 a4 a7 a6
         (0^1=1) (1^1=0) (2^1=3) ...
```

假设 `lane_mask = 2`（offset = 2）：

```
线程 ID: 0  1  2  3  4  5  6  7
原始值:  a0 a1 a2 a3 a4 a5 a6 a7
          ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
交换后:  a2 a3 a0 a1 a6 a7 a4 a5
         (0^2=2) (1^2=3) (2^2=0) ...
```

### 4.4 应用：Warp Reduction

使用 shuffle 实现 warp 内求和（无需共享内存）：

```cuda
__device__ float warp_reduce_sum(float val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_xor_sync(0xffffffff, val, offset);
  }
  return val;  // 所有线程都得到总和
}
```

**迭代过程**（warp size = 32）：

```
初始: [a0, a1, a2, a3, ..., a31]

offset=16: 线程 i 接收线程 i^16 的值
  [a0+a16, a1+a17, ..., a15+a31, a16+a0, ...]

offset=8: 线程 i 接收线程 i^8 的值
  [(a0+a16)+(a8+a24), ...]

offset=4, 2, 1: 类似迭代

最终: 所有线程都有 sum(a0...a31)
```

**对比共享内存方法**：
- Shuffle：5 次迭代，纯寄存器操作
- Shared memory：需要 `__syncthreads()`，内存读写

## 5. 类型双关（Type Punning）

### 5.1 为什么需要？

PTX 汇编的约束有限（如没有 `half2` 的约束符），需要通过 `uint32_t` 传递。

### 5.2 安全的类型双关

```cuda
__forceinline__ __device__ half2 uint32_as_half2(uint32_t x) {
  return *(half2*)&x;  // 将 uint32_t 的地址重新解释为 half2*
}

__forceinline__ __device__ uint32_t half2_as_uint32(half2 x) {
  return *(uint32_t*)&x;
}
```

**注意事项**：
- ⚠️ 必须保证两种类型大小相同（`sizeof(half2) == sizeof(uint32_t) == 4`）
- ⚠️ 只用于寄存器传递，不要用于语义转换

### 5.3 示例：Half2 的 tanh

```cuda
__forceinline__ __device__ half2 tanh(half2 x) {
  uint32_t y_u32;
  uint32_t x_u32 = half2_as_uint32(x);  // half2 → uint32_t
  asm volatile("tanh.approx.f16x2 %0, %1;"
               : "=r"(y_u32)  // 输出 uint32_t
               : "r"(x_u32)); // 输入 uint32_t
  return uint32_as_half2(y_u32);  // uint32_t → half2
}
```

## 6. 案例分析：如何使用 `math.cuh`

### 6.1 实现高效的 Softmax

Softmax 公式：$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$

**朴素实现**（慢）：
```cuda
for (int i = 0; i < n; ++i) {
  sum += exp(x[i]);  // exp() 很慢
}
for (int i = 0; i < n; ++i) {
  y[i] = exp(x[i]) / sum;
}
```

**优化实现**（使用 `ptx_exp2`）：

```cuda
// exp(x) = 2^(x * log2(e))
for (int i = 0; i < n; ++i) {
  float val = x[i] * math::log2e;  // log2e = 1.44269...
  sum += math::ptx_exp2(val);       // 单条指令！
}
for (int i = 0; i < n; ++i) {
  float val = x[i] * math::log2e;
  y[i] = math::ptx_exp2(val) / sum;
}
```

**加速原因**：
- `exp(x)` 可能编译为多条指令
- `ex2.approx.f32` 是单条硬件指令

### 6.2 RMSNorm 中的 rsqrt

RMSNorm：$y = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \cdot w$

```cuda
// 计算平方和
float sum_sq = 0.f;
for (int i = 0; i < d; ++i) {
  sum_sq += x[i] * x[i];
}

// 使用 rsqrt 计算倒数
float scale = math::rsqrt(sum_sq / d + epsilon);

// 应用
for (int i = 0; i < d; ++i) {
  y[i] = x[i] * scale * weight[i];
}
```

**对比**：
- `math::rsqrt(x)`：1 条指令
- `1.0f / sqrt(x)`：2 条指令

## 7. 实验：对比 PTX 指令与标准库

### 7.1 创建测试文件

创建 `test_ptx_performance.cu`：

```cuda
#include <cuda_runtime.h>
#include <stdio.h>

// PTX 版本
__device__ float rsqrt_ptx(float x) {
  float y;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// 标准库版本
__device__ float rsqrt_std(float x) {
  return 1.0f / sqrtf(x);
}

__global__ void benchmark_ptx(float* out, const float* in, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = rsqrt_ptx(in[idx]);
  }
}

__global__ void benchmark_std(float* out, const float* in, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = rsqrt_std(in[idx]);
  }
}

int main() {
  const int n = 1 << 20;  // 1M 元素
  float *d_in, *d_out;
  cudaMalloc(&d_in, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));

  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  // Warmup
  benchmark_ptx<<<blocks, threads>>>(d_out, d_in, n);
  benchmark_std<<<blocks, threads>>>(d_out, d_in, n);

  // Benchmark PTX
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < 1000; ++i) {
    benchmark_ptx<<<blocks, threads>>>(d_out, d_in, n);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float time_ptx;
  cudaEventElapsedTime(&time_ptx, start, stop);

  // Benchmark STD
  cudaEventRecord(start);
  for (int i = 0; i < 1000; ++i) {
    benchmark_std<<<blocks, threads>>>(d_out, d_in, n);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float time_std;
  cudaEventElapsedTime(&time_std, start, stop);

  printf("PTX rsqrt: %.3f ms\n", time_ptx / 1000);
  printf("STD rsqrt: %.3f ms\n", time_std / 1000);
  printf("Speedup: %.2fx\n", time_std / time_ptx);

  cudaFree(d_in);
  cudaFree(d_out);
  return 0;
}
```

### 7.2 编译运行

```bash
nvcc -O3 -arch=sm_80 test_ptx_performance.cu -o test_ptx
./test_ptx
```

预期输出（示例）：
```
PTX rsqrt: 0.012 ms
STD rsqrt: 0.018 ms
Speedup: 1.50x
```

## 8. 今日作业

1. **代码阅读**：
   - 完整阅读 `include/flashinfer/math.cuh`
   - 理解每个 PTX 指令的作用

2. **实验**：
   - 运行 `test_ptx_performance.cu`
   - 修改为测试 `exp2` vs `expf`
   - 测试 `tanh.approx` vs 标准 `tanhf`

3. **思考问题**：
   - `approx` 指令的精度损失有多大？（查阅 PTX ISA 文档）
   - `ftz`（Flush-To-Zero）是什么意思？何时需要？
   - 为什么 shuffle 比共享内存快？

4. **进阶挑战**：
   - 使用 `nsight-compute` 查看生成的 PTX 和 SASS 代码
   - 实现一个使用 shuffle 的 warp reduction
   - 对比 shuffle reduction 与共享内存 reduction 的性能

## 9. 参考资料

- [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA C++ Programming Guide - PTX Assembly](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asm)
- [Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions)
- FlashInfer 源码：`include/flashinfer/math.cuh`

## 下一天

[第 4 天：共享内存基础](./cuda_course_day04.md)

---

**重点回顾**：
- ✅ PTX 汇编：GPU 的中间表示，可手动优化
- ✅ 内联汇编语法：`asm volatile("指令;" : outputs : inputs)`
- ✅ 数学指令：exp2, log2, rsqrt, tanh（单指令，快速）
- ✅ Warp shuffle：寄存器间直接通信，无需共享内存
- ✅ 类型双关：通过 `uint32_t` 传递 `half2` 等特殊类型
