# 第 2 天：内存合并与向量化访问

## 学习目标

- 理解 GPU 内存合并（Memory Coalescing）的重要性
- 掌握向量化加载/存储技术
- 学习 FlashInfer 的 `vec_t` 向量类型
- 分析 `activation.cuh` 中的性能优化技巧

## 1. GPU 内存访问基础

### 1.1 为什么内存访问是性能瓶颈？

在 GPU 上，**内存带宽**通常是限制性能的主要因素：

| 操作 | 延迟 |
|------|------|
| 寄存器访问 | ~1 cycle |
| 共享内存访问 | ~20-30 cycles |
| L1 缓存命中 | ~30-50 cycles |
| L2 缓存命中 | ~200 cycles |
| 全局内存访问 | **300-600 cycles** |

**结论**：一次全局内存访问的时间，CPU 可以执行数百条算术指令！

### 1.2 内存合并（Memory Coalescing）

GPU 以**固定大小的事务**从全局内存读取数据：
- 32、64、或 128 字节为一个事务
- 事务必须对齐到其大小的倍数

**合并访问**（Coalesced Access）：
```
Warp 中的 32 个线程访问连续的内存地址
→ GPU 可以用 1-2 次内存事务完成
→ 高效！
```

**非合并访问**（Uncoalesced Access）：
```
Warp 中的线程访问随机/分散的地址
→ GPU 需要多次内存事务
→ 浪费带宽，效率低！
```

#### 示例：合并 vs 非合并

**✅ 合并访问**（推荐）：
```cuda
__global__ void coalesced_kernel(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = data[idx];  // 每个线程访问连续地址
}
```

**❌ 非合并访问**（避免）：
```cuda
__global__ void uncoalesced_kernel(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = data[idx * 137];  // 访问地址不连续，步长大
}
```

### 1.3 对齐（Alignment）

内存地址应该对齐到其大小的倍数：

| 类型 | 大小 | 对齐要求 |
|------|------|----------|
| `float` | 4 字节 | 4 字节对齐 |
| `float2` | 8 字节 | 8 字节对齐 |
| `float4` | 16 字节 | 16 字节对齐 |
| `int4` | 16 字节 | 16 字节对齐 |

**未对齐的访问**会触发多次内存事务，降低效率。

## 2. 向量化访问

### 2.1 为什么要向量化？

单个 `float` 加载：
```cuda
float x = data[idx];  // 加载 4 字节
```

使用 `float4` 向量化加载：
```cuda
float4 x = *((float4*)(data + idx));  // 一次加载 16 字节
```

**优势**：
- 一条指令加载 4 倍数据
- 更好地利用内存带宽
- 减少指令数量

### 2.2 CUDA 内置向量类型

CUDA 提供了内置向量类型：

```cuda
// 整型向量
int2, int3, int4
uint2, uint3, uint4

// 浮点向量
float2, float3, float4
double2, double3, double4

// Half 精度向量
half2  // 2 个 fp16
```

**访问成员**：
```cuda
float4 v;
v.x, v.y, v.z, v.w  // 分量访问
```

### 2.3 FlashInfer 的 `vec_t` 模板

FlashInfer 定义了统一的向量类型模板 `vec_t<T, N>`，支持自动类型转换。

#### 基本用法

```cuda
// 定义：8 个 float 的向量
vec_t<float, 8> x_vec;

// 加载：从 half* 指针加载，自动转换为 float
half* input_ptr = ...;
x_vec.cast_load(input_ptr);  // half → float 转换

// 访问元素
x_vec[0], x_vec[1], ..., x_vec[7]

// 存储：存储到 half* 指针，自动转换
half* output_ptr = ...;
x_vec.cast_store(output_ptr);  // float → half 转换
```

#### `vec_t` 的优势

1. **类型安全**：编译时检查
2. **自动转换**：`cast_load/cast_store` 处理精度转换
3. **向量化**：一次加载/存储多个元素
4. **灵活性**：支持各种数据类型（fp32, fp16, bf16, fp8）

## 3. 案例分析：`activation.cuh`

现在让我们详细分析 FlashInfer 的激活函数 kernel。

### 3.1 完整代码

文件位置：`include/flashinfer/activation.cuh`

```cuda
template <typename T, float (*Activation)(const float&)>
__global__ void act_and_mul_kernel(T* __restrict__ out, const T* __restrict__ input, const int d) {
  // 1. 确定向量大小：每次处理 16 字节
  constexpr uint32_t vec_size = 16 / sizeof(T);

  // 2. 获取线程索引
  const int64_t token_idx = blockIdx.x;       // 每个 block 处理一个 token
  const int64_t thread_idx = threadIdx.x;     // block 内的线程索引
  const int64_t stride = blockDim.x;          // block 的线程数
  const int64_t offset = token_idx * 2 * d;   // 当前 token 的起始位置

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");  // Hopper 架构的 grid 依赖控制
#endif

  // 3. 向量化循环：处理对齐部分
#pragma unroll 1  // 不展开循环，减少代码大小
  for (uint32_t idx = thread_idx; idx < d / vec_size; idx += stride) {
    vec_t<float, vec_size> x_vec, y_vec, out_vec;

    // 加载 x 和 y（自动类型转换）
    x_vec.cast_load(input + offset + idx * vec_size);
    y_vec.cast_load(input + offset + d + idx * vec_size);

    // 逐元素计算：out = activation(x) * y
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      out_vec[i] = Activation(x_vec[i]) * y_vec[i];
    }

    // 存储结果
    out_vec.cast_store(out + token_idx * d + idx * vec_size);
  }

  // 4. 标量循环：处理剩余元素
  const int64_t remaining_offset = d - d % (stride * vec_size);
#pragma unroll 1
  for (int64_t idx = thread_idx; idx < d % (stride * vec_size); idx += stride) {
    float x = input[offset + remaining_offset + idx],
          y = input[offset + remaining_offset + d + idx];
    out[token_idx * d + remaining_offset + idx] = Activation(x) * y;
  }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}
```

### 3.2 代码分解讲解

#### **Step 1：计算向量大小**

```cuda
constexpr uint32_t vec_size = 16 / sizeof(T);
```

- 目标：每次加载 **16 字节**（128 位）
- `sizeof(T)` 是元素大小
- 例如：
  - `T = half` (2字节) → `vec_size = 8`（一次处理 8 个 half）
  - `T = float` (4字节) → `vec_size = 4`（一次处理 4 个 float）

#### **Step 2：Grid 组织**

```cuda
const int64_t token_idx = blockIdx.x;
const int64_t thread_idx = threadIdx.x;
const int64_t stride = blockDim.x;
const int64_t offset = token_idx * 2 * d;
```

**Grid 设计**：
- 一个 block 处理一个 token（序列中的一个位置）
- Block 内有多个线程并行处理 `d` 维度

**为什么是 `2 * d`？**
- 激活函数需要两个输入：`x` 和 `y`
- 输入布局：`[x0, x1, ..., x_{d-1}, y0, y1, ..., y_{d-1}]`
- 输出：`[activation(x0)*y0, activation(x1)*y1, ...]`

#### **Step 3：向量化主循环**

```cuda
for (uint32_t idx = thread_idx; idx < d / vec_size; idx += stride) {
    vec_t<float, vec_size> x_vec, y_vec, out_vec;
    x_vec.cast_load(input + offset + idx * vec_size);
    y_vec.cast_load(input + offset + d + idx * vec_size);
    // ...
}
```

**循环模式**：**Grid-Stride Loop**
- 起始：`idx = thread_idx`（每个线程从自己的索引开始）
- 步长：`idx += stride`（跳过其他线程）
- 好处：自适应线程数，无论 `d` 多大都能处理

**内存访问**：
- `x_vec.cast_load(input + offset + idx * vec_size)`：
  - 加载 `vec_size` 个元素（合并访问）
  - 自动从 `T` 类型转换为 `float`

**示例**（假设 `vec_size=4`, `blockDim.x=256`）：
```
Thread 0: 处理 idx=0, 256, 512, ...
Thread 1: 处理 idx=1, 257, 513, ...
Thread 2: 处理 idx=2, 258, 514, ...
...
```

#### **Step 4：逐元素计算**

```cuda
#pragma unroll
for (uint32_t i = 0; i < vec_size; ++i) {
  out_vec[i] = Activation(x_vec[i]) * y_vec[i];
}
```

- `#pragma unroll`：**展开循环**
- 编译器会生成 `vec_size` 份代码，无循环开销
- `Activation` 是模板参数（SiLU、GELU 等）

#### **Step 5：处理剩余元素**

```cuda
const int64_t remaining_offset = d - d % (stride * vec_size);
for (int64_t idx = thread_idx; idx < d % (stride * vec_size); idx += stride) {
    float x = input[offset + remaining_offset + idx],
          y = input[offset + remaining_offset + d + idx];
    out[token_idx * d + remaining_offset + idx] = Activation(x) * y;
}
```

**为什么需要？**
- `d` 可能不是 `stride * vec_size` 的倍数
- 剩余的元素用标量方式处理

**示例**：
- `d = 1000`, `vec_size = 4`, `stride = 256`
- 向量化处理：`1000 / 4 = 250` 个向量（1000 个元素）
- 剩余：`0` 个元素（恰好整除）

但如果 `d = 1003`：
- 向量化处理：`1000` 个元素
- 剩余：`3` 个元素（由标量循环处理）

### 3.3 关键优化技巧总结

| 技巧 | 实现 | 效果 |
|------|------|------|
| **向量化加载** | `vec_t<float, vec_size>` + `cast_load` | 一次加载 16 字节 |
| **内存合并** | 连续地址访问 | 高效利用带宽 |
| **Grid-Stride Loop** | `idx += stride` | 自适应任意大小 |
| **循环展开** | `#pragma unroll` | 减少循环开销 |
| **处理边界** | 剩余元素标量处理 | 正确性保证 |
| **__restrict__** | 指针修饰符 | 告诉编译器指针不别名 |

## 4. 实验：观察向量化的性能提升

### 4.1 创建测试文件

创建 `test_activation_performance.py`：

```python
import torch
import flashinfer
import time

def benchmark(func, *args, num_iters=100):
    # Warmup
    for _ in range(10):
        func(*args)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_iters):
        func(*args)
    torch.cuda.synchronize()
    end = time.time()

    return (end - start) / num_iters * 1000  # ms

# 测试不同大小
batch_size = 32
for d in [512, 1024, 2048, 4096]:
    input_tensor = torch.randn(batch_size, 2 * d, device='cuda', dtype=torch.float16)
    output_tensor = torch.empty(batch_size, d, device='cuda', dtype=torch.float16)

    latency = benchmark(
        flashinfer.silu_and_mul,
        output_tensor, input_tensor
    )

    bandwidth = (batch_size * 2 * d * 2 * 2) / (latency / 1000) / 1e9  # GB/s
    # factor 1: input (2*d) + output (d) = 3*d elements
    # factor 2: fp16 = 2 bytes

    print(f"d={d:5d}: {latency:.3f} ms, bandwidth: {bandwidth:.1f} GB/s")
```

### 4.2 运行测试

```bash
python test_activation_performance.py
```

预期输出（示例）：
```
d=  512: 0.021 ms, bandwidth: 350.2 GB/s
d= 1024: 0.036 ms, bandwidth: 410.5 GB/s
d= 2048: 0.068 ms, bandwidth: 435.7 GB/s
d= 4096: 0.132 ms, bandwidth: 448.1 GB/s
```

**观察**：
- 随着 `d` 增大，带宽利用率提升
- 达到 GPU 理论带宽的 40-50%（合理范围）

## 5. 深入：`__restrict__` 关键字

### 5.1 什么是 Pointer Aliasing？

```cuda
void add(float* a, float* b, float* c) {
    *c = *a + *b;
}
```

**问题**：编译器不知道 `a`、`b`、`c` 是否指向同一块内存。
- 如果 `c == a`，那么写入 `*c` 会改变 `*a`
- 编译器必须保守优化

### 5.2 `__restrict__` 的作用

```cuda
void add(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c) {
    *c = *a + *b;
}
```

**承诺**：这些指针**不重叠**，互不影响。
- 编译器可以更激进地优化
- 例如：缓存 `*a` 和 `*b` 在寄存器中

## 6. 今日作业

1. **代码阅读**：
   - 完整阅读 `include/flashinfer/activation.cuh`
   - 理解每一行的作用

2. **实验**：
   - 运行 `test_activation_performance.py`
   - 尝试修改 `batch_size` 和 `d`，观察性能变化
   - 计算理论带宽利用率（查询你的 GPU 规格）

3. **思考问题**：
   - 为什么选择 16 字节作为向量大小？（提示：内存事务大小）
   - 如果 `d = 1`，这个 kernel 会怎样？（提示：向量化失效）
   - `#pragma unroll 1` 和 `#pragma unroll` 有什么区别？

4. **进阶挑战**：
   - 查看 `include/flashinfer/vec_dtypes.cuh` 中 `vec_t` 的实现
   - 理解 `cast_load` 和 `cast_store` 是如何实现类型转换的

## 7. 参考资料

- [CUDA Best Practices Guide - Memory Optimization](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)
- [Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)
- FlashInfer 源码：`include/flashinfer/activation.cuh`

## 下一天

[第 3 天：PTX 汇编与底层指令](./cuda_course_day03.md)

---

**重点回顾**：
- ✅ 内存合并：Warp 内线程访问连续地址
- ✅ 向量化：一次加载 16 字节（128 位）
- ✅ Grid-Stride Loop：自适应任意问题规模
- ✅ `vec_t<T, N>`：FlashInfer 的统一向量类型
- ✅ `__restrict__`：告诉编译器指针不重叠
