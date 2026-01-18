# 第 1 天：CUDA 编程基础与 FlashInfer 环境搭建

## 学习目标

- 理解 CUDA 编程模型（Grid、Block、Thread、Warp）
- 了解 FlashInfer 的代码结构和设计理念
- 搭建 FlashInfer 开发环境
- 理解 JIT（Just-In-Time）编译的工作原理

## 1. CUDA 编程模型基础

### 1.1 层次结构

CUDA 使用分层的并行结构来组织线程：

```
Grid (网格)
  └─ Block (线程块) [可有多个]
       └─ Warp (线程束) [32个线程为一组]
            └─ Thread (线程) [最小执行单元]
```

**关键概念**：

- **Thread（线程）**：最小执行单元，每个线程执行相同的 kernel 代码
- **Warp（线程束）**：32 个线程组成一个 warp，这是 GPU 的基本调度单元
- **Block（线程块）**：一组线程，共享同一块共享内存（Shared Memory）
- **Grid（网格）**：所有线程块的集合

### 1.2 线程索引

在 CUDA kernel 中，可以通过内置变量访问线程索引：

```cuda
// Block 维度
blockDim.x, blockDim.y, blockDim.z

// Thread 在 block 中的索引
threadIdx.x, threadIdx.y, threadIdx.z

// Block 在 grid 中的索引
blockIdx.x, blockIdx.y, blockIdx.z

// Grid 维度
gridDim.x, gridDim.y, gridDim.z
```

**计算全局索引示例**：
```cuda
// 1D grid of 1D blocks
int global_id = blockIdx.x * blockDim.x + threadIdx.x;

// 2D grid of 2D blocks
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

### 1.3 内存层次

CUDA 有多种类型的内存，速度和容量各不相同：

| 内存类型 | 位置 | 延迟 | 容量 | 作用域 |
|---------|------|------|------|--------|
| Register | 芯片上 | 最快（~1 cycle） | 很小（~64KB/SM） | 单个线程 |
| Shared Memory | 芯片上 | 快（~数十 cycles） | 小（~100KB/SM） | Block内所有线程 |
| L1/L2 Cache | 芯片上 | 中等 | 中等（MB级） | 自动管理 |
| Global Memory | 显存 | 慢（~数百 cycles） | 大（GB级） | 所有线程 |
| Constant Memory | 显存 | 中等（有缓存） | 中等（64KB） | 只读，所有线程 |

**优化目标**：尽可能使用 Register 和 Shared Memory，减少 Global Memory 访问。

## 2. FlashInfer 代码结构

### 2.1 目录组织

```
flashinfer/
├── include/flashinfer/          # ⭐ CUDA kernel 定义（框架无关）
│   ├── attention/               # Attention 相关 kernel
│   ├── gemm/                    # 矩阵乘法 kernel
│   ├── activation.cuh           # 激活函数
│   ├── norm.cuh                 # 归一化（RMSNorm, LayerNorm）
│   ├── math.cuh                 # 数学函数和 PTX 汇编
│   ├── vec_dtypes.cuh          # 向量类型定义
│   └── ...
│
├── csrc/                        # ⭐ PyTorch 绑定层（使用 TVM-FFI）
│   ├── norm.cu                  # RMSNorm 启动器
│   ├── flashinfer_norm_binding.cu  # TVM-FFI 导出
│   └── ...
│
├── flashinfer/                  # ⭐ Python 包
│   ├── jit/                     # JIT 编译系统
│   │   ├── core.py              # JitSpec 核心
│   │   ├── norm.py              # Norm JIT 模块生成器
│   │   └── ...
│   ├── norm.py                  # Python API
│   ├── decode.py                # Decode attention API
│   └── ...
│
├── tests/                       # 测试
└── benchmarks/                  # 性能测试
```

### 2.2 关键设计理念

**框架分离原则**：
- `include/` 目录中的 kernel **不依赖任何框架**（不能 include torch 头文件）
- 所有 kernel 接受原始指针（`float*`，`half*` 等）
- PyTorch/其他框架的绑定在 `csrc/` 中通过 TVM-FFI 实现

**为什么这样设计？**
- 同一个 kernel 可以被不同框架使用（PyTorch、JAX、TensorFlow 等）
- Kernel 代码更简洁，专注于计算逻辑
- 易于维护和测试

## 3. JIT（Just-In-Time）编译

### 3.1 为什么需要 JIT？

传统的 CUDA 库需要为所有可能的参数组合预编译 kernel：

```
dtype × head_dim × block_size × ...
(fp16, fp32, bf16) × (64, 128, 256) × (128, 256, 512) × ...
= 数百甚至数千个组合！
```

**问题**：
- 包体积巨大
- 编译时间长
- 无法支持用户自定义参数

**FlashInfer 的解决方案**：JIT 编译
- 首次调用时：根据实际参数生成并编译 kernel
- 编译结果缓存到 `~/.cache/flashinfer/`
- 后续调用：直接加载缓存的 `.so` 文件
- 源码改变时：自动检测并重新编译

### 3.2 JIT 工作流程

```
1. Python 调用 flashinfer.rmsnorm(input, weight)
   ↓
2. 检查缓存：是否已编译过相同参数的 kernel？
   ├─ 是 → 加载缓存的 .so 文件
   └─ 否 ↓
3. 生成代码：
   - 复制 kernel 源文件到临时目录
   - (可选) 使用 Jinja2 模板生成类型特化代码
   ↓
4. 编译：
   - 生成 build.ninja
   - 调用 ninja → nvcc 编译 .cu → .so
   ↓
5. 加载：
   - 通过 TVM-FFI 加载 .so
   - 缓存 Python 端的模块对象（@functools.cache）
   ↓
6. 执行 kernel
```

### 3.3 开发时的优势

**对开发者来说，JIT 意味着**：
- ✅ 修改 `include/` 中的 `.cuh` 文件后，**不需要重新 pip install**
- ✅ 只需重新运行测试，JIT 自动检测变化并重新编译
- ✅ 快速迭代开发

## 4. 环境搭建

### 4.1 前置要求

- CUDA Toolkit（推荐 12.6+ 或 13.0+）
- Python 3.8+
- PyTorch（已安装且与 CUDA 版本匹配）
- Ninja build system

### 4.2 克隆仓库

```bash
# --recursive 很重要！会下载 3rdparty/ 子模块（cutlass, spdlog）
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer

# 如果忘记了 --recursive
git submodule update --init --recursive
```

### 4.3 安装开发版本

```bash
# --no-build-isolation: 防止 pip 拉取不兼容的 PyTorch/CUDA 版本
# -e: 可编辑模式（editable）
# -v: 显示详细输出
pip install --no-build-isolation -e . -v
```

安装成功后，你可以：
- 修改 `include/` 中的任何 `.cuh` 文件
- 重新运行代码，JIT 会自动重新编译

### 4.4 验证安装

```bash
# 查看 FlashInfer 配置
flashinfer show-config

# 运行简单测试
pytest tests/utils/test_activation.py -v
```

### 4.5 （可选）安装预编译包

如果你不需要修改 kernel 代码，可以安装预编译版本以加快启动速度：

```bash
# 核心包（会 JIT 编译）
pip install flashinfer-python

# 预编译二进制（跳过大部分 JIT）
pip install flashinfer-cubin

# JIT 缓存（特定 CUDA 版本，如 cu129 表示 CUDA 12.9）
pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu129
```

## 5. 第一个 FlashInfer 程序

让我们运行一个简单的例子，理解整个流程。

### 5.1 创建测试文件

创建 `test_my_first_kernel.py`：

```python
import torch
import flashinfer

# 准备输入数据
batch_size = 4
hidden_size = 1024
input = torch.randn(batch_size, hidden_size, device='cuda', dtype=torch.float16)
weight = torch.randn(hidden_size, device='cuda', dtype=torch.float16)

# 调用 FlashInfer 的 RMSNorm
output = flashinfer.rmsnorm(input, weight, eps=1e-6)

print(f"Input shape: {input.shape}")
print(f"Output shape: {output.shape}")
print(f"First few values: {output[0, :5]}")
```

### 5.2 运行并观察

```bash
python test_my_first_kernel.py
```

**首次运行**：
- 你会看到 JIT 编译的日志（如果设置了 `FLASHINFER_JIT_VERBOSE=1`）
- 编译需要几秒钟
- 在 `~/.cache/flashinfer/` 中生成缓存

**再次运行**：
- 直接加载缓存，几乎瞬间完成

### 5.3 查看生成的代码

```bash
# 查看缓存目录
ls -la ~/.cache/flashinfer/

# 查看某个具体模块的生成代码
find ~/.cache/flashinfer/ -name "*.cu" | head -1 | xargs cat
```

## 6. 代码阅读：第一个简单 Kernel

让我们看一个最简单的例子：`vec_dtypes.cuh` 中的向量类型定义。

打开 `include/flashinfer/vec_dtypes.cuh`：

```cuda
// FLASHINFER_INLINE 宏定义
#define FLASHINFER_INLINE inline __attribute__((always_inline)) __device__

// __device__: 这个函数只能在 GPU 上调用
// __forceinline__: 强制内联，减少函数调用开销
__device__ __forceinline__ void st_global_release(int4 const& val, int4* addr) {
  // PTX 内联汇编：直接使用 GPU 指令
  // st.release.global.sys.v4.b32: 存储 4 个 32 位值，带 release 语义
  asm volatile("st.release.global.sys.v4.b32 [%4], {%0, %1, %2, %3};"
               ::"r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w), "l"(addr));
}
```

**知识点**：
- `__device__`：标记 GPU 函数
- `__forceinline__`：强制内联优化
- `asm volatile`：内联 PTX 汇编
- `int4`：CUDA 内置类型，包含 4 个 int（128 位）
- `v4.b32`：向量化操作，一次加载/存储 128 位

## 7. 今日作业

1. **环境搭建**：
   - 安装 FlashInfer 开发版本
   - 运行 `pytest tests/utils/test_activation.py -v` 确保测试通过

2. **代码阅读**：
   - 阅读 `include/flashinfer/vec_dtypes.cuh` 前 200 行
   - 理解 `FLASHINFER_INLINE` 宏的作用
   - 找到 `ld_global_acquire` 函数，理解它与 `st_global_release` 的对应关系

3. **实验**：
   - 修改第一个程序中的 `batch_size` 和 `hidden_size`
   - 观察不同参数下是否会触发重新编译（查看 `~/.cache/flashinfer/` 目录变化）

4. **思考问题**：
   - 为什么要使用 `int4` 而不是 4 个 `int`？
   - `asm volatile` 中的 `"r"` 和 `"l"` 是什么意思？（提示：查阅 PTX ISA 文档）
   - JIT 编译的缓存键是如何生成的？（提示：查看 `flashinfer/jit/core.py`）

## 8. 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [FlashInfer Documentation](https://docs.flashinfer.ai)
- [FlashInfer CLAUDE.md](./CLAUDE.md) - 项目开发指南

## 下一天

[第 2 天：内存合并与向量化访问](./cuda_course_day02.md)

---

**重点回顾**：
- ✅ CUDA 分层结构：Grid → Block → Warp → Thread
- ✅ FlashInfer 的框架无关设计：include/ 不依赖 PyTorch
- ✅ JIT 编译让开发更高效：改代码不需要重新安装
- ✅ 环境搭建：`pip install --no-build-isolation -e . -v`
