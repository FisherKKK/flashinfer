# FlashInfer CUDA 编程 14 天课程

## 课程概述

本课程通过学习 FlashInfer 这个高性能 GPU 内核库的源代码，系统学习 CUDA 编程。从基础概念到高级优化技术，每天都有明确的学习目标和实际代码示例。

## 课程特点

- ✅ 基于生产级代码学习（FlashInfer 被 vLLM、SGLang 等项目使用）
- ✅ 循序渐进：从简单的激活函数到复杂的 Attention 内核
- ✅ 实战导向：每个概念都有真实代码示例
- ✅ 性能优化：学习内存合并、共享内存、Tensor Core 等优化技术
- ✅ 架构理解：深入了解 JIT 编译系统和框架集成

## 课程结构

### 第一周：CUDA 基础与内存优化 (第 1-7 天)

**第 1 天：CUDA 编程基础与 FlashInfer 环境搭建**
- CUDA 编程模型：Grid、Block、Thread
- FlashInfer 代码结构
- 搭建开发环境

**第 2 天：内存合并与向量化访问**
- 全局内存访问模式
- 向量化加载/存储
- 学习文件：`include/flashinfer/activation.cuh`

**第 3 天：PTX 汇编与底层指令**
- PTX 内联汇编
- 数学指令（exp2, log2, rsqrt）
- Warp shuffle 操作
- 学习文件：`include/flashinfer/math.cuh`

**第 4 天：共享内存基础**
- 共享内存的作用
- Bank conflicts 及避免方法
- 简单的 Reduction 操作
- 学习文件：`include/flashinfer/quantization.cuh`

**第 5 天：Warp 级编程**
- Warp 作为执行单元
- Warp shuffle 指令
- 学习文件：`include/flashinfer/norm.cuh` (Warp reduction 部分)

**第 6 天：多级 Reduction**
- Warp 内 reduction
- Block 内跨 warp reduction
- 学习文件：`include/flashinfer/norm.cuh` (完整 RMSNorm)

**第 7 天：异步内存拷贝**
- cp.async 指令
- Memory pipeline
- Commit/Wait 语义
- 学习文件：`include/flashinfer/cp_async.cuh`

### 第二周：高级特性与完整系统 (第 8-14 天)

**第 8 天：CUB 库与复杂数据结构**
- CUB 库介绍
- Block-level Scan/Reduce
- Device functors
- 学习文件：`include/flashinfer/sampling.cuh`

**第 9 天：Tensor Core 编程基础**
- Tensor Core 架构
- MMA 指令（mma.sync）
- Fragment 布局
- 学习文件：`include/flashinfer/mma.cuh`

**第 10 天：矩阵乘法优化**
- ldmatrix/stmatrix 指令
- FP8 GEMM
- 多精度支持
- 学习文件：`include/flashinfer/gemm/bmm_fp8.cuh`

**第 11 天：Attention 内核（Decode）**
- Attention 机制
- Decode 阶段特点
- RoPE 位置编码
- 学习文件：`include/flashinfer/attention/decode.cuh`

**第 12 天：Attention 内核（Prefill）**
- Prefill 复杂性
- Multi-stage pipelining
- 模板元编程
- 学习文件：`include/flashinfer/attention/prefill.cuh`

**第 13 天：JIT 编译系统**
- JIT 架构设计
- 代码生成与缓存
- Jinja2 模板
- 学习文件：`flashinfer/jit/core.py`, `flashinfer/jit/norm.py`, `flashinfer/jit/activation.py`

**第 14 天：框架集成与完整流程**
- TVM-FFI 绑定
- PyTorch 集成
- 测试与调试
- 从 kernel 到 Python API 的完整流程

## 每日学习建议

1. **阅读理论部分**（30分钟）：理解当天的核心概念
2. **代码阅读**（60分钟）：仔细阅读指定的源文件，理解实现细节
3. **实践练习**（60分钟）：运行测试、修改代码、观察结果
4. **总结反思**（30分钟）：记录学习笔记，整理知识点

## 前置知识要求

- ✅ C/C++ 编程基础
- ✅ 基本的线性代数知识
- ✅ Python 编程基础
- ⚠️ 不需要 CUDA 经验（课程从零开始）

## 学习资源

### FlashInfer 相关
- 主仓库：https://github.com/flashinfer-ai/flashinfer
- 文档：https://docs.flashinfer.ai
- 博客：https://flashinfer.ai

### CUDA 官方资源
- CUDA Programming Guide：https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- PTX ISA：https://docs.nvidia.com/cuda/parallel-thread-execution/
- CUTLASS：https://github.com/NVIDIA/cutlass

### 推荐书籍
- Programming Massively Parallel Processors（PMPP）
- CUDA C++ Best Practices Guide

## 开始学习

现在就开始第 1 天的学习吧！

[第 1 天：CUDA 编程基础与 FlashInfer 环境搭建](./cuda_course_day01.md)

---

**祝学习愉快！如果遇到问题，可以参考 FlashInfer 的 CLAUDE.md 文件获取更多开发指导。**
