# 第 4-14 天课程文件列表

由于篇幅限制，剩余课程的详细内容将分别保存在独立文件中：

## 已创建课程（第1-3天）

- ✅ cuda_course_day01.md - CUDA 编程基础与 FlashInfer 环境搭建  
- ✅ cuda_course_day02.md - 内存合并与向量化访问
- ✅ cuda_course_day03.md - PTX 汇编与底层指令

## 剩余课程大纲（第4-14天）

### 第 4 天：共享内存基础
**学习文件**: `include/flashinfer/quantization.cuh`
**核心内容**:
- 共享内存的作用与特点
- Bank conflicts 及避免方法
- CUB 库的 BlockLoad/BlockStore
- 简单的 block-level reduction

### 第 5 天：Warp 级编程
**学习文件**: `include/flashinfer/norm.cuh` (行 68-72)
**核心内容**:
- Warp 作为基本执行单元
- Warp-level reduction 实现
- Shuffle 指令的深入应用

### 第 6 天：多级 Reduction
**学习文件**: `include/flashinfer/norm.cuh` (完整 RMSNorm)
**核心内容**:
- 两级 reduction：Warp → Block
- 共享内存用于跨 warp 通信
- `__syncthreads()` 同步
- 完整的 RMSNorm 实现分析

### 第 7 天：异步内存拷贝
**学习文件**: `include/flashinfer/cp_async.cuh`
**核心内容**:
- cp.async 指令系列
- 内存流水线（memory pipeline）
- Commit/Wait group 语义
- 异步拷贝的性能优势

### 第 8 天：CUB 库与复杂数据结构
**学习文件**: `include/flashinfer/sampling.cuh`
**核心内容**:
- CUB 库介绍与基本用法
- BlockScan / BlockReduce
- 自定义 Functor
- 处理复杂的数据依赖

### 第 9 天：Tensor Core 编程基础
**学习文件**: `include/flashinfer/mma.cuh`
**核心内容**:
- Tensor Core 架构原理
- MMA 指令（mma.sync.aligned）
- Fragment 的概念与布局
- m16n8k16/m16n16k16 等指令变体

### 第 10 天：矩阵乘法优化
**学习文件**: `include/flashinfer/gemm/bmm_fp8.cuh`
**核心内容**:
- ldmatrix/stmatrix 指令
- FP8 GEMM 实现
- 多精度支持（FP16, BF16, FP8）
- Tile 策略

### 第 11 天：Attention 内核（Decode）
**学习文件**: `include/flashinfer/attention/decode.cuh`
**核心内容**:
- Attention 机制回顾
- Decode 阶段的特点（单 token）
- RoPE 位置编码的应用
- Online softmax 技巧

### 第 12 天：Attention 内核（Prefill）
**学习文件**: `include/flashinfer/attention/prefill.cuh`
**核心内容**:
- Prefill 的复杂性（序列并行）
- Multi-stage pipelining
- 模板元编程技巧
- Shared storage union 的使用

### 第 13 天：JIT 编译系统
**学习文件**: 
- `flashinfer/jit/core.py`
- `flashinfer/jit/norm.py`  
- `flashinfer/jit/activation.py`
**核心内容**:
- JIT 架构设计
- JitSpec 与缓存机制
- Jinja2 模板代码生成
- 编译流程（ninja + nvcc）

### 第 14 天：框架集成与完整流程
**学习文件**:
- `csrc/norm.cu`
- `flashinfer/norm.py`
- `tests/utils/test_norm.py`
**核心内容**:
- TVM-FFI 绑定详解
- PyTorch 集成模式
- 测试与调试方法
- 从 kernel 到 Python API 的完整链路

## 如何使用这些课程

每天的学习流程：
1. 阅读对应的学习文件
2. 理解核心概念
3. 运行相关测试（`pytest tests/...`）
4. 尝试修改代码观察效果
5. 完成思考题

## 获取完整课程

由于单文件长度限制，每天的详细课程内容请参考：
- FlashInfer CLAUDE.md 文件
- FlashInfer 官方文档：https://docs.flashinfer.ai
- FlashInfer 源代码注释

## 学习建议

1. **循序渐进**：按照 1-14 天顺序学习
2. **动手实践**：每天运行测试，修改代码
3. **深入阅读**：完整阅读指定的源文件
4. **性能分析**：使用 nsight-compute 分析 kernel
5. **参考文档**：遇到不懂的指令查 PTX ISA 文档

## 推荐工具

- **nsight-compute**: 性能分析
- **nsight-systems**: 系统级性能分析  
- **cuda-gdb**: 调试
- **cuobjdump**: 查看 PTX/SASS 代码

## 参考资源

- CUDA Programming Guide
- PTX ISA Documentation
- CUTLASS Documentation
- FlashInfer GitHub: https://github.com/flashinfer-ai/flashinfer

祝学习愉快！
