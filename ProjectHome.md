This project aims to reduce the complexity of GPGPU code development with an optimizing compiler. The objective is to relieve application developers of device-specific performance optimizations and to facilitate algorithm-level exploration.

In this project, we argue that application developers should be presented a simplified view of GPU hardware: many independent processors connected with offchip memory. The detailed GPU hardware features such as register files, shared memory, thread warps, on-chip memory controllers,etc., should be hidden from GPU programmers and be managed by the compiler. Based on the simplified view of GPU hardware, application developers only need to develop a "naive" version of their algorithms and the compiler will take over to generate highly optimized GPU code. The naive version, typically, represents the fine-grain data-level parallelism in the algorithm. For example, the computation to generate one element or pixel in the output domain or image.


Our compiler supports the naive kernel code written in either OpenCL or CUDA and can generate the optimized code in either OpenCL or CUDA. Although the currently target machine of the optimized code is Nvidia GTX 280, the optimized code delivers high performance on other GPUs. In our near future work, we will add support for different GPU models, including both AMD/ATI and Nvidia GPUs.

This work is supported by an NSF CAREER award CCF-0968667.

You are welcome to send your comments/feedbacks to yangyi@gmail.com or zhouhuiy@gmail.com.