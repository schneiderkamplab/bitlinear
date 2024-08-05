# import torch
# import triton
# import triton.language as tl 
# from .utils import Kernel, get_cuda_autotune_config

# ###### Baseline ######
# class default(Kernel):
        
#     def __call__(self, activations, weights, bias, scale):
#         # Check constraints.
#         assert activations.shape[1] == weights.shape[1], "Incompatible dimensions"
#         assert activations.is_contiguous(), "Matrix A must be contiguous"
#         assert activations.shape[0] == bias.shape[0], "Bias dimension must match input"
        
#         M, K = activations.shape
#         N = weights.shape[0]
        
#         weights = weights.transpose(0, 1)
        
#         # Allocates output.
#         output = torch.empty((M, N), device=activations.device, dtype=torch.float16)
        
#         # 1D launch kernel where each block gets its own program.
#         grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
        
#         kernel[grid](
#             activations, weights, output, bias,  #
#             M, N, K,  #
#             activations.stride(0), activations.stride(1),  #
#             weights.stride(0), weights.stride(1),  #
#             output.stride(0), output.stride(1),  #
#             bias.stride(0)
#             )
#         return output * scale
    

# def get_cuda_autotune_config():
#     return [
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
#                       num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
#                       num_warps=2),
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
#                       num_warps=2),
#         # Good config for fp8 inputs.
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
#                       num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
#                       num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
#                       num_warps=4)
#     ]
    
# @triton.autotune(
#     configs=get_cuda_autotune_config(),
#     key=['M', 'N', 'K'],
# )
# @triton.jit
# def kernel(
#         # Pointers to matrices
#         activation_ptr, weights_ptr, ouput_ptr, bias_ptr,
#         # Matrix dimensions
#         M, N, K,
#         # The stride variables represent how much to increase the ptr by when moving by 1
#         # element in a particular dimension. E.g. `stride_am` is how much to increase `activation_ptr`
#         # by to get the element one row down (A has M rows).
#         stride_am, stride_ak,  
#         stride_bk, stride_bn,  
#         stride_cm, stride_cn,
#         stride_dm,
#         # Meta-parameters
#         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  
#         GROUP_SIZE_M: tl.constexpr
#         ):
#     """Kernel for computing the matmul C = A x B + D
#     A has shape (M, K), B has shape (K, N) and D has shape (M, 1)
#     Output C has shape (M, N)
#     """
#     # -----------------------------------------------------------
#     # Map program ids `pid` to the block of C it should compute.
#     # This is done in a grouped ordering to promote L2 data reuse.
    
#     pid = tl.program_id(axis=0)
    
#     num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#     num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
#     num_pid_in_group = GROUP_SIZE_M * num_pid_n
#     group_id = pid // num_pid_in_group
#     first_pid_m = group_id * GROUP_SIZE_M
#     group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
#     pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
#     pid_n = (pid % num_pid_in_group) // group_size_m

#     # ----------------------------------------------------------
#     # Create pointers for the first blocks of A and B.
#     # We will advance this pointer as we move in the K direction
#     # and accumulate
#     offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
#     offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
#     offs_k = tl.arange(0, BLOCK_SIZE_K)
#     activation_ptrs = activation_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
#     weights_ptrs = weights_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

#     # -----------------------------------------------------------
#     # Iterate to compute a block of the C matrix.
#     # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
#     # of fp32 values for higher accuracy.
#     accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
#     for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
#         # Load the next block of A and B, generate a mask by checking the K dimension.
#         # If it is out of bounds, set it to 0.
#         inputs = tl.load(activation_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
#         weights = tl.load(weights_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
#         # We accumulate along the K dimension.
#         accumulator += tl.dot(inputs, weights)
#         # Advance the ptrs to the next K block.
#         activation_ptrs += BLOCK_SIZE_K * stride_ak
#         weights_ptrs += BLOCK_SIZE_K * stride_bk

#     # Add bias D to the accumulated result
#     bias_ptrs = bias_ptr + offs_am[:, None] * stride_dm
#     bias = tl.load(bias_ptrs)
#     accumulator += bias

#     output = accumulator.to(tl.float16)

#     # -----------------------------------------------------------
#     # Write back the block of the output matrix C with masks.
#     offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     ouput_ptrs = ouput_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
#     ouput_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
#     tl.store(ouput_ptrs, output, mask=ouput_mask)
    