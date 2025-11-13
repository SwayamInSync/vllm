#!/usr/bin/env python3
"""
Quick verification script for LayerNorm kernel optimization.
This script performs a simple smoke test to verify the optimization works.
"""

import torch
import time
from vllm.model_executor.layers.layernorm import RMSNorm

def verify_correctness():
    """Verify that optimized kernel produces correct results."""
    print("=" * 60)
    print("Verifying Correctness...")
    print("=" * 60)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Test configuration
    num_tokens = 1024
    hidden_size = 4096
    dtype = torch.float16

    # Create layer and data
    layer = RMSNorm(hidden_size).to(dtype=dtype).cuda()
    layer.weight.data.normal_(mean=1.0, std=0.1)

    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device='cuda')
    x *= 1 / (2 * hidden_size)

    # Test without residual
    print(f"\nTest 1: Without residual (tokens={num_tokens}, hidden={hidden_size})")
    ref_out = layer.forward_native(x.clone())
    opt_out = layer(x.clone())

    max_diff = (ref_out - opt_out).abs().max().item()
    mean_diff = (ref_out - opt_out).abs().mean().item()

    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")

    if max_diff < 1e-2:
        print("  ✓ PASS: Correctness verified!")
    else:
        print("  ✗ FAIL: Difference too large!")
        return False

    # Test with residual
    print(f"\nTest 2: With residual (tokens={num_tokens}, hidden={hidden_size})")
    residual = torch.randn_like(x) * (1 / (2 * hidden_size))

    ref_out, ref_res = layer.forward_native(x.clone(), residual.clone())
    opt_out, opt_res = layer(x.clone(), residual.clone())

    max_diff_out = (ref_out - opt_out).abs().max().item()
    max_diff_res = (ref_res - opt_res).abs().max().item()

    print(f"  Max difference (output): {max_diff_out:.6f}")
    print(f"  Max difference (residual): {max_diff_res:.6f}")

    if max_diff_out < 1e-2 and max_diff_res < 1e-2:
        print("  ✓ PASS: Correctness verified!")
    else:
        print("  ✗ FAIL: Difference too large!")
        return False

    return True

def quick_benchmark():
    """Quick performance benchmark."""
    print("\n" + "=" * 60)
    print("Quick Performance Benchmark")
    print("=" * 60)

    configs = [
        (1024, 2048, torch.float16),
        (2048, 4096, torch.float16),
        (4096, 8192, torch.float16),
    ]

    print(f"\n{'Tokens':<10} {'Hidden':<10} {'DType':<10} {'Latency (μs)':<15}")
    print("-" * 60)

    for num_tokens, hidden_size, dtype in configs:
        torch.cuda.manual_seed(42)

        layer = RMSNorm(hidden_size).to(dtype=dtype).cuda()
        layer.weight.data.normal_(mean=1.0, std=0.1)

        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device='cuda')
        x *= 1 / (2 * hidden_size)

        # Warmup
        for _ in range(10):
            _ = layer(x)

        torch.cuda.synchronize()

        # Benchmark
        num_iters = 100
        start = time.perf_counter()
        for _ in range(num_iters):
            _ = layer(x)
        torch.cuda.synchronize()
        end = time.perf_counter()

        latency_us = ((end - start) / num_iters) * 1e6
        print(f"{num_tokens:<10} {hidden_size:<10} {str(dtype).split('.')[-1]:<10} {latency_us:<15.3f}")

    print("\nNote: Compare these results with baseline to verify 3x+ speedup")

def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires a CUDA-enabled GPU.")
        return 1

    print("\nLayerNorm Kernel Optimization Verification")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Verify correctness
    if not verify_correctness():
        print("\n❌ Correctness verification failed!")
        return 1

    # Quick benchmark
    try:
        quick_benchmark()
    except Exception as e:
        print(f"\n⚠ Benchmark failed: {e}")
        print("This is not critical, but you should run full benchmarks manually.")

    print("\n" + "=" * 60)
    print("✅ Verification Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run full test suite: pytest tests/kernels/core/test_layernorm.py -v")
    print("2. Run comprehensive benchmarks: ./test_and_benchmark_layernorm.sh")
    print()

    return 0

if __name__ == "__main__":
    exit(main())
