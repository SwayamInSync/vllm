#!/bin/bash
# Test and Benchmark Script for Optimized LayerNorm Kernel
# This script tests functional correctness and benchmarks performance improvements

set -e

echo "========================================="
echo "LayerNorm Kernel Optimization Test Suite"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Build the optimized kernel
echo -e "${YELLOW}Step 1: Building optimized LayerNorm kernel...${NC}"
python3 setup.py build_ext --inplace || {
    echo -e "${RED}Build failed! Please check CUDA installation and dependencies.${NC}"
    exit 1
}
echo -e "${GREEN}Build successful!${NC}"
echo ""

# Step 2: Run the test suite
echo -e "${YELLOW}Step 2: Running test suite for functional correctness...${NC}"
pytest tests/kernels/core/test_layernorm.py -v || {
    echo -e "${RED}Tests failed! Functional correctness not verified.${NC}"
    exit 1
}
echo -e "${GREEN}All tests passed! Functional correctness verified.${NC}"
echo ""

# Step 3: Benchmark different configurations
echo -e "${YELLOW}Step 3: Running benchmarks across different shapes and sizes...${NC}"
echo ""

# Array of configurations to test
declare -a num_tokens_arr=(256 512 1024 2048 4096 8192)
declare -a hidden_sizes_arr=(768 1024 2048 4096 5120 8192)
declare -a dtypes=("half" "bfloat16" "float")

echo "Configuration,NumTokens,HiddenSize,DType,AddResidual,Latency(us)" > benchmark_results.csv

for dtype in "${dtypes[@]}"; do
    for num_tokens in "${num_tokens_arr[@]}"; do
        for hidden_size in "${hidden_sizes_arr[@]}"; do
            for add_residual in "" "--add-residual"; do
                echo -e "${YELLOW}Testing: tokens=${num_tokens}, hidden=${hidden_size}, dtype=${dtype}, residual=${add_residual}${NC}"

                output=$(python3 benchmarks/kernels/benchmark_layernorm.py \
                    --num-tokens ${num_tokens} \
                    --hidden-size ${hidden_size} \
                    --dtype ${dtype} \
                    ${add_residual} \
                    --num-iters 100 2>&1 | grep "Kernel running time")

                latency=$(echo "$output" | grep -oP '\d+\.\d+')
                residual_flag=$([ -z "$add_residual" ] && echo "False" || echo "True")

                echo "Config,${num_tokens},${hidden_size},${dtype},${residual_flag},${latency}" >> benchmark_results.csv
                echo -e "${GREEN}Latency: ${latency} us${NC}"
            done
        done
    done
done

echo ""
echo -e "${GREEN}Benchmark complete! Results saved to benchmark_results.csv${NC}"
echo ""

# Step 4: Compare with baseline (if available)
echo -e "${YELLOW}Step 4: Performance Summary${NC}"
echo "================================================"
python3 << 'EOF'
import csv
import sys

try:
    with open('benchmark_results.csv', 'r') as f:
        reader = csv.DictReader(f)
        results = list(reader)

    print(f"\nTotal configurations tested: {len(results)}")
    print("\n=== Performance by Configuration ===")
    print(f"{'NumTokens':<12} {'HiddenSize':<12} {'DType':<10} {'Residual':<10} {'Latency(us)':<15}")
    print("-" * 70)

    for result in results:
        print(f"{result['NumTokens']:<12} {result['HiddenSize']:<12} {result['DType']:<10} {result['AddResidual']:<10} {result['Latency(us)']:<15}")

    # Calculate average latency
    latencies = [float(r['Latency(us)']) for r in results if r['Latency(us)']]
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        print(f"\nAverage latency across all configurations: {avg_latency:.3f} us")
        print("\nNote: Compare these results with baseline measurements to verify 3x+ speedup")

except Exception as e:
    print(f"Error analyzing results: {e}", file=sys.stderr)
    sys.exit(1)
EOF

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}Test and Benchmark Complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
