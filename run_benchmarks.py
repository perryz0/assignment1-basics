#!/usr/bin/env python3
"""
Run benchmark.py multiple times and aggregate results.
"""
import subprocess
import re
import matplotlib.pyplot as plt
from statistics import mean, stdev

def run_benchmark():
    """Run benchmark.py and parse the output."""
    result = subprocess.run(
        ["uv", "run", "python", "benchmark.py"],
        capture_output=True,
        text=True
    )
    
    # Parse output lines
    no_cp_time = None
    uniform_cp_time = None
    boundary_cp_time = None
    
    for line in result.stdout.split('\n'):
        line = line.strip()
        # Match lines that start with the benchmark names
        if line.startswith("No checkpointing:"):
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                no_cp_time = float(match.group(1))
        elif line.startswith("Uniform checkpointing:"):
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                uniform_cp_time = float(match.group(1))
        elif line.startswith("Boundary-aware CP:"):
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                boundary_cp_time = float(match.group(1))
    
    return no_cp_time, uniform_cp_time, boundary_cp_time


def main():
    num_runs = 20
    
    print(f"Running benchmark {num_runs} times...")
    print("This may take a while...\n")
    
    no_cp_times = []
    uniform_cp_times = []
    boundary_cp_times = []
    
    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}...", end=" ", flush=True)
        no_cp, uniform_cp, boundary_cp = run_benchmark()
        
        if no_cp is not None and uniform_cp is not None and boundary_cp is not None:
            no_cp_times.append(no_cp)
            uniform_cp_times.append(uniform_cp)
            boundary_cp_times.append(boundary_cp)
            print("✓")
        else:
            print("✗ (failed to parse)")
            if no_cp is None:
                print("  Warning: No checkpointing value not found")
            if uniform_cp is None:
                print("  Warning: Uniform checkpointing value not found")
            if boundary_cp is None:
                print("  Warning: Boundary-aware CP value not found")
    
    if len(no_cp_times) == 0:
        print("\nError: No successful runs!")
        return
    
    # Compute statistics
    no_cp_mean = mean(no_cp_times)
    no_cp_std = stdev(no_cp_times) if len(no_cp_times) > 1 else 0.0
    
    uniform_cp_mean = mean(uniform_cp_times)
    uniform_cp_std = stdev(uniform_cp_times) if len(uniform_cp_times) > 1 else 0.0
    
    boundary_cp_mean = mean(boundary_cp_times)
    boundary_cp_std = stdev(boundary_cp_times) if len(boundary_cp_times) > 1 else 0.0
    
    # Print summary table
    print("\n===== Aggregate Benchmark Results =====")
    print(f"No-CP:        mean = {no_cp_mean:.6f}, std = {no_cp_std:.6f}")
    print(f"Uniform-CP:   mean = {uniform_cp_mean:.6f}, std = {uniform_cp_std:.6f}")
    print(f"Boundary-CP:  mean = {boundary_cp_mean:.6f}, std = {boundary_cp_std:.6f}")
    print("======================================")
    
    # Create bar plot with error bars
    labels = ["No-CP", "Uniform-CP", "Boundary-CP"]
    means = [no_cp_mean, uniform_cp_mean, boundary_cp_mean]
    stds = [no_cp_std, uniform_cp_std, boundary_cp_std]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, means, yerr=stds, capsize=10, alpha=0.7, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.ylabel("Time per step (sec)")
    plt.title("Aggregate Checkpointing Benchmark Results")
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (mean_val, std_val) in enumerate(zip(means, stds)):
        plt.text(i, mean_val + std_val + max(means) * 0.02, 
                f'{mean_val:.4f}±{std_val:.4f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("aggregate_results.png", dpi=150)
    print(f"\nSaved plot to aggregate_results.png")


if __name__ == "__main__":
    main()

