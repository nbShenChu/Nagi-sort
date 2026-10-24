Here is a complete, professional, and impressive `README.md` for your **Nagi** sorting framework. It is written in English, structured like a top-tier open-source C++ library, and highlights its impressive performance and architectural design.

You can copy and paste this directly into your `README.md` file.

***

# Nagi (凪) — The Ripples of Yore

> *"From a single ripple in Python to a highly optimized adaptive sorting framework in C++. The wind has stopped, the sea is calm, but the ripples remember."*

**Nagi** is a single-header, drop-in adaptive sorting framework for C++17 and C++20. It is designed with a hyper-focus on **integers, large data arrays, and multi-core performance**. 

By intelligently probing data characteristics before sorting, Nagi dynamically routes the workload to the most efficient algorithm—achieving up to **2.4× speedup over `pdqsort`** and **5.6× speedup over `std::sort`** on large integer arrays.

## ✨ Features

- **Adaptive Dispatch Architecture**: A pure-function dispatcher (`plan_initial_route`) combined with a data-aware refinement layer (`refine_route`) that routes data based on size, shape, and entropy.
- **Blazing Fast for Large Integers**: Uses a hybrid of MSD+LSD radix sort and counting sort. Up to 2.4× faster than `pdqsort`.
- **Multi-Core Ready**: A highly optimized parallel radix sort using ping-pong memory buffers to eliminate memory bandwidth bottlenecks.
- **Worst-Case Guaranteed**: Every strategy has a fallback. If adaptive paths fail or degrade, it seamlessly falls back to a highly optimized `introsort`.
- **Zero Abstraction Overhead**: Extensive use of compile-time polymorphism (`if constexpr`, type traits, and `static_assert`) ensures you don't pay for what you don't use.
- **Bulletproof Correctness**: Built to survive property-based fuzzing, boundary conditions, and adversarial data (e.g., NaN handling, `INT_MAX` overflows, partition invariants).

## 🚀 Quick Start

Since Nagi is a single-header library, just drop `nagi.h` into your project and include it:

```cpp
#include "nagi.h"
#include <vector>
#include <iostream>

int main() {
    std::vector<int> data = {5, 2, 9, 1, 5, 6};

    // Ascending sort (default)
    nagi::sort(data);

    // Descending sort
    nagi::sort(data, std::greater<int>{});

    // Top-K (only guarantees the first k elements are sorted)
    nagi::topk(data, 3); 

    // Force Radix Sort (bypasses prescan)
    nagi::radix_sort(data);

    return 0;
}
```

## 🏗️ Architecture

Nagi decouples the **decision** of how to sort from the **execution** of sorting. 

```text
L5. Public API      → sort / topk / radix_sort
L4. Dispatch Layer  → plan_initial_route (Pure function, no data access)
L3. Refine Layer    → refine_route (Data-aware decision)
L2. Feature Probe   → try_prescan (Data shape detection)
L1. Strategies      → small_sort / radix / counting / introsort
```

1. **Dispatch Layer**: Evaluates `n`, `k`, and the comparator type at compile time.
2. **Refine Layer**: Scans the data to detect already-sorted, reversed, nearly-sorted, or highly duplicated arrays.
3. **Strategy Execution**: 
   - `small_sort`: pdqsort-inspired quicksort for `n <= 128`.
   - `radix / counting`: Entropy-adaptive integer sorting for large data.
   - `introsort`: The ultimate worst-case fallback.

## 🔬 Rigorous Correctness & Benchmarks

We don't just claim it's fast; we prove it's correct and measure it rigorously.

### 1. Bulletproof Correctness
Nagi is validated against an exhaustive, adversarial test matrix **before** any performance benchmarks are executed. If a single element is out of place, the suite fails immediately.
- **16,576 test cases passed (0 failed)**.
- **19 data distributions**: All Equal, Sorted, Reversed, Organ Pipe, Sawtooth, Bit Reversal, Narrow Band, Many Duplicates, etc.
- **6 integer types**: `int`, `unsigned int`, `long long`, `unsigned long long`, `short`, `int8_t`.
- **Strict invariants checked**: 
  - `std::is_sorted` (or reverse) validated.
  - **Permutation check**: The output must be a strict permutation of the input (no elements lost or duplicated).
  - `topk` and `LazySortedView` APIs strictly maintain prefix ordering without corrupting the underlying multiset.

### 2. Performance Benchmark
*Measured with Google Benchmark (min_time=0.5s, repetitions=3) on an Intel i5-4210U (2 physical / 4 logical cores, 1.7 GHz, 3MB L3 cache).*

**Raw Execution Time (Median of 3 runs, int, uniformly random, ascending)**

| Scale (N) | Nagi (ms) | `std::sort` (ms) | `std::stable_sort` (ms) | Speedup (vs `std::sort`) |
| :--- | :--- | :--- | :--- | :--- |
| **100,000** | **2.408** | 9.081 | 8.653 | **3.77×** |
| **1,000,000** | **13.997** | 104.817 | 98.839 | **7.49×** |
| **10,000,000** | **165.020** | 1240.447 | 1145.933 | **7.52×** |

> **Note:** 
> 1. The 1M and 10M benchmarks automatically trigger Nagi's multi-threaded `radix_sort_32_parallel` path (baseline is single-threaded).
> 2. *Empirical Complexity*: The single-threaded paths (1K to 1M) exhibit an empirical complexity of **O(N log N)** with a fitted coefficient of **0.782** and an RMS error of just **0.44%**. For large integer datasets, the adaptive router kicks into linear O(N) radix/counting sort, resulting in massive speedups.

### 3. How to Reproduce
All benchmarks and correctness checks are fully reproducible. 
```bash
# Clone the repo
git clone https://github.com/nbShenChu/Nagi-sort.git
cd Nagi-sort/bench

# Run the full suite (Correctness + Benchmarks)
run.bat


## 📊 Performance Benchmark

*Test environment: Random integers, single & multi-threaded, compiled with `-O3`.*

| Scenario | Nagi | pdqsort | std::sort | Speedup (vs std::sort) |
| :--- | :--- | :--- | :--- | :--- |
| **100K Random (ASC)** | **1.61 ms** | 4.60 ms | 8.38 ms | **5.2×** |
| **100K Random (DESC)** | **1.56 ms** | 3.70 ms | 8.32 ms | **5.3×** |
| **1M Random (ASC)** | **18.01 ms** | 43.22 ms | 104.19 ms | **5.7×** |
| **1M Random (DESC)** | **18.21 ms** | 42.93 ms | 101.85 ms | **5.6×** |
| **All Equal (1000)** | **0.0002 ms** | 0.0025 ms | 0.0012 ms | **6.0×** |
| **Sawtooth (1000)** | **0.0032 ms** | 0.0275 ms | 0.0238 ms | **7.4×** |

*(Note: Nagi's performance shines brightest on medium-to-large datasets where radix and counting sort paths kick in).*

## 🧠 Design Philosophy

- **No Magic Thresholds**: All thresholds (e.g., `SMALL_DATA_THRESHOLD`, `MSD_LSD_THRESHOLD`) are rigorously benchmarked and tunable via macros.
- **Permutation Invariant**: All APIs (`sort`, `topk`) guarantee that the resulting vector is a strict permutation of the input (no elements are lost or duplicated).
- **Worst-Case Fallback**: A bug in a heuristic should never degrade your performance to \(O(N^2)\). Every adaptive path has a bounded budget and falls back to `introsort` safely.

## 👥 The Evolution / Credits

Nagi was not built in a day. It evolved through multiple iterations of AI-driven code reviews, property-based fuzzing, and architectural rewrites:
- **The Creator**: Started the journey as a Python ripple sort, evolving into a C++ Introsort + Radix hybrid.
- **The Architect**: Introduced the `Dispatch → Refine → Strategy` layered architecture (v5.0.0).
- **The Hotfix Engineer**: Fixed the latent `partial_insertion_sort` partition bug (v5.1.4).
- **The Static & Dynamic Auditors**: Identified the `block_partition_right` residual ordering bug and O(N²) nearly-sorted vulnerability (v5.1.5).

## 📄 License

MIT License. Feel free to use it, fork it, and make it faster.
