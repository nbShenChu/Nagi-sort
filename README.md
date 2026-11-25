🏗️ Architecture
Nagi decouples the decision of how to sort from the execution of sorting.
plain
L5. Public API      → sort / topk / radix_sort / LazySortedView
L4. Dispatch Layer  → plan_initial_route (Pure function, no data access)
L3. Refine Layer    → refine_route (Data-aware decision)
L2. Feature Probe   → try_prescan (Data shape detection)
L1. Strategies      → small_sort / radix / counting / introsort
Dispatch Layer: Evaluates n, k, and the comparator type at compile time.
Refine Layer: Scans the data to detect already-sorted, reversed, nearly-sorted, or highly duplicated arrays.
Strategy Execution:
small_sort: pdqsort-inspired quicksort for n <= 128.
radix / counting: Entropy-adaptive integer/float sorting for large data.
introsort: The ultimate worst-case fallback with block partitioning for arithmetic types.
📊 Performance (v5.1.6 vs v5.1.5 vs std::sort)
Geometric mean across 216 test cases: 1.660× vs v5.1.5
int32 Random
Scale 	vs v5.1.5 	vs std::sort (100万) 
512	1.19×	—
2048	1.00×	—
50K	2.48×	—
200K	1.30×	—
1M	0.93×	7.4×
4M	1.06×	—
float Random
Scale 	vs v5.1.5 	vs std::sort (100万) 
512	1.43×	—
2048	3.33×	—
50K	3.53×	—
200K	3.64×	—
1M	3.63×	6.1×
4M	4.46×	—
Pattern-Specific Speedups (v5.1.6)
Pattern 	Speedup vs v5.1.5 
Already Sorted	5.0–5.9×
All Equal	5.0–5.9×
Sawtooth	1.8–2.0×
Organ Pipe (float/double)	3.1–6.1×
Note: The float speedups come from the new IEEE-754 radix path (bit-flip mapping for sign-magnitude + NaN detection). v5.1.5 had no float radix support at all.
🔬 Rigorous Correctness
Nagi is validated against an exhaustive, adversarial test matrix before any performance benchmarks are executed.
81,060 test cases passed (0 failed, 0 warnings).
10 types: int, unsigned int, long, long long, unsigned long long, short, int8_t, uint8_t, float, double.
Boundary lengths: 0, 1, 2, …, 49, 50, 51, 131072.
14 adversarial distributions: Random, Sorted, Reversed, Nearly Sorted, All Equal, Organ Pipe, Sawtooth, Bit Reversal, Narrow Band, Many Duplicates, Few Unique, Gaussian, Zipf, Adversarial Median-of-Medians.
5 APIs tested: sort, sort(greater), topk, radix_sort, LazySortedView.
Edge cases: INT_MIN/INT_MAX, ±0.0, inf, subnormals, NaN (float/double).
Compilers: MSVC C++17/C++20 /W4, clang-cl.
Parallel path: Additional 2M–3M scale专项验证 for n >= 2e6.
Strict Invariants Checked
std::is_sorted (or reverse) validated.
Permutation check: The output must be a strict permutation of the input (no elements lost or duplicated).
topk and LazySortedView APIs strictly maintain prefix ordering without corrupting the underlying multiset.
🐛 v5.1.6 Bug Fixes (5 Real Bugs Caught)
Bug 	Impact 	How Found 
MSD buckets of size ≤1 never copied back from scratch buffer — d still held pre-scatter values.	Silent mis-sort when singleton buckets occurred.	Float random data (integers almost never hit singletons).
Descending MSD+LSD: buckets laid out back-to-front, but refill derived ranges front-to-back.	Entire wrong spans sorted for large n descending.	Systematic descending test.
used_digits not clamped to digits	Histogram buffer overflow → segfault.	Fuzz testing.
48 KiB histogram on stack	Stack overflow when caller inlined multiple instantiations (1 MiB stack).	Test program crashed on startup.
radix64_capable_v<double> hard compile error	make_unsigned_t instantiated before && short-circuit.	Template instantiation test.
⚡ v5.1.6 Performance Changes
Radix Engine Rewrite
Unified LSD kernel serves every key width (8/16/32/64 bits). Digit width chosen per-run by a cost model, capped at 8 bits once the run exceeds L1 cache (48 KiB).
Single-pass histograms: All digit histograms built in one scan. Constant digits are skipped entirely.
No more 32 KiB zeroing per bucket: The old 12+12-bit bucket pass paid this for runs of a few hundred elements, dominating sort time between ~50K and ~150K elements.
64-bit MSD+LSD ping-pongs instead of copying each bucket back after every pass.
Float/double radix path: IEEE-754 sign-magnitude mapped to unsigned order via bit inversion. NaN detected during counting pass; nagi::sort falls back to comparisons, nagi::radix_sort throws std::domain_error.
Pre-scan (runs for every n >= 2048)
Two integer counters instead of three bool flags. Vectorizes into pcmpgtd/paddd, 5.9× faster than v5.1.5.
Parallel Radix
Spin barrier synchronizes workers created once, instead of 2×(N−1) std::thread constructions per pass.
Thread-local histogram reuse — no malloc per pass.
Constant digit skipping merged into the single counting scan.
Routing
Data with few adjacent inversions and no narrow value range is sent to comparison sort (introsort finishes nearly-ordered partitions in linear time; radix cannot).
⚠️ Behavioral Changes (v5.1.6)
float/double now allocate an n-element scratch buffer (same as integer paths), whereas v5.1.5 used in-place comparison sort.
nagi::radix_sort throws std::domain_error on NaN (use nagi::sort for silent comparison fallback).
nagi::radix_sort is never rejected by size heuristics — it promises a radix sort and delivers one. (A first-version bug caused 2,856 fuzz cases to silently fall back, caught by case-count drop from 81,060 to 78,204.)
🧠 Design Philosophy
No Magic Thresholds: All thresholds (e.g., SMALL_DATA_THRESHOLD, MSD_LSD_THRESHOLD) are rigorously benchmarked and tunable via -D macros.
Permutation Invariant: All APIs (sort, topk) guarantee that the resulting vector is a strict permutation of the input.
Worst-Case Fallback: A bug in a heuristic should never degrade performance to O(N²). Every adaptive path has a bounded budget and falls back to introsort safely.
🛠️ Tunable Parameters
All constants can be overridden at compile time via -D macros:
Macro 	Default 	Description 
NAGI_INSERTION_SORT_THRESHOLD	64	Switch to insertion sort below this size
NAGI_MAX_NEARLY_SORTED_INVERSION	16	Max adjacent inversions for "nearly sorted" fast path
NAGI_PRESCAN_THRESHOLD	2048	Minimum size for prescan vectorized probe
NAGI_SMALL_DATA_THRESHOLD	128	Use small-data specialized sorter below this
NAGI_PARALLEL_RADIX_MIN	2,000,000	Minimum size for multi-threaded radix
NAGI_MSD_LSD_THRESHOLD	50,000	Switch to MSD+LSD hybrid above this size
👥 The Evolution
Nagi was not built in a day. It evolved through multiple iterations of AI-driven code reviews, property-based fuzzing, and architectural rewrites:
v1.x (Ripple Sort): Python prototype → C++ Introsort + Radix hybrid.
v5.0.0 (Nagi): Introduced the Dispatch → Refine → Strategy layered architecture.
v5.1.4: Fixed the latent partial_insertion_sort partition bug (buried for 4 versions).
v5.1.5: Fixed block_partition_right residual ordering bug and O(N²) nearly-sorted vulnerability.
v5.1.6: Radix engine rewrite, float/double support, 5 bug fixes, 81,060-case validation.
📄 License
MIT License. Feel free to use it, fork it, and make it faster.
