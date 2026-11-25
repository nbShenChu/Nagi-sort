// ============================================================
// nagi.h - Adaptive Sort Framework
// Version: v5.1.6 (Performance: radix engine rewrite + prescan hot path)
// ============================================================
// 凪 —— 往昔的涟漪
// Nagi: The Ripples of Yore
//
// 从 Python 一圈涟漪开始，一路到这里。
// 风停了，海面平静。但水波还记得曾经的涟漪。
//
// Historical note:
//   前身：ripple_sort.h（波纹排序）
//   更名于 v5.0.0，标志从「具体实现」到「自适应框架」的转变。
//
// v5.1.6 (performance / robustness):
//
//   Radix engine
//   - one LSD kernel now serves every key width (8 / 16 / 32 / 64 bits); the
//     digit width is picked per run from a cost model instead of being fixed at
//     11 or 12 bits
//   - every digit histogram is built in a single pass, and a digit whose bits
//     never change is skipped outright (a stable pass over a constant digit is
//     a no-op)
//   - MSD buckets no longer zero a 32 KiB counter table each: the old 12+12-bit
//     bucket pass paid that for runs of a few hundred elements, which dominated
//     the whole sort between roughly 50k and 150k elements
//   - the 64-bit MSD+LSD path ping-pongs instead of copying each bucket back
//     after every one of its seven passes
//   - float and double gained a radix path: IEEE-754 is sign-magnitude, so
//     inverting the bits of negatives and setting the sign bit of positives maps
//     float order onto unsigned order.  NaN is detected during the counting pass;
//     nagi::sort falls back to comparisons and nagi::radix_sort raises
//     std::domain_error instead of ordering garbage
//   - multi-threaded radix creates its workers once and synchronises them with a
//     spin barrier, instead of 2*(N-1) std::thread constructions per pass, and
//     skips constant digits as well
//   - the whole-array digit width is capped at 8 bits once the run no longer fits
//     in L1: 2048-bucket scatters keep 2048 destination lines live, which only
//     pays off while everything is cache resident
//   - counting sort rejects a wide range from a 32-element sample instead of
//     paying for a full min/max pass first
//
//   Pre-scan (runs for every n >= 2048)
//   - two integer counters instead of three flag variables; the loop now
//     vectorises into pcmpgtd/paddd and is 2-6x faster
//
//   Routing
//   - data with few adjacent inversions, and no narrow value range for the
//     counting-sort path, is sent to the comparison sort: introsort finishes
//     nearly-ordered partitions in linear time, a radix sweep cannot
//
//   Correctness / stability
//   - MSD buckets of size 0 or 1 were never copied back from the scratch buffer,
//     leaving the pre-scatter elements in place (silently wrong for any input
//     that produced a singleton bucket)
//   - descending MSD+LSD laid its buckets out back to front but derived the
//     per-bucket ranges front to back, so it sorted the wrong spans
//   - the digit count derived from the varying mask was not clamped to the
//     histogram capacity and could write past the end of it
//   - the histogram table moved off the stack: 48 KiB frames made any caller
//     that inlines several instantiations overflow a 1 MiB stack
//   - radix64_capable_v<double> was a hard compile error rather than false,
//     because `make_unsigned_t` was instantiated before the `&&` short-circuit
//
// v5.1.5 (correctness / contract hotfix):
//   - block_partition_right: residual offsets are consumed in reverse;
//     the pivot is always placed at first - 1 (no special early return)
//   - topk: removed the heap branch that copied a prefix without removing
//     the originals, which duplicated/lost elements; all branches now keep
//     the whole vector a permutation of the input
//   - radix_sort: property-based width traits (long / int64_t / short work
//     on every data model); unsupported types and custom comparators are
//     compile-time errors instead of silent no-ops
//   - nearly-sorted refinement: bounded by insertion moves, not just the
//     number of adjacent inversions
//   - size > INT_MAX throws length_error instead of silently mis-sorting
//   - ARM64 prefetch; LazySortedView rvalue guard and k == 0 fast path
//   - counting sort writes in place (and builds its bitmap online);
//     radix paths use one-scan histograms, ping-pong buffers and skip
//     constant digits
//
// A single-header drop-in sorting framework for integer types.
// Core focus: integers + large data + multi-core.
//
// Design Principles:
//   - Speed:     no abstraction overhead, compile-time dispatch
//   - Stability: every strategy has a worst-case fallback
//   - Robustness: strategies are independent, no shared state
//
// Architecture:
//   L5. Public API      → sort / topk / radix_sort
//   L4. Dispatch Layer  → plan_initial_route (pure function, no data access)
//   L3. Refine Layer    → refine_route (data-aware decision)
//   L2. Feature Probe   → try_prescan (data shape detection)
//   L1. Strategies      → small_sort / radix / counting / introsort
//
// Works on C++17 and C++20.
// ============================================================

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <memory>

// MSVC x86/x64 exposes _mm_prefetch (SSE); MSVC ARM64 exposes __prefetch.
// clang-cl also defines _MSC_VER, but supports __builtin_prefetch everywhere.
#if defined(_MSC_VER) && !defined(__clang__)
#   if defined(_M_ARM64)
#       include <intrin.h>
#       define NAGI_PREFETCH(p) __prefetch(p)
#   else
#       include <intrin.h>
#       include <xmmintrin.h>
#       define NAGI_PREFETCH(p) _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T0)
#   endif
#else
#   define NAGI_PREFETCH(p) __builtin_prefetch(p)
#endif

namespace nagi {
    namespace detail {

        // ============================================================
        // Cross-platform bit utilities
        // ============================================================
        inline int ctz64(unsigned long long x)
        {
            if (x == 0ull) return 64;

#if defined(_MSC_VER)
            unsigned long idx;
#if defined(_M_X64) || defined(_M_ARM64)
            _BitScanForward64(&idx, x);
            return static_cast<int>(idx);
#else
            if (_BitScanForward(&idx, static_cast<unsigned long>(x & 0xFFFFFFFFull)))
                return static_cast<int>(idx);
            _BitScanForward(&idx, static_cast<unsigned long>(x >> 32));
            return static_cast<int>(idx) + 32;
#endif
#else
            return __builtin_ctzll(x);
#endif
        }

        inline int clz64(unsigned long long x)
        {
            if (x == 0ull) return 64;

#if defined(_MSC_VER)
            unsigned long idx;
#if defined(_M_X64) || defined(_M_ARM64)
            _BitScanReverse64(&idx, x);
            return 63 - static_cast<int>(idx);
#else
            if (x >> 32)
            {
                _BitScanReverse(&idx, static_cast<unsigned long>(x >> 32));
                return 31 - static_cast<int>(idx);
            }
            _BitScanReverse(&idx, static_cast<unsigned long>(x));
            return 63 - static_cast<int>(idx);
#endif
#else
            return __builtin_clzll(x);
#endif
        }

        // ============================================================
        // Compile-time constants (all tunable via -D)
        // ============================================================
#ifndef NAGI_INSERTION_SORT_THRESHOLD
#define NAGI_INSERTION_SORT_THRESHOLD 64
#endif
        constexpr int INSERTION_SORT_THRESHOLD = NAGI_INSERTION_SORT_THRESHOLD;

        constexpr int NINTHER_THRESHOLD = 128;

#ifndef NAGI_MAX_NEARLY_SORTED_INVERSION
#define NAGI_MAX_NEARLY_SORTED_INVERSION 16
#endif
        constexpr int MAX_NEARLY_SORTED_INVERSION = NAGI_MAX_NEARLY_SORTED_INVERSION;

#ifndef NAGI_PRESCAN_THRESHOLD
#define NAGI_PRESCAN_THRESHOLD 2048
#endif
        constexpr int PRESCAN_THRESHOLD = NAGI_PRESCAN_THRESHOLD;

        constexpr int PARTIAL_INSERTION_LIMIT = 8;
        constexpr int RADIX_SORT_THRESHOLD = 128;
        constexpr int SORT_NETWORK_THRESHOLD = 5;

#ifndef NAGI_SMALL_DATA_THRESHOLD
#define NAGI_SMALL_DATA_THRESHOLD 128
#endif
        constexpr int SMALL_DATA_THRESHOLD = NAGI_SMALL_DATA_THRESHOLD;

#ifndef NAGI_SMALL_INSERTION_THRESHOLD
#define NAGI_SMALL_INSERTION_THRESHOLD 32
#endif
        constexpr int SMALL_INSERTION_THRESHOLD = NAGI_SMALL_INSERTION_THRESHOLD;

        constexpr int MAX_SMALL_DEPTH = 32;

#ifndef NAGI_PARALLEL_RADIX_MIN
#define NAGI_PARALLEL_RADIX_MIN 2000000
#endif
        constexpr int PARALLEL_RADIX_MIN = NAGI_PARALLEL_RADIX_MIN;

#ifndef NAGI_MSD_LSD_THRESHOLD
#define NAGI_MSD_LSD_THRESHOLD 50000
#endif
        constexpr int MSD_LSD_THRESHOLD = NAGI_MSD_LSD_THRESHOLD;

#ifndef NAGI_QUICK_REJECT_SAMPLE
#define NAGI_QUICK_REJECT_SAMPLE 64
#endif
        constexpr int QUICK_REJECT_SAMPLE = NAGI_QUICK_REJECT_SAMPLE;

        constexpr unsigned long long COUNTING_SORT_RANGE_LIMIT = 1ull << 20;
        constexpr int COUNTING_STACK_BUCKETS = 4096;

        static_assert(COUNTING_STACK_BUCKETS <= 16384,
            "COUNTING_STACK_BUCKETS too large, may cause stack overflow");

        constexpr int PREFETCH_DISTANCE = 16;
        constexpr int PREFETCH_THRESHOLD = 16384;

        // ---- radix engine tuning ----
        // Digit widths in bits.  8 keeps the histogram inside L1 when a run is
        // only a few hundred elements; 12 minimises the number of scatter
        // passes once a run is big enough to amortise a 16 KiB histogram.
        constexpr int RADIX_BITS_MIN = 8;
        constexpr int RADIX_BITS_MAX = 12;
        // Budget for the histogram table (digits * buckets).
        constexpr int RADIX32_HIST_WORDS = 8192;   // 32 KiB, 3 * 2^12 max
        constexpr int RADIX64_HIST_WORDS = 12288;  // 48 KiB, 6 * 2^11 max
        // A run bigger than this cannot keep a wide digit's destination
        // streams in cache, so the digit width is capped at 8 bits.
        constexpr int RADIX_WIDE_MAX_BYTES = 49152;
        // Inside an MSD bucket the histogram competes with a run that is only
        // a few KiB, so a wide digit must clear a much lower bar.
        constexpr int RADIX_BUCKET_WIDE_MAX_BYTES = 8192;
        // n / this > adjacent inversions  =>  treat the data as almost ordered.
        constexpr int NEARLY_SORTED_INV_DIVISOR = 64;
        // Elements per worker thread before extra threads pay for themselves.
        constexpr int PARALLEL_RADIX_CHUNK = 250000;
        // A 64-bit key needs many scatter passes; below this many elements the
        // comparison sort wins (measured crossover on a 1.7 GHz Haswell).
        constexpr int RADIX_MIN_N_WIDE = 1024;

        // Move budget for the "nearly sorted" fast path.  The number of
        // adjacent inversions alone is not a bound on insertion work (a
        // rotated array can have a single inversion and still require
        // O(n^2) moves), so the refine layer must be able to bail out
        // after a small, fixed amount of work.
        constexpr int NEARLY_SORTED_MOVE_BUDGET = 32;
        constexpr unsigned long long COUNTING_RANGE_FACTOR = 16ull;

        // ============================================================
        // Type traits (property based, not spelling based)
        // ============================================================
        // Identifying types by `int` / `long long` is not portable:
        //   - `long` is 32-bit on Windows/LLP64 and 64-bit on LP64;
        //   - `int64_t` is an alias of an implementation-chosen type;
        //   - `short` / `int8_t` etc. were silently unsupported.
        template <typename T>
        constexpr bool is_integral_sortable_v =
            std::is_integral_v<T> &&
            !std::is_same_v<std::remove_cv_t<T>, bool>;

        // `std::make_unsigned_t<T>` is ill-formed for non-integral T, and a
        // variable template still instantiates both operands of `&&`, so
        // `radix64_capable_v<double>` used to be a hard compile error instead
        // of a `false`.  Routing the width through a partial specialisation
        // keeps the traits usable for every T.
        template <typename T, bool = is_integral_sortable_v<T>>
        struct integral_width
        {
            static constexpr int value = 0;
        };

        template <typename T>
        struct integral_width<T, true>
        {
            static constexpr int value =
                std::numeric_limits<std::make_unsigned_t<T>>::digits;
        };

        template <typename T>
        constexpr bool radix32_capable_v =
            (integral_width<T>::value != 0) && (integral_width<T>::value <= 32);

        template <typename T>
        constexpr bool radix64_capable_v =
            (integral_width<T>::value != 0) && (integral_width<T>::value <= 64);

        enum class order_kind {
            ascending,
            descending,
            custom
        };

        template <typename Compare, typename T>
        struct order_of : std::integral_constant<order_kind, order_kind::custom> {};

        template <typename T>
        struct order_of<std::less<T>, T>
            : std::integral_constant<order_kind, order_kind::ascending> {};

        template <typename T>
        struct order_of<std::greater<T>, T>
            : std::integral_constant<order_kind, order_kind::descending> {};

        template <typename T>
        struct order_of<std::less<>, T>
            : std::integral_constant<order_kind, order_kind::ascending> {};

        template <typename T>
        struct order_of<std::greater<>, T>
            : std::integral_constant<order_kind, order_kind::descending> {};

        template <typename Compare, typename T>
        constexpr order_kind order_of_v = order_of<Compare, T>::value;

        template <typename Compare, typename T>
        constexpr bool is_ascending_v = order_of_v<Compare, T> == order_kind::ascending;

        // ------------------------------------------------------------
        // Index range guard
        // ------------------------------------------------------------
        // The algorithms currently use `int` indices.  A silent
        // static_cast<int>(size) for a container with > INT_MAX elements
        // can turn a multi-GB sort into a no-op, so fail loudly instead.
        inline int checked_size(std::size_t size)
        {
            if (size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::length_error(
                    "nagi: container too large for the current 32-bit index type");
            return static_cast<int>(size);
        }

        // ============================================================
        // Sorting networks
        // ============================================================
        template <typename T, typename Compare>
        inline void sort_network_3(T* data, Compare comp)
        {
            auto cswap = [&](int i, int j) { if (comp(data[j], data[i])) std::swap(data[i], data[j]); };
            cswap(0, 2); cswap(0, 1); cswap(1, 2);
        }

        template <typename T, typename Compare>
        inline void sort_network_4(T* data, Compare comp)
        {
            auto cswap = [&](int i, int j) { if (comp(data[j], data[i])) std::swap(data[i], data[j]); };
            cswap(0, 2); cswap(1, 3); cswap(0, 1); cswap(2, 3); cswap(1, 2);
        }

        template <typename T, typename Compare>
        inline void sort_network_5(T* data, Compare comp)
        {
            auto cswap = [&](int i, int j) { if (comp(data[j], data[i])) std::swap(data[i], data[j]); };
            cswap(0, 2); cswap(1, 3); cswap(0, 1); cswap(2, 4);
            cswap(1, 2); cswap(3, 4); cswap(0, 1); cswap(2, 3); cswap(1, 2);
        }

        // ============================================================
        // Insertion sort variants
        // ============================================================
        // 前提：data[low - 1] 必须是整个 [low, high] 区间的最小值（哨兵）。
        // 调用方（三路划分 / block_partition_right）保证父区间的 pivot 在
        // data[low - 1] 位置且小于等于区间内所有元素。
        // 如果将来修改划分逻辑，务必维持这个不变量，否则 while 循环会越界。
        template <typename T, typename Compare>
        void unguarded_insertion_sort(std::vector<T>& data, int low, int high, Compare comp)
        {
            for (int i = low + 1; i <= high; ++i)
            {
                T key = std::move(data[i]);
                int j = i - 1;
                while (comp(key, data[j])) { data[j + 1] = std::move(data[j]); --j; }
                data[j + 1] = std::move(key);
            }
        }
        //partial_insertion_sort 丢元素,这个埋藏了4个大版本的bug 终于被找出来修了
        template <typename T, typename Compare>
        bool partial_insertion_sort(std::vector<T>& data, int low, int high,
            Compare comp, int limit = PARTIAL_INSERTION_LIMIT)
        {
            if (low >= high) return true;
            int moves = 0;
            for (int i = low + 1; i <= high; ++i)
            {
                if (!comp(data[i], data[i - 1])) continue;   // 已就位，跳过
                T key = std::move(data[i]);
                int j = i - 1;
                while (j >= low && comp(key, data[j]))
                {
                    data[j + 1] = std::move(data[j]);
                    --j;
                }
                data[j + 1] = std::move(key);
                moves += i - 1 - j;
                if (moves > limit) return false;             // 此时数据完整
            }
            return true;
        }

        // Linear insertion sort (faster than binary for n < 64)
        template <typename T, typename Compare>
        void linear_insertion_sort(std::vector<T>& data, int low, int high, Compare comp)
        {
            for (int i = low + 1; i <= high; ++i)
            {
                T key = std::move(data[i]);
                int j = i - 1;
                while (j >= low && comp(key, data[j]))
                {
                    data[j + 1] = std::move(data[j]);
                    --j;
                }
                data[j + 1] = std::move(key);
            }
        }

        template <typename T, typename Compare>
        inline void sort_small_range(std::vector<T>& data, int low, int high, Compare comp)
        {
            int len = high - low + 1;
            if (len <= 1) return;
            switch (len)
            {
            case 2: if (comp(data[high], data[low])) std::swap(data[low], data[high]); break;
            case 3: sort_network_3(data.data() + low, comp); break;
            case 4: sort_network_4(data.data() + low, comp); break;
            case 5: sort_network_5(data.data() + low, comp); break;
            default: linear_insertion_sort(data, low, high, comp); break;
            }
        }

        template <typename T, typename Compare>
        void insertion_sort(std::vector<T>& data, int low, int high, Compare comp)
        {
            for (int i = low + 1; i <= high; ++i)
            {
                T key = std::move(data[i]);
                int left = low, right = i;
                while (left < right)
                {
                    int mid = left + (right - left) / 2;
                    if (comp(key, data[mid])) right = mid;
                    else left = mid + 1;
                }
                for (int j = i; j > left; --j) data[j] = std::move(data[j - 1]);
                data[left] = std::move(key);
            }
        }

        // ============================================================
        // Median of three
        // ============================================================

        template <typename T, typename Compare>
        inline void median_swap3(std::vector<T>& data, int a, int b, int c, Compare comp)
        {
            if (comp(data[b], data[a])) std::swap(data[a], data[b]);
            if (comp(data[c], data[b])) std::swap(data[b], data[c]);
            if (comp(data[b], data[a])) std::swap(data[a], data[b]);
        }

        // ============================================================
        // Block Partition (BlockQuicksort-inspired, two-way)
        // ============================================================
        template <typename T, typename Compare>
        inline int block_partition_right(std::vector<T>& data, int low, int high, Compare comp)
        {
            constexpr int BLOCK = 64;

            T pivot = std::move(data[low]);

            int first = low;
            int last = high + 1;

            while (comp(data[++first], pivot)) {}
            if (first - 1 == low)
                while (first < last && !comp(data[--last], pivot)) {}
            else
                while (!comp(data[--last], pivot)) {}

            // The pivot's final position is always first - 1, including the
            // already-partitioned case.  Returning `last` here would put the
            // pivot one slot too far right and leak a >= pivot element into
            // the left partition.
            if (first < last)
            {
                std::swap(data[first], data[last]);
                ++first;

                alignas(64) unsigned char offsets_l[BLOCK + 64];
                alignas(64) unsigned char offsets_r[BLOCK + 64];

                int l_base = first;
                int r_base = last;
                int l_cnt = 0, r_cnt = 0;
                int l_start = 0, r_start = 0;

                while (first < last)
                {
                    int avail = last - first;
                    int left_take = (l_cnt == 0) ? (r_cnt == 0 ? avail / 2 : avail) : 0;
                    int right_take = (r_cnt == 0) ? (avail - left_take) : 0;

                    if (left_take >= BLOCK)
                    {
                        for (int i = 0; i < BLOCK; )
                        {
                            offsets_l[l_cnt] = (unsigned char)i++; l_cnt += !comp(data[first], pivot); ++first;
                            offsets_l[l_cnt] = (unsigned char)i++; l_cnt += !comp(data[first], pivot); ++first;
                            offsets_l[l_cnt] = (unsigned char)i++; l_cnt += !comp(data[first], pivot); ++first;
                            offsets_l[l_cnt] = (unsigned char)i++; l_cnt += !comp(data[first], pivot); ++first;
                            offsets_l[l_cnt] = (unsigned char)i++; l_cnt += !comp(data[first], pivot); ++first;
                            offsets_l[l_cnt] = (unsigned char)i++; l_cnt += !comp(data[first], pivot); ++first;
                            offsets_l[l_cnt] = (unsigned char)i++; l_cnt += !comp(data[first], pivot); ++first;
                            offsets_l[l_cnt] = (unsigned char)i++; l_cnt += !comp(data[first], pivot); ++first;
                        }
                    }
                    else
                    {
                        for (int i = 0; i < left_take; )
                        {
                            offsets_l[l_cnt] = (unsigned char)i++; l_cnt += !comp(data[first], pivot); ++first;
                        }
                    }

                    if (right_take >= BLOCK)
                    {
                        for (int i = 0; i < BLOCK; )
                        {
                            offsets_r[r_cnt] = (unsigned char)++i; r_cnt += comp(data[--last], pivot);
                            offsets_r[r_cnt] = (unsigned char)++i; r_cnt += comp(data[--last], pivot);
                            offsets_r[r_cnt] = (unsigned char)++i; r_cnt += comp(data[--last], pivot);
                            offsets_r[r_cnt] = (unsigned char)++i; r_cnt += comp(data[--last], pivot);
                            offsets_r[r_cnt] = (unsigned char)++i; r_cnt += comp(data[--last], pivot);
                            offsets_r[r_cnt] = (unsigned char)++i; r_cnt += comp(data[--last], pivot);
                            offsets_r[r_cnt] = (unsigned char)++i; r_cnt += comp(data[--last], pivot);
                            offsets_r[r_cnt] = (unsigned char)++i; r_cnt += comp(data[--last], pivot);
                        }
                    }
                    else
                    {
                        for (int i = 0; i < right_take; )
                        {
                            offsets_r[r_cnt] = (unsigned char)++i; r_cnt += comp(data[--last], pivot);
                        }
                    }

                    int n = (l_cnt < r_cnt) ? l_cnt : r_cnt;
                    for (int i = 0; i < n; ++i)
                        std::swap(data[l_base + offsets_l[l_start + i]],
                            data[r_base - offsets_r[r_start + i]]);
                    l_cnt -= n; r_cnt -= n;
                    l_start += n; r_start += n;

                    if (l_cnt == 0) { l_start = 0; l_base = first; }
                    if (r_cnt == 0) { r_start = 0; r_base = last; }
                }

                // Residual offsets must be consumed in reverse order.  The
                // destination cursor (--last / first++) advances, while the
                // source offsets are relative to a fixed base; consuming them
                // front-to-back can swap a slot twice and break the partition
                // invariant.  This is part of BlockQuicksort's correctness,
                // not a micro-optimization.
                if (l_cnt)
                {
                    while (l_cnt)
                    {
                        --l_cnt;
                        std::swap(data[l_base + offsets_l[l_start + l_cnt]], data[--last]);
                    }
                    first = last;
                }
                if (r_cnt)
                {
                    while (r_cnt)
                    {
                        --r_cnt;
                        std::swap(data[r_base - offsets_r[r_start + r_cnt]], data[first++]);
                    }
                    last = first;
                }
            }

            int pivot_pos = first - 1;
            data[low] = std::move(data[pivot_pos]);
            data[pivot_pos] = std::move(pivot);
            return pivot_pos;
        }

        // ============================================================
        // Radix keys: order-preserving unsigned images of T
        // ============================================================
        // Integers: flipping the sign bit maps signed order onto unsigned.
        // IEEE-754: the representation is sign-magnitude, so inverting every
        // bit of a negative value and setting the sign bit of a positive one
        // maps float order onto unsigned order.  NaN has no place in a total
        // order, so it is detected during the histogram pass and reported;
        // the caller then falls back to a comparison sort.
        template <typename T, bool = is_integral_sortable_v<T>>
        struct key_traits
        {
            // Not radix-sortable.  Members exist so that dependent code can be
            // written without hard errors; supported == false keeps it unused.
            using U = std::uint64_t;
            static constexpr bool supported = false;
            static constexpr int  bits = 0;
            static U raw(T) noexcept { return 0; }
            static U fold(U u) noexcept { return u; }
            static bool bad(U) noexcept { return false; }
        };

        template <typename T>
        struct key_traits<T, true>
        {
            using U = std::make_unsigned_t<T>;
            static constexpr bool supported = true;
            static constexpr int  bits = std::numeric_limits<U>::digits;
            static U raw(T x) noexcept { return static_cast<U>(x); }
            static U fold(U u) noexcept
            {
                if constexpr (std::is_signed_v<T>)
                    return static_cast<U>(u ^ static_cast<U>(U(1) << (bits - 1)));
                else
                    return u;
            }
            static bool bad(U) noexcept { return false; }
        };

        template <>
        struct key_traits<float, false>
        {
            using U = std::uint32_t;
            static constexpr bool supported = true;
            static constexpr int  bits = 32;
            static U raw(float x) noexcept { U u; std::memcpy(&u, &x, sizeof(u)); return u; }
            static U fold(U u) noexcept
            {
                return static_cast<U>(u ^ static_cast<U>((U(0) - (u >> 31)) | 0x80000000u));
            }
            static bool bad(U u) noexcept { return (u & 0x7FFFFFFFu) > 0x7F800000u; }
        };

        template <>
        struct key_traits<double, false>
        {
            using U = std::uint64_t;
            static constexpr bool supported = true;
            static constexpr int  bits = 64;
            static U raw(double x) noexcept { U u; std::memcpy(&u, &x, sizeof(u)); return u; }
            static U fold(U u) noexcept
            {
                return static_cast<U>(u ^ static_cast<U>((U(0) - (u >> 63)) | 0x8000000000000000ull));
            }
            static bool bad(U u) noexcept { return (u & 0x7FFFFFFFFFFFFFFFull) > 0x7FF0000000000000ull; }
        };

        template <typename T>
        constexpr bool radixable_v =
            key_traits<T>::supported && (key_traits<T>::bits <= 64);

        // ============================================================
        // Counting sort (output written in place, bitmap built online)
        // ============================================================
        // Once the frequencies are known the original order is worthless,
        // so the output can overwrite `data` directly.  This removes the
        // extra N-element buffer and one full allocation.
        template <typename T>
        inline T counting_bucket_value(T minV, unsigned int b)
        {
            using U = std::make_unsigned_t<T>;
            return static_cast<T>(static_cast<U>(minV) + static_cast<U>(b));
        }

        template <typename T>
        void counting_sort_range_vec(std::vector<T>& data, T minV, unsigned int range, bool descending = false)
        {
            const int n = checked_size(data.size());
            const int SIZE = static_cast<int>(range);
            const unsigned int umin = static_cast<unsigned int>(minV);

            std::vector<int> cnt(static_cast<std::size_t>(SIZE), 0);
            const int WORDS = (SIZE + 63) >> 6;
            std::vector<unsigned long long> bitmap(static_cast<std::size_t>(WORDS), 0ull);
            int nonEmpty = 0;

            // Build both the frequencies and the non-empty bitmap in the
            // same pass.  The old code scanned all SIZE buckets a second
            // time just to build the bitmap, which defeated its purpose.
            for (int i = 0; i < n; ++i)
            {
                const unsigned int idx = static_cast<unsigned int>(data[i]) - umin;
                int& c = cnt[idx];
                if (c++ == 0)
                {
                    bitmap[idx >> 6] |= 1ull << (idx & 63);
                    ++nonEmpty;
                }
            }

            // 稠密/稀疏分界：非空桶 > 75% 时，bitmap 遍历的额外开销
            // 不划算，走稠密顺序扫描。
            constexpr int DENSE_NUM = 3;
            constexpr int DENSE_DEN = 4;
            const bool dense = (nonEmpty * DENSE_DEN > SIZE * DENSE_NUM);

            int pos = 0;
            if (!descending)
            {
                if (dense)
                {
                    for (int b = 0; b < SIZE; ++b)
                    {
                        T val = counting_bucket_value(minV, static_cast<unsigned int>(b));
                        int c = cnt[b];
                        while (c--) data[pos++] = val;
                    }
                }
                else
                {
                    for (int w = 0; w < WORDS; ++w)
                    {
                        unsigned long long word = bitmap[w];
                        while (word)
                        {
                            int bit = ctz64(word);
                            int b = (w << 6) | bit;
                            T val = counting_bucket_value(minV, static_cast<unsigned int>(b));
                            int c = cnt[b];
                            while (c--) data[pos++] = val;
                            word &= word - 1;
                        }
                    }
                }
            }
            else
            {
                if (dense)
                {
                    for (int b = SIZE - 1; b >= 0; --b)
                    {
                        T val = counting_bucket_value(minV, static_cast<unsigned int>(b));
                        int c = cnt[b];
                        while (c--) data[pos++] = val;
                    }
                }
                else
                {
                    for (int w = WORDS - 1; w >= 0; --w)
                    {
                        unsigned long long word = bitmap[w];
                        while (word)
                        {
                            int bit = 63 - clz64(word);
                            int b = (w << 6) | bit;
                            T val = counting_bucket_value(minV, static_cast<unsigned int>(b));
                            int c = cnt[b];
                            while (c--) data[pos++] = val;
                            word &= ~(1ull << bit);
                        }
                    }
                }
            }
        }

        template <typename T>
        void counting_sort_range_stack(std::vector<T>& data, T minV, unsigned int range, bool descending = false)
        {
            const int n = checked_size(data.size());
            const int SIZE = static_cast<int>(range);
            const unsigned int umin = static_cast<unsigned int>(minV);

            int cnt[COUNTING_STACK_BUCKETS];
            for (int i = 0; i < SIZE; ++i) cnt[i] = 0;

            for (int i = 0; i < n; ++i)
                ++cnt[static_cast<unsigned int>(data[i]) - umin];

            int pos = 0;
            if (!descending)
            {
                for (int b = 0; b < SIZE; ++b)
                {
                    T val = counting_bucket_value(minV, static_cast<unsigned int>(b));
                    int c = cnt[b];
                    while (c--) data[pos++] = val;
                }
            }
            else
            {
                for (int b = SIZE - 1; b >= 0; --b)
                {
                    T val = counting_bucket_value(minV, static_cast<unsigned int>(b));
                    int c = cnt[b];
                    while (c--) data[pos++] = val;
                }
            }
        }

        // Sampled feasibility test for the counting-sort path.  Sampling may
        // only reject, so a wrong guess can never produce a wrong sort; it just
        // lets the dispatcher skip the full min/max scan for wide-range input.
        template <typename T>
        bool counting_sort_possible(const std::vector<T>& data, int n)
        {
            if constexpr (!is_integral_sortable_v<T>)
            {
                (void)data; (void)n;
                return false;
            }
            else
            {
                if (n <= 256) return true;
                constexpr int SAMPLES = 32;
                const int step = n / SAMPLES;
                T smin = data[0], smax = data[0];
                for (int i = step; i < n; i += step)
                {
                    if (data[i] < smin) smin = data[i];
                    if (data[i] > smax) smax = data[i];
                }
                const unsigned long long sdiff =
                    static_cast<unsigned long long>(smax) -
                    static_cast<unsigned long long>(smin);
                if (sdiff >= COUNTING_SORT_RANGE_LIMIT) return false;
                if (sdiff + 1ull > static_cast<unsigned long long>(n) * COUNTING_RANGE_FACTOR)
                    return false;
                return true;
            }
        }

        template <typename T>
        bool try_counting_sort(std::vector<T>& data, bool descending = false)
        {
            if constexpr (!is_integral_sortable_v<T>)
                return false;
            else
            {
                const int n = checked_size(data.size());
                if (n <= 1) return true;

                // Cheap rejection first.  A full min/max scan is a whole pass
                // over the array and the range test fails for every wide-range
                // input, which is the common case.  Sampling may only *reject*
                // (never accept), so a wrong guess can never produce a wrong
                // sort.  The samples are strided, so they touch few cachelines.
                if (!counting_sort_possible(data, n)) return false;

                T minV = data[0], maxV = data[0];
                for (int i = 1; i < n; ++i)
                {
                    if (data[i] < minV) minV = data[i];
                    if (data[i] > maxV) maxV = data[i];
                }

                // Compute the difference without +1 first; for a full
                // 64-bit domain `diff + 1` would wrap to zero and could
                // incorrectly pass the range test.
                const unsigned long long diff =
                    static_cast<unsigned long long>(maxV) -
                    static_cast<unsigned long long>(minV);

                if (diff >= COUNTING_SORT_RANGE_LIMIT) return false;
                const unsigned long long rangeLL = diff + 1ull;
                if (rangeLL > static_cast<unsigned long long>(n) * COUNTING_RANGE_FACTOR)
                    return false;

                if (rangeLL == 1) return true;

                const unsigned int range = static_cast<unsigned int>(rangeLL);
                if (range <= COUNTING_STACK_BUCKETS)
                    counting_sort_range_stack(data, minV, range, descending);
                else
                    counting_sort_range_vec(data, minV, range, descending);
                return true;
            }
        }

        // ============================================================
        // LSD radix engine (one implementation for every key width)
        // ============================================================
        // Everything below shares a single scatter kernel:
        //
        //   * all digit histograms are built in ONE pass over the input,
        //   * a digit that does not vary is skipped (a stable pass over a
        //     constant digit cannot change the order),
        //   * the digit width is chosen at run time from the number of
        //     elements, because a 2048-bucket histogram (8 KiB) is pure
        //     overhead when the whole run is a few hundred elements,
        //   * the two buffers ping-pong, so a pass costs exactly n reads and
        //     n writes with no extra copy.
        inline int bit_length(unsigned long long u) noexcept
        {
            return (u == 0ull) ? 0 : (64 - clz64(u));
        }

        // Pick the digit width for `total_bits` of key material over `m`
        // elements of `elem_bytes` bytes each.
        //
        // Two effects fight each other:
        //   * a wider digit means fewer scatter passes but a bigger histogram,
        //     and the histogram is zeroed and prefix-summed on every pass, so
        //     for a few hundred elements the bookkeeping dominates;
        //   * a wide digit also means many "open" destination streams.  With
        //     2048 buckets the scatter keeps 2048 cache lines live, which only
        //     works while the whole run sits in L1; past that the stores miss
        //     constantly and one extra pass of a 256-bucket digit is cheaper.
        //
        // Hence: narrow digits when the run is either tiny or larger than L1.
        inline int choose_radix_bits(int total_bits, int m, int elem_bytes, int hist_cap,
            int wide_max_bytes = RADIX_WIDE_MAX_BYTES) noexcept
        {
            const long long run_bytes = static_cast<long long>(m) * elem_bytes;
            const int maxb = (run_bytes <= wide_max_bytes) ? RADIX_BITS_MAX : RADIX_BITS_MIN;

            int best = RADIX_BITS_MIN;
            long long best_cost = std::numeric_limits<long long>::max();
            for (int b = RADIX_BITS_MIN; b <= maxb; ++b)
            {
                const int d = (total_bits + b - 1) / b;
                const long long buckets = 1LL << b;
                if (static_cast<long long>(d) * buckets > static_cast<long long>(hist_cap))
                    continue;
                // 11 units per element and pass (a read plus a write), 2 units
                // per histogram slot (zero + prefix sum + scatter cursor).
                const long long cost = static_cast<long long>(d) *
                    (11LL * static_cast<long long>(m) + 2LL * buckets);
                if (cost < best_cost)
                {
                    best_cost = cost;
                    best = b;
                }
            }
            return best;
        }

        // A very small MSD bucket is cheaper to finish with insertion sort than
        // with a multi-pass radix that has to rebuild its histogram table.  The
        // break-even point grows with the number of remaining digits, so it is
        // derived from the same cost model instead of being a magic constant.
        inline bool bucket_prefers_insertion(int m, int total_bits, int elem_bytes, int hist_cap,
            int wide_max_bytes = RADIX_BUCKET_WIDE_MAX_BYTES)
        {
            if (m < 2) return true;
            const int bits = choose_radix_bits(total_bits, m, elem_bytes, hist_cap, wide_max_bytes);
            const int d = (total_bits + bits - 1) / bits;
            const long long ins = 3LL * m * m / 4;
            const long long rad = static_cast<long long>(d) * (11LL * m + 2LL * (1LL << bits));
            return ins < rad;
        }

        // Scratch histogram table.
        //
        // Kept on the heap (a 48 KiB frame would make any caller that inlines
        // several sort instantiations overflow a 1 MiB stack) but retained per
        // thread, so a sort never pays for a fresh allocation either.  Only one
        // radix pass is ever live at a time, so sharing one buffer is safe.
        inline int* radix_hist_scratch(int words)
        {
            static thread_local std::vector<int> buffer;
            if (static_cast<int>(buffer.size()) < words)
                buffer.resize(static_cast<std::size_t>(words));
            return buffer.data();
        }

        // Build every digit histogram in a single pass.  Both the digit count
        // and the digit width are compile-time constants, so every shift and
        // every histogram offset is a constant: the obvious alternative (a
        // running `t >>= bits`) serialises on the shifter for `D` cycles and
        // costs ~13 cycles per element on 64-bit keys.
        template <int D, int BITS, typename T, typename KT>
        inline void radix_count_digits(const T* src, int m, int* hist,
            typename KT::U& varying, bool& bad)
        {
            using U = typename KT::U;
            constexpr int BUCKETS = 1 << BITS;
            constexpr U MASK = static_cast<U>(BUCKETS - 1);
            const U first = KT::fold(KT::raw(src[0]));
            const bool do_prefetch = (m > PREFETCH_THRESHOLD);

            for (int i = 0; i < m; ++i)
            {
                if (do_prefetch && i + PREFETCH_DISTANCE < m)
                    NAGI_PREFETCH(&src[i + PREFETCH_DISTANCE]);
                const U r = KT::raw(src[i]);
                bad |= KT::bad(r);
                const U k = KT::fold(r);
                varying |= static_cast<U>(k ^ first);
                for (int g = 0; g < D; ++g)
                    ++hist[g * BUCKETS + static_cast<int>((k >> (g * BITS)) & MASK)];
            }
        }

        template <int D, typename T, typename KT>
        inline void radix_count_bits(const T* src, int m, int bits, int* hist,
            typename KT::U& varying, bool& bad)
        {
            switch (bits)
            {
            case 8:  radix_count_digits<D, 8, T, KT>(src, m, hist, varying, bad); break;
            case 9:  radix_count_digits<D, 9, T, KT>(src, m, hist, varying, bad); break;
            case 10: radix_count_digits<D, 10, T, KT>(src, m, hist, varying, bad); break;
            case 11: radix_count_digits<D, 11, T, KT>(src, m, hist, varying, bad); break;
            default: radix_count_digits<D, 12, T, KT>(src, m, hist, varying, bad); break;
            }
        }

        template <typename T, typename KT>
        inline void radix_count_dispatch(const T* src, int m, int bits, int /*buckets*/,
            int digits, int* hist, typename KT::U& varying, bool& bad)
        {
            switch (digits)
            {
            case 1: radix_count_bits<1, T, KT>(src, m, bits, hist, varying, bad); break;
            case 2: radix_count_bits<2, T, KT>(src, m, bits, hist, varying, bad); break;
            case 3: radix_count_bits<3, T, KT>(src, m, bits, hist, varying, bad); break;
            case 4: radix_count_bits<4, T, KT>(src, m, bits, hist, varying, bad); break;
            case 5: radix_count_bits<5, T, KT>(src, m, bits, hist, varying, bad); break;
            case 6: radix_count_bits<6, T, KT>(src, m, bits, hist, varying, bad); break;
            case 7: radix_count_bits<7, T, KT>(src, m, bits, hist, varying, bad); break;
            default: radix_count_bits<8, T, KT>(src, m, bits, hist, varying, bad); break;
            }
        }

        // LSD sort of `m` elements by the low `total_bits` bits of the folded
        // key.  The result ends up in `src` or `dst`; the returned pointer
        // says which.  Returns nullptr when a NaN is present (floats only).
        template <typename T, typename KT>
        T* radix_lsd(T* src, int m, T* dst, int total_bits, bool descending,
            int* hist, int hist_cap, int wide_max_bytes = RADIX_WIDE_MAX_BYTES)
        {
            using U = typename KT::U;
            if (m <= 1) return src;

            const int bits = choose_radix_bits(total_bits, m, static_cast<int>(sizeof(T)),
                hist_cap, wide_max_bytes);
            const int buckets = 1 << bits;
            const int digits = (total_bits + bits - 1) / bits;
            const U mask = static_cast<U>(buckets - 1);

            std::memset(hist, 0, sizeof(int) * static_cast<std::size_t>(digits) * buckets);

            U varying = 0;
            bool bad = false;
            radix_count_dispatch<T, KT>(src, m, bits, buckets, digits, hist, varying, bad);

            if (bad) return nullptr;         // NaN: caller must use comparisons
            if (varying == 0) return src;    // every key equal: already sorted

            // Digits above the highest varying bit hold nothing.
            // The varying mask can carry bits above `total_bits` when the
            // caller hands us a range that is not a real key prefix; the digit
            // count must never exceed what the histogram was sized for.
            const int used_digits = std::min(digits, (bit_length(varying) + bits - 1) / bits);

            T* a = src;
            T* b = dst;
            int shift = 0;
            for (int g = 0; g < used_digits; ++g, shift += bits)
            {
                int* p = hist + static_cast<std::size_t>(g) * buckets;

                if (((varying >> shift) & mask) == 0) continue;   // constant digit

                if (!descending)
                {
                    int s = 0;
                    for (int k2 = 0; k2 < buckets; ++k2) { const int c = p[k2]; p[k2] = s; s += c; }
                }
                else
                {
                    int s = 0;
                    for (int k2 = buckets - 1; k2 >= 0; --k2) { const int c = p[k2]; p[k2] = s; s += c; }
                }

                const bool do_prefetch = (m > PREFETCH_THRESHOLD);
                for (int i = 0; i < m; ++i)
                {
                    if (do_prefetch && i + PREFETCH_DISTANCE < m)
                        NAGI_PREFETCH(&a[i + PREFETCH_DISTANCE]);
                    const U k = KT::fold(KT::raw(a[i]));
                    b[p[static_cast<int>((k >> shift) & mask)]++] = a[i];
                }
                T* tmp = a; a = b; b = tmp;
            }
            return a;
        }

        // Insertion sort for the remains of an MSD bucket.
        template <typename T>
        void bucket_insertion_sort(T* a, int m, bool descending)
        {
            if (m < 2) return;
            if (!descending)
            {
                for (int i = 1; i < m; ++i)
                {
                    const T key = a[i];
                    int j = i - 1;
                    while (j >= 0 && key < a[j]) { a[j + 1] = a[j]; --j; }
                    a[j + 1] = key;
                }
            }
            else
            {
                for (int i = 1; i < m; ++i)
                {
                    const T key = a[i];
                    int j = i - 1;
                    while (j >= 0 && key > a[j]) { a[j + 1] = a[j]; --j; }
                    a[j + 1] = key;
                }
            }
        }

        // Whole-array LSD (used when the input fits in cache, or the key
        // range is narrow enough that an MSD split would not pay off).
        template <typename T, typename KT>
        bool radix_sort_whole(std::vector<T>& data, bool descending)
        {
            constexpr int BITS = KT::bits;
            constexpr int CAP = (BITS <= 32) ? RADIX32_HIST_WORDS : RADIX64_HIST_WORDS;
            const int n = checked_size(data.size());
            if (n <= 1) return true;

            std::vector<T> buffer(static_cast<std::size_t>(n));
            int* hist = radix_hist_scratch(CAP);

            T* r = radix_lsd<T, KT>(data.data(), n, buffer.data(), BITS, descending,
                hist, CAP);
            if (!r) return false;
            if (r != data.data())
                std::memcpy(data.data(), r, sizeof(T) * static_cast<std::size_t>(n));
            return true;
        }

        // MSD bucket split on the top byte, then one cache-resident LSD sort
        // per bucket.  This is what keeps multi-megabyte inputs from turning
        // every scatter pass into a full memory round trip.
        template <typename T, typename KT>
        bool radix_sort_msd_lsd(std::vector<T>& data, bool descending)
        {
            using U = typename KT::U;
            constexpr int BITS = KT::bits;
            constexpr int MSDB = 8;
            constexpr int MSDN = 1 << MSDB;
            constexpr int TOP_SHIFT = BITS - MSDB;
            constexpr int CAP = (BITS <= 32) ? RADIX32_HIST_WORDS : RADIX64_HIST_WORDS;

            const int n = checked_size(data.size());
            if (n <= 1) return true;

            T* d = data.data();
            std::vector<T> scratch(static_cast<std::size_t>(n));
            T* sc = scratch.data();

            int cnt[MSDN] = { 0 };
            U varying = 0;
            bool bad = false;
            {
                const U first = KT::fold(KT::raw(d[0]));
                const bool do_prefetch = (n > PREFETCH_THRESHOLD);
                for (int i = 0; i < n; ++i)
                {
                    if (do_prefetch && i + PREFETCH_DISTANCE < n)
                        NAGI_PREFETCH(&d[i + PREFETCH_DISTANCE]);
                    const U r = KT::raw(d[i]);
                    bad |= KT::bad(r);
                    const U k = KT::fold(r);
                    varying |= static_cast<U>(k ^ first);
                    ++cnt[static_cast<int>(k >> TOP_SHIFT)];
                }
            }
            if (bad) return false;
            if (varying == 0) return true;

            // The top byte is constant: a split would put everything in one
            // bucket, so go straight to a whole-array LSD instead of paying
            // for a scatter that changes nothing.
            if (((varying >> TOP_SHIFT) & static_cast<U>(0xFF)) == 0)
                return radix_sort_whole<T, KT>(data, descending);

            // Bucket b lands at bstart[b].  For descending order the buckets are
            // laid out back to front, so the per-bucket pass must use this table
            // instead of accumulating cnt[] in ascending order.
            int pos[MSDN];
            int bstart[MSDN];
            if (!descending)
            {
                int s = 0;
                for (int b = 0; b < MSDN; ++b) { const int c = cnt[b]; pos[b] = s; s += c; }
            }
            else
            {
                int s = 0;
                for (int b = MSDN - 1; b >= 0; --b) { const int c = cnt[b]; pos[b] = s; s += c; }
            }
            for (int b = 0; b < MSDN; ++b) bstart[b] = pos[b];

            for (int i = 0; i < n; ++i)
            {
                const U k = KT::fold(KT::raw(d[i]));
                sc[pos[static_cast<int>(k >> TOP_SHIFT)]++] = d[i];
            }

            int* hist = radix_hist_scratch(CAP);
            for (int b = 0; b < MSDN; ++b)
            {
                const int m = cnt[b];
                T* bsrc = sc + bstart[b];
                T* bdst = d + bstart[b];

                // Every bucket must be copied back, including the singletons:
                // `d` still holds the pre-scatter data at this position.
                if (m <= 1) { if (m == 1) bdst[0] = bsrc[0]; continue; }

                if (bucket_prefers_insertion(m, TOP_SHIFT, static_cast<int>(sizeof(T)), CAP))
                {
                    bucket_insertion_sort(bsrc, m, descending);
                    std::memcpy(bdst, bsrc, sizeof(T) * static_cast<std::size_t>(m));
                }
                else
                {
                    T* r = radix_lsd<T, KT>(bsrc, m, bdst, TOP_SHIFT, descending, hist, CAP,
                        RADIX_BUCKET_WIDE_MAX_BYTES);
                    if (!r) return false;
                    if (r != bdst)
                        std::memcpy(bdst, r, sizeof(T) * static_cast<std::size_t>(m));
                }
            }
            return true;
        }

        // ============================================================
        // Multi-threaded 8-bit radix sort
        // ============================================================
        // All worker threads are created once and synchronised with a
        // spin barrier; the previous version created and joined N-1 threads
        // (and copied the thread position table) on every counting and every
        // scattering phase, which on a two-core laptop is a sizeable fraction
        // of the whole sort.
        class spin_barrier
        {
        public:
            explicit spin_barrier(unsigned total) : m_total(total), m_count(0), m_gen(0) {}

            void wait()
            {
                const unsigned g = m_gen.load(std::memory_order_acquire);
                if (m_count.fetch_add(1, std::memory_order_acq_rel) + 1 == m_total)
                {
                    m_count.store(0, std::memory_order_relaxed);
                    m_gen.store(g + 1, std::memory_order_release);
                }
                else
                {
                    unsigned spins = 0;
                    while (m_gen.load(std::memory_order_acquire) == g)
                    {
                        if (++spins > 2048) std::this_thread::yield();
                    }
                }
            }

        private:
            unsigned m_total;
            std::atomic<unsigned> m_count;
            std::atomic<unsigned> m_gen;
        };

        template <typename T>
        bool radix_sort_parallel(std::vector<T>& data, bool descending)
        {
            using KT = key_traits<T>;
            using U = typename KT::U;
            constexpr int BITS = KT::bits;
            constexpr int PASSES = (BITS + 7) / 8;
            static_assert(BITS % 8 == 0 && BITS >= 24,
                "parallel radix expects byte-aligned keys of at least 24 bits");

            const int n = checked_size(data.size());
            if (n <= 1) return true;

            unsigned int nt = std::thread::hardware_concurrency();
            if (nt == 0) nt = 4;
            if (nt > 16) nt = 16;
            unsigned int maxByData = static_cast<unsigned int>(n / PARALLEL_RADIX_CHUNK);
            if (maxByData < 1) maxByData = 1;
            if (nt > maxByData) nt = maxByData;
            if (nt <= 1) return radix_sort_msd_lsd<T, KT>(data, descending);

            struct alignas(64) AlignedBucket { std::array<int, 256> data; };
            // One cache line per thread: both fields plus explicit padding, so
            // the workers never share a line while publishing their key masks.
            struct alignas(64) AlignedVary
            {
                U andv;
                U orv;
                unsigned char pad[64 - 2 * sizeof(U)];
            };

            std::vector<T> buffer(static_cast<std::size_t>(n));
            const int blockSize = static_cast<int>(
                (static_cast<long long>(n) + static_cast<long long>(nt) - 1) /
                static_cast<long long>(nt));

            auto block_bounds = [&](unsigned int t, int& start, int& end) {
                long long s = static_cast<long long>(t) * static_cast<long long>(blockSize);
                long long e = (t + 1 == nt) ? n : s + blockSize;
                if (s > n) s = n;
                if (e > n) e = n;
                start = static_cast<int>(s);
                end = static_cast<int>(e);
            };

            std::vector<AlignedBucket> localCount(nt);
            std::vector<AlignedBucket> threadPos(nt);
            std::vector<AlignedVary> threadVary(nt);
            std::array<int, 256> bucketStart;
            std::array<unsigned char, (std::size_t)PASSES> live;
            live.fill(1);

            const bool do_prefetch = (n > PREFETCH_THRESHOLD);
            spin_barrier bar(nt);

            auto worker = [&](unsigned int t) {
                int start, end;
                block_bounds(t, start, end);

                for (int pass = 0; pass < PASSES; ++pass)
                {
                    const int shift = pass * 8;
                    // Pass 0 always counts: that scan is what tells us which
                    // digits actually vary, so every later constant digit can
                    // be dropped entirely.
                    const bool count_this = (pass == 0) || (live[(std::size_t)pass] != 0);

                    if (count_this)
                    {
                        auto& lc = localCount[t].data;
                        lc.fill(0);
                        U av = ~static_cast<U>(0);
                        U ov = 0;
                        for (int i = start; i < end; ++i)
                        {
                            if (do_prefetch && i + PREFETCH_DISTANCE < end)
                                NAGI_PREFETCH(&data[i + PREFETCH_DISTANCE]);
                            const U k = KT::fold(KT::raw(data[i]));
                            ++lc[static_cast<std::size_t>((k >> shift) & 0xFF)];
                            av &= k; ov |= k;
                        }
                        threadVary[t].andv = av;
                        threadVary[t].orv = ov;
                    }
                    bar.wait();

                    if (t == 0)
                    {
                        if (pass == 0)
                        {
                            U av = ~static_cast<U>(0);
                            U ov = 0;
                            for (unsigned int u = 0; u < nt; ++u)
                            {
                                av &= threadVary[u].andv;
                                ov |= threadVary[u].orv;
                            }
                            const U varying = static_cast<U>(av ^ ov);
                            for (int q = 0; q < PASSES; ++q)
                                live[(std::size_t)q] =
                                (((varying >> (q * 8)) & static_cast<U>(0xFF)) != 0) ? 1 : 0;
                        }
                        if (live[(std::size_t)pass])
                        {
                            int sum = 0;
                            for (int b = 0; b < 256; ++b)
                            {
                                bucketStart[static_cast<std::size_t>(b)] = sum;
                                int total = 0;
                                for (unsigned int u = 0; u < nt; ++u)
                                    total += localCount[u].data[static_cast<std::size_t>(b)];
                                sum += total;
                            }
                            for (unsigned int u = 0; u < nt; ++u)
                            {
                                for (int b = 0; b < 256; ++b)
                                {
                                    int offset = bucketStart[static_cast<std::size_t>(b)];
                                    for (unsigned int w = 0; w < u; ++w)
                                        offset += localCount[w].data[static_cast<std::size_t>(b)];
                                    threadPos[u].data[static_cast<std::size_t>(b)] = offset;
                                }
                            }
                        }
                    }
                    bar.wait();

                    if (live[(std::size_t)pass])
                    {
                        T* out = buffer.data();
                        auto pos = threadPos[t].data;
                        for (int i = start; i < end; ++i)
                        {
                            if (do_prefetch && i + PREFETCH_DISTANCE < end)
                                NAGI_PREFETCH(&data[i + PREFETCH_DISTANCE]);
                            const int idx = static_cast<int>((KT::fold(KT::raw(data[i])) >> shift) & 0xFF);
                            out[static_cast<std::size_t>(pos[static_cast<std::size_t>(idx)]++)] = data[i];
                        }
                    }
                    bar.wait();

                    if (t == 0 && live[(std::size_t)pass])
                        data.swap(buffer);
                    bar.wait();
                }
            };

            std::vector<std::thread> threads;
            threads.reserve(nt - 1);
            for (unsigned int t = 1; t < nt; ++t)
                threads.emplace_back(worker, t);
            worker(0);
            for (auto& th : threads) th.join();

            if (descending) std::reverse(data.begin(), data.end());
            return true;
        }

        // ============================================================
        // Radix dispatcher
        // ============================================================
        // `allow_comparison_fallback` is false for the explicit nagi::radix_sort
        // entry point, which promises a radix sort and must not quietly decline
        // work because of a size heuristic.
        template <typename T>
        bool dispatch_radix(std::vector<T>& data, bool descending,
            bool allow_comparison_fallback = true)
        {
            using KT = key_traits<T>;
            if constexpr (!radixable_v<T>)
            {
                (void)data; (void)descending;
                return false;
            }
            else
            {
                constexpr int BITS = KT::bits;
                const int n = checked_size(data.size());

                // A wide key costs `BITS / digit` full scatter passes no matter
                // how the data looks, so for a small array the comparison sort
                // is cheaper.  Below this size the radix path is declined and
                // the introsort fallback takes over.
                if constexpr (BITS >= 48)
                {
                    if (allow_comparison_fallback && n < RADIX_MIN_N_WIDE) return false;
                }

                if constexpr (BITS == 32 || BITS == 64)
                {
                    if (n >= PARALLEL_RADIX_MIN)
                        return radix_sort_parallel(data, descending);
                    if (n >= MSD_LSD_THRESHOLD)
                        return radix_sort_msd_lsd<T, KT>(data, descending);
                }
                return radix_sort_whole<T, KT>(data, descending);
            }
        }

        // ============================================================
        // Heap sort helpers
        // ============================================================
        template <typename T, typename Compare>
        void sift_down(std::vector<T>& data, int root, int end, int offset, Compare comp)
        {
            T root_val = std::move(data[root]);
            while (true)
            {
                int child = (root - offset) * 2 + 1 + offset;
                if (child > end) break;
                if (child + 1 <= end && comp(data[child], data[child + 1])) child++;
                if (!comp(root_val, data[child])) break;
                data[root] = std::move(data[child]);
                root = child;
            }
            data[root] = std::move(root_val);
        }

        template <typename T, typename Compare>
        void heap_sort(std::vector<T>& data, int low, int high, Compare comp)
        {
            int n = high - low + 1;
            for (int start = low + n / 2 - 1; start >= low; --start) sift_down(data, start, high, low, comp);
            for (int end = high; end > low; --end)
            {
                std::swap(data[low], data[end]);
                sift_down(data, low, end - 1, low, comp);
            }
        }

        // ============================================================
        // Small-data specialized sorter (n < SMALL_DATA_THRESHOLD)
        // Design: pdqsort-inspired, no radix, no prescan, linear insertion
        // ============================================================
        template <typename T, typename Compare>
        void small_quicksort(std::vector<T>& data, int low, int high, Compare comp, int depth)
        {
            while (low < high)
            {
                int len = high - low + 1;

                if (len <= SMALL_INSERTION_THRESHOLD)
                {
                    sort_small_range(data, low, high, comp);
                    return;
                }

                if (depth > MAX_SMALL_DEPTH)
                {
                    heap_sort(data, low, high, comp);
                    return;
                }

                int mid = low + (high - low) / 2;
                median_swap3(data, low, mid, high, comp);
                std::swap(data[low], data[mid]);
                T pivot = data[low];

                // Correct three-way partition:
                // lt starts at low, i starts at low+1, gt starts at high.
                // pivot stays at data[low] until the very end.
                int lt = low;
                int i = low + 1;
                int gt = high;

                while (i <= gt)
                {
                    if (comp(data[i], pivot))
                    {
                        ++lt;
                        std::swap(data[lt], data[i]);
                        ++i;
                    }
                    else if (comp(pivot, data[i]))
                    {
                        std::swap(data[i], data[gt]);
                        --gt;
                    }
                    else
                    {
                        ++i;
                    }
                }
                std::swap(data[low], data[lt]);

                int l_size = lt - low;
                int r_size = high - gt;

                if (l_size <= 16 && r_size <= 16)
                {
                    bool left_ok = partial_insertion_sort(data, low, lt - 1, comp);
                    bool right_ok = partial_insertion_sort(data, gt + 1, high, comp);
                    if (left_ok && right_ok) return;
                }

                if (l_size < r_size)
                {
                    small_quicksort(data, low, lt - 1, comp, depth + 1);
                    low = gt + 1;
                }
                else
                {
                    small_quicksort(data, gt + 1, high, comp, depth + 1);
                    high = lt - 1;
                }
                depth++;
            }
        }

        template <typename T, typename Compare>
        void small_sort_core(std::vector<T>& data, int low, int high, Compare comp)
        {
            int len = high - low + 1;
            if (len <= 1) return;

            if (len <= SMALL_INSERTION_THRESHOLD)
            {
                sort_small_range(data, low, high, comp);
                return;
            }

            small_quicksort(data, low, high, comp, 0);
        }

        // ============================================================
        // Pre-scan
        // ============================================================
        enum ScanClass {
            SCAN_UNKNOWN,
            SCAN_ALL_SAME,
            SCAN_SORTED,
            SCAN_REVERSED,
            SCAN_NEARLY_SORTED
        };

        struct ScanResult
        {
            ScanClass cls;
            int invCount;   // -1 when the scan bailed out early
        };

        template <typename T, typename Compare>
        ScanResult try_prescan(std::vector<T>& data, int n, Compare comp)
        {
            if (n > QUICK_REJECT_SAMPLE)
            {
                int descCount = 0;
                int ascCount = 0;
                for (int i = 1; i < QUICK_REJECT_SAMPLE; ++i)
                {
                    if (comp(data[i], data[i - 1])) ++descCount;
                    else if (comp(data[i - 1], data[i])) ++ascCount;
                }
                bool monotonicDesc = (descCount >= QUICK_REJECT_SAMPLE - 2);
                bool monotonicAsc = (ascCount >= QUICK_REJECT_SAMPLE - 2);
                if (!monotonicDesc && !monotonicAsc && descCount > 8 && ascCount > 8)
                    return ScanResult{ SCAN_UNKNOWN, -1 };
            }

            // Two plain counters instead of three flag variables.  Same
            // information (no descent => sorted, no ascent => reversed,
            // neither => all equal), but this form vectorises into a pair of
            // pcmpgtd/paddd: measured 5.9x faster on int32 and 2x on double,
            // and this loop runs for every input of 2048 elements or more.
            int invCount = 0;    // pairs with data[i] < data[i-1]
            int ascCount = 0;    // pairs with data[i] > data[i-1]

            for (int i = 1; i < n; ++i)
            {
                invCount += comp(data[i], data[i - 1]);
                ascCount += comp(data[i - 1], data[i]);
            }

            if (invCount == 0 && ascCount == 0) return ScanResult{ SCAN_ALL_SAME, invCount };
            if (invCount == 0) return ScanResult{ SCAN_SORTED, invCount };
            if (ascCount == 0) return ScanResult{ SCAN_REVERSED, invCount };
            //不要简单地把 MAX_NEARLY_SORTED_INVERSION 调大，那会让真正的随机数据误判成“近乎有序”，导致 O(n²) 退化。!!

            if (invCount < MAX_NEARLY_SORTED_INVERSION && n > 32)
            {
                // Integers have a direct radix/counting fast path; spending
                // a full insertion attempt on them is not worthwhile.  For
                // non-integers the caller uses a move budget and falls back
                // to a worst-case-safe strategy if the budget is exceeded.
                if constexpr (is_integral_sortable_v<T>)
                    return ScanResult{ SCAN_UNKNOWN, invCount };
                else
                    return ScanResult{ SCAN_NEARLY_SORTED, invCount };
            }

            return ScanResult{ SCAN_UNKNOWN, invCount };
        }
// ============================================================
// Dispatch Layer (v5.0.0 引入)
// ============================================================
// 框架的核心：把「决策」从「执行」里分出来。
//
// 每个 Route 对应一个策略（Strategy）。当前支持的策略：
//   - SMALL_SORT  : n <= SMALL_DATA_THRESHOLD，pdqsort 风格
//   - PRESCAN     : n >= PRESCAN_THRESHOLD，特征探测
//   - RADIX_PATH  : 256 < n < 5e8，counting 或 radix
//   - TOPK        : k in [1, n]
//   - INTROSORT   : 通用比较排序兜底
//
// 未来新增策略时，只需：
//   1. 在 plan_initial_route 里加一条决策规则
//   2. 在 intro_sort_core 里加对应的执行分支
//   3. 其他策略与特征探测逻辑不受影响
//
// 约束：
//   - plan_initial_route 是纯函数：不访问 data、不做 I/O、不分配内存
//   - 决策只依赖 n、k、比较器类型（编译期）
//   - 所有策略必须有 worst-case 兜底（当前是 INTROSORT）

        enum class Route : int {
            SMALL_SORT,
            SORTED_CHECK,
            PRESCAN,
            RADIX_PATH,
            TOPK,
            INTROSORT,
            DONE           
        };

        template <typename T, typename Compare>
        inline Route plan_initial_route(int n, int k) noexcept
        {
            constexpr order_kind ord = order_of_v<Compare, T>;
            constexpr bool is_default = (ord != order_kind::custom);
            constexpr bool can_use_prescan = is_default;
            constexpr bool can_use_radix = is_default && radixable_v<T>;

            if (k == -1 && n <= SMALL_DATA_THRESHOLD)
                return Route::SMALL_SORT;

            if (k == -1 && n < PRESCAN_THRESHOLD)
                return Route::SORTED_CHECK;

            if (k == -1 && can_use_prescan && n >= PRESCAN_THRESHOLD)
                return Route::PRESCAN;

            if (k == -1 && can_use_radix && n > RADIX_SORT_THRESHOLD && n < 500000000)
                return Route::RADIX_PATH;

            if (k != -1 && k > 0 && k <= n)
                return Route::TOPK;

            return Route::INTROSORT;
        }

        // ============================================================
// Refine Layer (v5.1.3 引入)
// ============================================================
// 二次路由：在 plan_initial_route 之后，看到数据后做精化决策。
//
// 与 plan_initial_route 的区别：
//   - plan_initial_route: 纯函数，只看 n、k、比较器类型（编译期）
//   - refine_route:       可以访问数据，做"看数据"的决策
//
// 返回语义：
//   - DONE:       数据已被 refine 层处理完毕，调用者直接返回
//   - 其他 Route: 精化后的最终路由，调用者执行对应策略
//
// 当前负责的两次精化：
//   1. SORTED_CHECK: 对 256 < n < PRESCAN_THRESHOLD 的数据做廉价探测
//   2. PRESCAN:      对 n >= PRESCAN_THRESHOLD 的数据做特征探测

        template <typename T, typename Compare>
        Route refine_route(Route initial, std::vector<T>& data, int n, Compare comp)
        {
            constexpr bool can_use_radix =
                (order_of_v<Compare, T> != order_kind::custom) &&
                radixable_v<T>;

            // ============================================================
            // Refine: SORTED_CHECK
            // ============================================================
            // 对 256 < n < 2048 的数据，先做一次廉价的"是否有序/逆序"探测。
            //   有序 → 直接返回
            //   逆序 → std::reverse 后返回
            //   都不像 → 落到 RADIX_PATH 或 INTROSORT
            // 收益：命中时节省 3 趟 radix 扫描。
            if (initial == Route::SORTED_CHECK)
            {
                // 廉价探测：只查前 16 个元素
                const int probe = std::min(n, 16);
                bool looks_asc = true;
                bool looks_desc = true;
                for (int i = 1; i < probe; ++i)
                {
                    if (comp(data[i], data[i - 1])) looks_asc = false;
                    if (comp(data[i - 1], data[i])) looks_desc = false;
                    if (!looks_asc && !looks_desc) break;
                }

                // 采样已否定有序/逆序 → 直接交给下一档，省一次全扫
                if (!looks_asc && !looks_desc)
                {
                    return can_use_radix ? Route::RADIX_PATH : Route::INTROSORT;
                }

                // 全扫：一次扫描同时统计升序对和逆序对
                int ascPairs = 0;
                int descPairs = 0;
                for (int i = 1; i < n; ++i)
                {
                    if (comp(data[i], data[i - 1]))      ++descPairs;
                    else if (comp(data[i - 1], data[i])) ++ascPairs;
                }

                // 全部相等 → 已"有序"
                if (ascPairs == 0 && descPairs == 0) return Route::DONE;

                // 非递增 → 反转即有序
                if (ascPairs == 0)
                {
                    std::reverse(data.begin(), data.end());
                    return Route::DONE;
                }

                // 非递减 → 已有序
                if (descPairs == 0) return Route::DONE;

                // 其他情况：交给下一档
                return can_use_radix ? Route::RADIX_PATH : Route::INTROSORT;
            }

            // ============================================================
            // Refine: PRESCAN
            // ============================================================
            if (initial == Route::PRESCAN)
            {
                const ScanResult sr = try_prescan(data, n, comp);
                const ScanClass sc = sr.cls;
                const int invCount = sr.invCount;
                if (sc == SCAN_ALL_SAME || sc == SCAN_SORTED) return Route::DONE;
                if (sc == SCAN_REVERSED)
                {
                    std::reverse(data.begin(), data.end());
                    return Route::DONE;
                }
                if (sc == SCAN_NEARLY_SORTED)
                {
                    // Budgeted insertion: finish only while the number of
                    // moved elements stays small.  If the budget is exceeded,
                    // the current element has still been inserted, so the
                    // data remains a valid permutation and can safely fall
                    // through to a worst-case-safe strategy.
                    if (partial_insertion_sort(data, 0, n - 1, comp,
                            NEARLY_SORTED_MOVE_BUDGET))
                        return Route::DONE;

                    // The insertion pass ran out of budget, but the data is
                    // still almost ordered: the comparison path exploits that
                    // for free, a radix sweep ignores it.  (Counting sort is
                    // already linear, so it is never a mistake to prefer it.)
                    if (!can_use_radix || !counting_sort_possible(data, n))
                        return Route::INTROSORT;
                    return Route::RADIX_PATH;
                }
                // Few adjacent inversions, but too many for the insertion fast
                // path: the data is "almost ordered".  A radix sweep pays for
                // every digit regardless, while the comparison path finishes
                // nearly-ordered partitions in linear time, so it wins by a wide
                // margin (measured ~1.5x on 0.5 %-shuffled doubles).  Introsort
                // keeps the worst case at O(n log n), so the guess cannot
                // backfire.
                //
                // Two exemptions:
                //   * a narrow value range means counting sort is already
                //     linear, and duplicate-heavy data would suffer in a
                //     comparison sort;
                //   * a 32-bit key at parallel sizes has a 4-thread radix pass
                //     available, which beats any single-threaded comparison
                //     sort.
                constexpr bool has_parallel_radix =
                    radixable_v<T> && (key_traits<T>::bits == 32);
                const bool parallel_wins = has_parallel_radix && n >= PARALLEL_RADIX_MIN;

                if (!parallel_wins && invCount >= 0 && n >= 4096 &&
                    static_cast<long long>(invCount) * NEARLY_SORTED_INV_DIVISOR < n &&
                    !counting_sort_possible(data, n))
                    return Route::INTROSORT;

                // prescan 未能确定数据特征，落到 RADIX_PATH / INTROSORT
                return can_use_radix ? Route::RADIX_PATH : Route::INTROSORT;
            }

            // 其他 Route 原样返回
            return initial;
        }

        // ============================================================
        // Introsort core
        // ============================================================
        struct Range { int low; int high; int depth; int bad_allowed; };

        template <typename T, typename Compare>
        void intro_sort_core(std::vector<T>& data, int k, Compare comp)
        {
            int n = checked_size(data.size());
            if (n <= 1) return;

            // ============================================================
            // Stage 0: 初路由（纯函数，不看数据）
            // ============================================================
            Route route = plan_initial_route<T, Compare>(n, k);

            // ============================================================
            // Stage 1: 二次路由（看数据，做精化决策）
            // ============================================================
            route = refine_route(route, data, n, comp);

            // ============================================================
            // Stage 2: 执行
            // ============================================================

            // ---------- 已处理完毕 ----------
            if (route == Route::DONE) return;

            // ---------- Strategy: SMALL_SORT ----------
            if (route == Route::SMALL_SORT)
            {
                small_sort_core(data, 0, n - 1, comp);
                return;
            }

            // ---------- Strategy: RADIX_PATH ----------
            if (route == Route::RADIX_PATH)
            {                // plan_initial_route 已保证: k==-1, RADIX_SORT_THRESHOLD < n < 500000000
                constexpr bool is_descending =
                    order_of_v<Compare, T> == order_kind::descending;

                if (k == -1 && n > RADIX_SORT_THRESHOLD && n < 500000000)
                {
                    if (try_counting_sort(data, is_descending)) return;
                    if (dispatch_radix(data, is_descending)) return;
                }
                // radix 路径未能处理（罕见类型或超大规模），落到 INTROSORT
                route = Route::INTROSORT;
            }

            // ---------- Strategy: TOPK ----------
            if (route == Route::TOPK)
            {
                if (k == n)
                {
                    // k == n 语义上就是全排序，落到 INTROSORT
                    route = Route::INTROSORT;
                }
                else if (k <= n / 2)
                {
                    // partial_sort produces the sorted prefix in place; the
                    // rest stays a permutation of the input.
                    std::partial_sort(data.begin(), data.begin() + k, data.end(), comp);
                    return;
                }
                else
                {
                    std::nth_element(data.begin(), data.begin() + k, data.end(), comp);
                    std::sort(data.begin(), data.begin() + k, comp);
                    return;
                }
            }

            // ---------- Strategy: INTROSORT (兜底) ----------
            constexpr bool is_default_compare =
                order_of_v<Compare, T> != order_kind::custom;
            // The branchless block partition pays off only for cheap,
            // arithmetic predicates.  std::string and user types keep the
            // general three-way partition even with a default comparator.
            constexpr bool use_block_partition =
                is_default_compare && std::is_arithmetic_v<T>;

            int max_depth = 0;
            while ((1LL << max_depth) < n) max_depth++;
            max_depth *= 2;

            std::vector<Range> stk;
            stk.reserve(static_cast<size_t>(max_depth) + 16);
            stk.push_back({ 0, n - 1, max_depth, std::max(1, max_depth / 4) });

            while (!stk.empty())
            {
                Range r = stk.back();
                stk.pop_back();
                int low = r.low;
                int high = r.high;
                int depth = r.depth;
                int bad_allowed = r.bad_allowed;

                while (true)
                {
                    int len = high - low + 1;

                    if (len < INSERTION_SORT_THRESHOLD)
                    {
                        if (len <= SORT_NETWORK_THRESHOLD)
                        {
                            sort_small_range(data, low, high, comp);
                        }
                        else if (low == 0)
                            insertion_sort(data, low, high, comp);
                        else
                            unguarded_insertion_sort(data, low, high, comp);
                        break;
                    }

                    if (depth == 0)
                    {
                        heap_sort(data, low, high, comp);
                        break;
                    }

                    int mid = low + (high - low) / 2;
                    T pivot;

                    if (len >= NINTHER_THRESHOLD)
                    {
                        median_swap3(data, low, low + 1, low + 2, comp);
                        median_swap3(data, mid - 1, mid, mid + 1, comp);
                        median_swap3(data, high - 2, high - 1, high, comp);
                        median_swap3(data, low + 1, mid, high - 1, comp);
                        std::swap(data[low], data[mid]);
                        pivot = data[low];
                    }
                    else
                    {
                        median_swap3(data, low, mid, high, comp);
                        std::swap(data[low], data[mid]);
                        pivot = data[low];
                    }

                    int lt, gt;

                    if (len >= 512 && use_block_partition)
                    {
                        int pivot_pos = block_partition_right(data, low, high, comp);
                        lt = pivot_pos;
                        gt = pivot_pos;
                    }
                    else
                    {
                        int i = low;
                        lt = low;
                        gt = high;
                        while (i <= gt)
                        {
                            if (comp(data[i], pivot))
                            {
                                std::swap(data[lt], data[i]);
                                lt++;
                                i++;
                            }
                            else if (comp(pivot, data[i]))
                            {
                                std::swap(data[i], data[gt]);
                                gt--;
                            }
                            else
                                i++;
                        }
                    }

                    int l_size = lt - low;
                    int r_size = high - gt;
                    bool highly_unbalanced = (l_size < len / 8 || r_size < len / 8);

                    if (highly_unbalanced)
                    {
                        if (--bad_allowed == 0)
                        {
                            heap_sort(data, low, high, comp);
                            break;
                        }
                        if (l_size >= 4)
                        {
                            std::swap(data[low], data[low + l_size / 4]);
                            std::swap(data[lt - 1], data[lt - 1 - l_size / 4]);
                        }
                        if (r_size >= 4)
                        {
                            std::swap(data[gt + 1], data[gt + 1 + r_size / 4]);
                            std::swap(data[high], data[high - r_size / 4]);
                        }
                    }
                    else
                    {
                        if (l_size <= 16 && r_size <= 16)
                        {
                            bool left_ok = partial_insertion_sort(data, low, lt - 1, comp);
                            bool right_ok = partial_insertion_sort(data, gt + 1, high, comp);
                            if (left_ok && right_ok) break;
                        }
                    }

                    if (l_size < r_size)
                    {
                        if (gt + 1 <= high) stk.push_back({ gt + 1, high, depth - 1, bad_allowed });
                        high = lt - 1;
                    }
                    else
                    {
                        if (low <= lt - 1) stk.push_back({ low, lt - 1, depth - 1, bad_allowed });
                        low = gt + 1;
                    }
                    depth--;
                }
            }
        }

    }  // namespace detail

    // ============================================================
    // Public API
    // ============================================================

    // Radix sorting has a numeric meaning only.  Accepting an arbitrary
    // comparator would silently ignore it (or, worse, require guessing the
    // predicate); expose the two orders explicitly instead.
    enum class order {
        ascending,
        descending
    };

    template <typename T, typename Compare = std::less<T>>
    void sort(std::vector<T>& data, Compare comp = {})
    {
        detail::intro_sort_core(data, -1, comp);
    }

    // Convenience API: materializes a temporary vector (O(n) extra memory
    // plus a full input and output copy).  Use the vector overload for the
    // high-performance path; this overload exists for generic call sites.
    template <typename Iter, typename Compare = std::less<typename std::iterator_traits<Iter>::value_type>>
    void sort(Iter begin, Iter end, Compare comp = {})
    {
        using Cat = typename std::iterator_traits<Iter>::iterator_category;
        static_assert(
            std::is_base_of_v<std::random_access_iterator_tag, Cat>,
            "nagi::sort requires random-access iterators (use std::vector or std::array)");

        using T = typename std::iterator_traits<Iter>::value_type;
        std::vector<T> data(begin, end);
        detail::intro_sort_core(data, -1, comp);
        std::copy(data.begin(), data.end(), begin);
    }

    template <typename T, std::size_t N, typename Compare = std::less<T>>
    void sort(T(&arr)[N], Compare comp = {})
    {
        std::vector<T> data(arr, arr + N);
        detail::intro_sort_core(data, -1, comp);
        std::copy(data.begin(), data.end(), arr);
    }

    // topk contract:
    //   - data[0, k) contains the k smallest elements in sorted order;
    //   - the whole vector remains a permutation of the input (the order of
    //     the remaining elements is unspecified).
    template <typename T, typename Compare = std::less<T>>
    void topk(std::vector<T>& data, int k, Compare comp = {})
    {
        if (k <= 0) return;
        int n = detail::checked_size(data.size());
        if (k > n) k = n;
        if (k == n) { sort(data, comp); return; }
        detail::intro_sort_core(data, k, comp);
    }

    template <typename T, typename Compare = std::less<T>>
    void radix_sort(std::vector<T>& data, Compare comp = {})
    {
        static_assert(detail::radixable_v<T>,
            "nagi::radix_sort supports types with a radix key: integers "
            "except bool (up to 64 bits), float and double");

        constexpr detail::order_kind ord = detail::order_of_v<Compare, T>;
        static_assert(ord != detail::order_kind::custom,
            "nagi::radix_sort supports std::less / std::greater "
            "(including transparent std::less<> / std::greater<>); "
            "use nagi::sort for arbitrary comparators");

        (void)comp;
        const bool descending = (ord == detail::order_kind::descending);
        detail::checked_size(data.size());
        if (!detail::dispatch_radix(data, descending, false))
        {
            if constexpr (std::is_floating_point_v<T>)
                throw std::domain_error(
                    "nagi::radix_sort: input contains NaN, which has no total "
                    "order; use nagi::sort instead");
            else
                throw std::logic_error("nagi::radix_sort: unsupported type");
        }
    }

    template <typename T>
    void radix_sort(std::vector<T>& data, order ord)
    {
        if (ord == order::descending)
            radix_sort(data, std::greater<T>{});
        else
            radix_sort(data, std::less<T>{});
    }

    // ============================================================
    // LazySortedView — lazily materialized top-K view
    // ============================================================
    template <typename T, typename Compare = std::less<T>>
    class LazySortedView
    {
    public:
        LazySortedView(const std::vector<T>& source, int k, Compare comp = {})
            : m_source(source)
            , m_k((k < 0) ? 0
                : static_cast<int>(std::min<std::size_t>(
                    static_cast<std::size_t>(k), source.size())))
            , m_comp(comp)
            , m_ready(false)
        {
        }

        // A view must never outlive its source.  Reject temporaries instead
        // of leaving a dangling reference.
        LazySortedView(std::vector<T>&&, int, Compare = {}) = delete;

        const T& operator[](int idx) const
        {
            ensure_sorted();
            return m_sorted[static_cast<std::size_t>(idx)];
        }

        const T* begin() const { ensure_sorted(); return m_sorted.data(); }
        const T* end() const { ensure_sorted(); return m_sorted.data() + m_k; }

        int size() const { return m_k; }

        void materialize() const { ensure_sorted(); }

    private:
        void ensure_sorted() const
        {
            if (m_ready) return;
            if (m_k == 0)
            {
                m_ready = true;
                return;
            }

            if (static_cast<long long>(m_k) * 4 <
                static_cast<long long>(m_source.size()))
            {
                // 小 k：partial_sort_copy 直接产出前 k 个，避免全拷
                // 大 k（k*4 >= n）：走全拷 + topk（O(n) 拷贝 + nth_element O(n)）
                m_sorted.resize(static_cast<std::size_t>(m_k));
                std::partial_sort_copy(m_source.begin(), m_source.end(),
                    m_sorted.begin(), m_sorted.end(), m_comp);
            }
            else
            {
                m_sorted = m_source;
                nagi::topk(m_sorted, m_k, m_comp);
            }
            m_ready = true;
        }

        const std::vector<T>& m_source;
        mutable std::vector<T> m_sorted;
        int m_k;
        Compare m_comp;
        mutable bool m_ready;
    };

}  // namespace nagi
