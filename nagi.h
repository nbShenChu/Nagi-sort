// ============================================================
// nagi.h - Adaptive Sort Framework
// Version: v5.1.5 (Framework: Refine Layer + correctness hotfix)
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

        constexpr int MSD_BUCKET_BITS = 8;
        constexpr int MSD_BUCKET_SIZE = 1 << MSD_BUCKET_BITS;
        constexpr int MSD_BUCKET_MASK = MSD_BUCKET_SIZE - 1;
        constexpr int MSD_SHIFT = 24;
        constexpr int LSD12_SIZE = 1 << 12;
        constexpr int LSD12_MASK = LSD12_SIZE - 1;
        constexpr int PREFETCH_DISTANCE = 16;
        constexpr int PREFETCH_THRESHOLD = 16384;

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

        template <typename T>
        constexpr bool radix32_capable_v =
            is_integral_sortable_v<T> &&
            (std::numeric_limits<std::make_unsigned_t<T>>::digits <= 32);

        template <typename T>
        constexpr bool radix64_capable_v =
            is_integral_sortable_v<T> &&
            (std::numeric_limits<std::make_unsigned_t<T>>::digits <= 64);

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
        // Order-preserving radix keys
        // ============================================================
        // The key has the same width as make_unsigned_t<T>.  For signed
        // types, flipping the sign bit maps the signed ordering onto the
        // unsigned ordering; this also works for 8/16-bit types and for
        // `long` / implementation-defined fixed-width aliases.
        template <typename T>
        inline auto radix_key(T x)
        {
            using U = std::make_unsigned_t<T>;
            U u = static_cast<U>(x);
            if constexpr (std::is_signed_v<T>)
                u ^= static_cast<U>(U(1) << (std::numeric_limits<U>::digits - 1));
            return u;
        }

        template <typename T>
        inline unsigned int to_unsigned_32(T x)
        {
            static_assert(radix32_capable_v<T>, "type does not fit in a 32-bit radix key");
            return static_cast<unsigned int>(radix_key(x));
        }

        template <typename T>
        inline unsigned long long to_unsigned_64(T x)
        {
            static_assert(radix64_capable_v<T>, "type does not fit in a 64-bit radix key");
            return static_cast<unsigned long long>(radix_key(x));
        }

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

        template <typename T>
        bool try_counting_sort(std::vector<T>& data, bool descending = false)
        {
            if constexpr (!is_integral_sortable_v<T>)
                return false;
            else
            {
                const int n = checked_size(data.size());
                if (n <= 1) return true;

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
        // 11-bit single-threaded radix sort (3 passes)
        // Direction handled by reversing pos array initialization.
        // ============================================================
        // ============================================================
        // 11-bit single-threaded radix sort
        // All digit histograms are collected in one input scan; passes whose
        // digit is constant are skipped.  Direction is handled by reversing
        // each pass's bucket prefix order.
        // ============================================================
        template <typename T, std::size_t N>
        void radix_scatter_32(std::vector<T>& data, std::vector<T>& buffer,
            const int (&count)[N], int shift, unsigned int mask,
            bool descending, bool do_prefetch)
        {
            int pos[N];
            const int SIZE = static_cast<int>(N);

            if (!descending)
            {
                int sum = 0;
                for (int i = 0; i < SIZE; ++i) { pos[i] = sum; sum += count[i]; }
            }
            else
            {
                pos[SIZE - 1] = 0;
                for (int i = SIZE - 2; i >= 0; --i) pos[i] = pos[i + 1] + count[i + 1];
            }

            const int n = checked_size(data.size());
            for (int i = 0; i < n; ++i)
            {
                if (do_prefetch && i + PREFETCH_DISTANCE < n) NAGI_PREFETCH(&data[i + PREFETCH_DISTANCE]);
                const int idx = static_cast<int>((to_unsigned_32(data[i]) >> shift) & mask);
                buffer[static_cast<std::size_t>(pos[idx]++)] = data[i];
            }
            data.swap(buffer);
        }

        template <typename T>
        bool radix_sort_32_11bit(std::vector<T>& data, bool descending = false)
        {
            if constexpr (!radix32_capable_v<T>)
                return false;

            const int n = checked_size(data.size());
            if (n <= 1) return true;

            const bool do_prefetch = (n > PREFETCH_THRESHOLD);
            std::vector<T> buffer(static_cast<std::size_t>(n));

            constexpr int SHIFT1 = 0, SHIFT2 = 11, SHIFT3 = 22;
            constexpr unsigned int MASK1 = 0x7FF, MASK2 = 0x7FF, MASK3 = 0x3FF;
            constexpr int SIZE1 = 2048, SIZE2 = 2048, SIZE3 = 1024;

            int count0[SIZE1] = { 0 };
            int count1[SIZE2] = { 0 };
            int count2[SIZE3] = { 0 };

            unsigned int firstKey = 0;
            unsigned int varying = 0;

            for (int i = 0; i < n; ++i)
            {
                if (do_prefetch && i + PREFETCH_DISTANCE < n) NAGI_PREFETCH(&data[i + PREFETCH_DISTANCE]);
                const unsigned int key = to_unsigned_32(data[i]);
                if (i == 0) firstKey = key;
                varying |= key ^ firstKey;
                ++count0[key & MASK1];
                ++count1[(key >> SHIFT2) & MASK2];
                ++count2[(key >> SHIFT3) & MASK3];
            }

            if (varying == 0) return true;   // all keys equal

            if (((varying >> SHIFT1) & MASK1) != 0)
                radix_scatter_32(data, buffer, count0, SHIFT1, MASK1, descending, do_prefetch);
            if (((varying >> SHIFT2) & MASK2) != 0)
                radix_scatter_32(data, buffer, count1, SHIFT2, MASK2, descending, do_prefetch);
            if (((varying >> SHIFT3) & MASK3) != 0)
                radix_scatter_32(data, buffer, count2, SHIFT3, MASK3, descending, do_prefetch);

            return true;
        }

        // ============================================================
        // MSD+LSD hybrid radix sort（桶内排序部分）
        // 契约：桶内一律升序。方向由调用方 radix_sort_32_msd_lsd 末尾的
        //       if (descending) std::reverse(...) 统一处理。
        // 如需桶内降序，必须同时删掉外层 reverse，否则双重反转。
        // ============================================================
        template <typename T>
        void radix_sort_bucket_24bit(std::vector<T>& data, std::vector<T>& buffer, int lo, int hi)
        {
            const int m = hi - lo;
            if (m <= 1) return;

            const bool do_prefetch = (m > PREFETCH_THRESHOLD);

            {
                int count[LSD12_SIZE] = { 0 };
                for (int i = 0; i < m; ++i)
                {
                    if (do_prefetch && i + PREFETCH_DISTANCE < m) NAGI_PREFETCH(&data[lo + i + PREFETCH_DISTANCE]);
                    ++count[to_unsigned_32(data[lo + i]) & LSD12_MASK];
                }
                int pos[LSD12_SIZE];
                {
                    int sum = 0;
                    for (int i = 0; i < LSD12_SIZE; ++i) { pos[i] = sum; sum += count[i]; }
                }
                for (int i = 0; i < m; ++i)
                {
                    if (do_prefetch && i + PREFETCH_DISTANCE < m) NAGI_PREFETCH(&data[lo + i + PREFETCH_DISTANCE]);
                    int idx = to_unsigned_32(data[lo + i]) & LSD12_MASK;
                    buffer[pos[idx]++] = data[lo + i];
                }
            }

            {
                int count[LSD12_SIZE] = { 0 };
                for (int i = 0; i < m; ++i)
                {
                    if (do_prefetch && i + PREFETCH_DISTANCE < m) NAGI_PREFETCH(&buffer[i + PREFETCH_DISTANCE]);
                    ++count[(to_unsigned_32(buffer[i]) >> 12) & LSD12_MASK];
                }
                int pos[LSD12_SIZE];
                {
                    int sum = 0;
                    for (int i = 0; i < LSD12_SIZE; ++i) { pos[i] = sum; sum += count[i]; }
                }
                for (int i = 0; i < m; ++i)
                {
                    if (do_prefetch && i + PREFETCH_DISTANCE < m) NAGI_PREFETCH(&buffer[i + PREFETCH_DISTANCE]);
                    int idx = (to_unsigned_32(buffer[i]) >> 12) & LSD12_MASK;
                    data[lo + pos[idx]++] = buffer[i];
                }
            }
        }

        template <typename T>
        bool radix_sort_32_msd_lsd(std::vector<T>& data, bool descending = false)
        {
            if constexpr (!radix32_capable_v<T>)
                return false;

            const int n = checked_size(data.size());
            if (n <= 1) return true;

            const bool do_prefetch = (n > PREFETCH_THRESHOLD);

            std::vector<T> msdBuffer(static_cast<std::size_t>(n));
            int count[MSD_BUCKET_SIZE] = { 0 };

            for (int i = 0; i < n; ++i)
            {
                if (do_prefetch && i + PREFETCH_DISTANCE < n) NAGI_PREFETCH(&data[i + PREFETCH_DISTANCE]);
                ++count[(to_unsigned_32(data[i]) >> MSD_SHIFT) & MSD_BUCKET_MASK];
            }

            int pos[MSD_BUCKET_SIZE];
            {
                int sum = 0;
                for (int i = 0; i < MSD_BUCKET_SIZE; ++i) { pos[i] = sum; sum += count[i]; }
            }

            for (int i = 0; i < n; ++i)
            {
                if (do_prefetch && i + PREFETCH_DISTANCE < n) NAGI_PREFETCH(&data[i + PREFETCH_DISTANCE]);
                int idx = (to_unsigned_32(data[i]) >> MSD_SHIFT) & MSD_BUCKET_MASK;
                msdBuffer[pos[idx]++] = data[i];
            }
            data.swap(msdBuffer);

            // Reuse msdBuffer as LSD scratch space. After the swap above,
            // msdBuffer contains the old (unsorted) data, which is no longer
            // needed. It can be safely used as scratch space.
            std::vector<T>& lsdBuffer = msdBuffer;

            int bucketStart = 0;
            for (int b = 0; b < MSD_BUCKET_SIZE; ++b)
            {
                int bucketEnd = bucketStart + count[b];
                if (count[b] <= 16) {
                    // 全局用 reverse 统一处理降序，所以桶内一律升序
                    insertion_sort(data, bucketStart, bucketEnd - 1, std::less<T>{});
                }
                else if (count[b] > 1) {
                    radix_sort_bucket_24bit(data, lsdBuffer, bucketStart, bucketEnd);
                }
                bucketStart = bucketEnd;
            }

            if (descending) std::reverse(data.begin(), data.end());

            return true;
        }

        // ============================================================
        // Multi-threaded 8-bit radix sort
        // Direction handled by final reverse.
        // ============================================================
        template <typename T>
        bool radix_sort_32_parallel(std::vector<T>& data, bool descending = false)
        {
            if constexpr (!radix32_capable_v<T>)
                return false;

            const int n = checked_size(data.size());
            if (n <= 1) return true;

            unsigned int numThreads = std::thread::hardware_concurrency();
            if (numThreads == 0) numThreads = 4;
            if (numThreads > 16) numThreads = 16;
            unsigned int maxByData = static_cast<unsigned int>(n / 250000);
            if (maxByData < 1) maxByData = 1;
            if (numThreads > maxByData) numThreads = maxByData;

            if (numThreads <= 1) return radix_sort_32_11bit(data, descending);

            const bool do_prefetch = (n > PREFETCH_THRESHOLD);

            struct alignas(64) AlignedBucket {
                std::array<int, 256> data;
            };

            // Ping-pong owner: scatter into `buffer`, then O(1) swap the
            // two vectors.  The old code did a full swap_ranges after every
            // pass, reading and rewriting all N elements for no reason.
            std::vector<T> buffer(static_cast<std::size_t>(n));
            const int blockSize = static_cast<int>(
                (static_cast<long long>(n) + static_cast<long long>(numThreads) - 1) /
                static_cast<long long>(numThreads));

            auto block_bounds = [&](unsigned int t, int& start, int& end) {
                long long s = static_cast<long long>(t) * static_cast<long long>(blockSize);
                long long e = (t + 1 == numThreads) ? n : s + blockSize;
                if (s > n) s = n;
                if (e > n) e = n;
                start = static_cast<int>(s);
                end = static_cast<int>(e);
            };

            // 持久状态：复用，避免每趟重建
            std::vector<AlignedBucket> localCount(numThreads);
            std::vector<AlignedBucket> threadPos(numThreads);
            std::array<int, 256> bucketStart;
            std::vector<std::thread> threads;
            threads.reserve(numThreads - 1);

            for (int pass = 0; pass < 4; ++pass)
            {
                const int shift = pass * 8;
                const int mask = 0xFF;

                for (unsigned int t = 0; t < numThreads; ++t)
                    localCount[t].data.fill(0);
                threads.clear();

                // ==================== 计数阶段 ====================
                for (unsigned int t = 1; t < numThreads; ++t)
                {
                    threads.emplace_back([&, t]() {
                        int start, end;
                        block_bounds(t, start, end);
                        auto& lc = localCount[t].data;
                        for (int i = start; i < end; ++i)
                        {
                            if (do_prefetch && i + PREFETCH_DISTANCE < end)
                                NAGI_PREFETCH(&data[i + PREFETCH_DISTANCE]);
                            ++lc[(to_unsigned_32(data[i]) >> shift) & mask];
                        }
                        });
                }
                {
                    int start, end;
                    block_bounds(0, start, end);
                    auto& lc = localCount[0].data;
                    for (int i = start; i < end; ++i)
                    {
                        if (do_prefetch && i + PREFETCH_DISTANCE < end)
                            NAGI_PREFETCH(&data[i + PREFETCH_DISTANCE]);
                        ++lc[(to_unsigned_32(data[i]) >> shift) & mask];
                    }
                }
                for (auto& th : threads) th.join();

                // ==================== 前缀和 ====================
                {
                    int sum = 0;
                    for (int b = 0; b < 256; ++b)
                    {
                        bucketStart[b] = sum;
                        int total = 0;
                        for (unsigned int t = 0; t < numThreads; ++t)
                            total += localCount[t].data[b];
                        sum += total;
                    }
                }
                for (unsigned int t = 0; t < numThreads; ++t)
                {
                    for (int b = 0; b < 256; ++b)
                    {
                        int offset = bucketStart[b];
                        for (unsigned int u = 0; u < t; ++u)
                            offset += localCount[u].data[b];
                        threadPos[t].data[b] = offset;
                    }
                }

                // ==================== 分散阶段 ====================
                threads.clear();
                T* out = buffer.data();
                for (unsigned int t = 1; t < numThreads; ++t)
                {
                    threads.emplace_back([&, t, out]() {
                        int start, end;
                        block_bounds(t, start, end);
                        auto pos = threadPos[t].data;
                        for (int i = start; i < end; ++i)
                        {
                            if (do_prefetch && i + PREFETCH_DISTANCE < end)
                                NAGI_PREFETCH(&data[i + PREFETCH_DISTANCE]);
                            int idx = (to_unsigned_32(data[i]) >> shift) & mask;
                            out[static_cast<std::size_t>(pos[idx]++)] = data[i];
                        }
                        });
                }
                {
                    int start, end;
                    block_bounds(0, start, end);
                    auto pos = threadPos[0].data;
                    for (int i = start; i < end; ++i)
                    {
                        if (do_prefetch && i + PREFETCH_DISTANCE < end)
                            NAGI_PREFETCH(&data[i + PREFETCH_DISTANCE]);
                        int idx = (to_unsigned_32(data[i]) >> shift) & mask;
                        out[static_cast<std::size_t>(pos[idx]++)] = data[i];
                    }
                }
                for (auto& th : threads) th.join();

                data.swap(buffer);
            }

            if (descending) std::reverse(data.begin(), data.end());

            return true;
        }

        // ============================================================
        // 64-bit radix sort (direction handled via pos initialization)
        // ============================================================
        template <typename T>
        bool radix_sort_64(std::vector<T>& data, bool descending = false)
        {
            if constexpr (!radix64_capable_v<T>)
                return false;

            const int n = checked_size(data.size());
            if (n <= 1) return true;

            const bool do_prefetch = (n > PREFETCH_THRESHOLD);

            std::vector<T> buffer(static_cast<std::size_t>(n));

            // Collect all eight byte histograms in a single input scan.
            // Digit frequencies do not depend on the current permutation,
            // so the later LSD passes must not re-count.
            int count[8][256] = {};
            unsigned long long firstKey = 0;
            unsigned long long varying = 0;

            for (int i = 0; i < n; ++i)
            {
                if (do_prefetch && i + PREFETCH_DISTANCE < n) NAGI_PREFETCH(&data[i + PREFETCH_DISTANCE]);
                const unsigned long long key = to_unsigned_64(data[i]);
                if (i == 0) firstKey = key;
                varying |= key ^ firstKey;
                for (int byte = 0; byte < 8; ++byte)
                    ++count[byte][(key >> (byte * 8)) & 0xFF];
            }

            if (varying == 0) return true;   // all keys equal

            int pos[256];
            for (int byte = 0; byte < 8; ++byte)
            {
                // A constant digit cannot change the ordering; skipping the
                // stable pass is safe and avoids 8 full memory sweeps for
                // values whose high bytes are all zero.
                if (((varying >> (byte * 8)) & 0xFFull) == 0)
                    continue;

                if (!descending)
                {
                    int sum = 0;
                    for (int i = 0; i < 256; ++i) { pos[i] = sum; sum += count[byte][i]; }
                }
                else
                {
                    pos[255] = 0;
                    for (int i = 254; i >= 0; --i) pos[i] = pos[i + 1] + count[byte][i + 1];
                }

                const int shift = byte * 8;
                for (int i = 0; i < n; ++i)
                {
                    if (do_prefetch && i + PREFETCH_DISTANCE < n) NAGI_PREFETCH(&data[i + PREFETCH_DISTANCE]);
                    const int idx = static_cast<int>((to_unsigned_64(data[i]) >> shift) & 0xFF);
                    buffer[static_cast<std::size_t>(pos[idx]++)] = data[i];
                }
                data.swap(buffer);
            }
            return true;
        }

        // ============================================================
// 64-bit MSD+LSD hybrid (symmetric to 32-bit)
// ============================================================
        template <typename T>
        bool radix_sort_64_msd_lsd(std::vector<T>& data, bool descending = false)
        {
            if constexpr (!radix64_capable_v<T>)
                return false;

            const int n = checked_size(data.size());
            if (n <= 1) return true;

            std::vector<T> msdBuf(static_cast<std::size_t>(n));
            int count[256] = { 0 };
            for (int i = 0; i < n; ++i)
                ++count[(to_unsigned_64(data[i]) >> 56) & 0xFF];

            int pos[256];
            {
                int sum = 0;
                for (int b = 0; b < 256; ++b) { pos[b] = sum; sum += count[b]; }
            }
            for (int i = 0; i < n; ++i)
            {
                int idx = static_cast<int>((to_unsigned_64(data[i]) >> 56) & 0xFF);
                msdBuf[static_cast<std::size_t>(pos[idx]++)] = data[i];
            }
            // Ping-pong: msdBuf now holds the sorted-by-top-byte data and
            // `data` holds the old storage, which becomes the scratch.
            data.swap(msdBuf);

            std::vector<T>& scratch = msdBuf;
            int bucketStart = 0;
            for (int b = 0; b < 256; ++b)
            {
                int bucketEnd = bucketStart + count[b];
                int m = bucketEnd - bucketStart;
                if (m <= 1) { bucketStart = bucketEnd; continue; }

                if (m <= 16)
                {
                    insertion_sort(data, bucketStart, bucketEnd - 1, std::less<T>{});
                }
                else
                {
                    int c2[256], p2[256];
                    for (int byte = 0; byte < 7; ++byte)
                    {
                        std::memset(c2, 0, sizeof(c2));
                        int shift = byte * 8;
                        for (int i = bucketStart; i < bucketEnd; ++i)
                            ++c2[(to_unsigned_64(data[i]) >> shift) & 0xFF];
                        int s = 0;
                        for (int k = 0; k < 256; ++k) { p2[k] = s; s += c2[k]; }
                        for (int i = bucketStart; i < bucketEnd; ++i)
                        {
                            int idx = static_cast<int>((to_unsigned_64(data[i]) >> shift) & 0xFF);
                            scratch[static_cast<std::size_t>(p2[idx]++)] = data[i];
                        }
                        for (int i = 0; i < m; ++i) data[bucketStart + i] = scratch[static_cast<std::size_t>(i)];
                    }
                }
                bucketStart = bucketEnd;
            }

            if (descending) std::reverse(data.begin(), data.end());
            return true;
        }

        // ============================================================
        // Radix dispatcher
        // ============================================================
        template <typename T>
        bool dispatch_radix(std::vector<T>& data, bool descending)
        {
            if constexpr (radix32_capable_v<T>)
            {
                const int n = checked_size(data.size());
                if (n >= PARALLEL_RADIX_MIN)
                    return radix_sort_32_parallel(data, descending);
                else if (n >= MSD_LSD_THRESHOLD)
                    return radix_sort_32_msd_lsd(data, descending);
                else
                    return radix_sort_32_11bit(data, descending);
            }
            else if constexpr (radix64_capable_v<T>)
            {
                const int n = checked_size(data.size());
                if (n >= MSD_LSD_THRESHOLD)
                    return radix_sort_64_msd_lsd(data, descending);
                return radix_sort_64(data, descending);
            }
            else
            {
                (void)data;
                (void)descending;
                return false;
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

        template <typename T, typename Compare>
        ScanClass try_prescan(std::vector<T>& data, int n, Compare comp)
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
                    return SCAN_UNKNOWN;
            }

            bool allSame = true;
            bool isSorted = true;
            bool isReversed = true;
            int invCount = 0;

            for (int i = 1; i < n; ++i)
            {
                bool lt = comp(data[i], data[i - 1]);
                bool gt = comp(data[i - 1], data[i]);
                if (lt) { isSorted = false; ++invCount; }
                if (gt) { isReversed = false; }
                if (lt || gt) { allSame = false; }
            }

            if (allSame) return SCAN_ALL_SAME;
            if (isSorted) return SCAN_SORTED;
            if (isReversed) return SCAN_REVERSED;
            //不要简单地把 MAX_NEARLY_SORTED_INVERSION 调大，那会让真正的随机数据误判成“近乎有序”，导致 O(n²) 退化。!!

            if (invCount < MAX_NEARLY_SORTED_INVERSION && n > 32)
            {
                // Integers have a direct radix/counting fast path; spending
                // a full insertion attempt on them is not worthwhile.  For
                // non-integers the caller uses a move budget and falls back
                // to a worst-case-safe strategy if the budget is exceeded.
                if constexpr (is_integral_sortable_v<T>)
                    return SCAN_UNKNOWN;
                else
                    return SCAN_NEARLY_SORTED;
            }

            return SCAN_UNKNOWN;
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
            constexpr bool can_use_radix = is_default && is_integral_sortable_v<T>;

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
                is_integral_sortable_v<T>;

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
                ScanClass sc = try_prescan(data, n, comp);
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

                    return can_use_radix ? Route::RADIX_PATH : Route::INTROSORT;
                }
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
                    if constexpr (is_integral_sortable_v<T>)
                    {
                        if (try_counting_sort(data, is_descending)) return;
                        if (dispatch_radix(data, is_descending)) return;
                    }
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
        static_assert(detail::is_integral_sortable_v<T>,
            "nagi::radix_sort supports integer types (except bool) only");
        static_assert(detail::radix32_capable_v<T> || detail::radix64_capable_v<T>,
            "nagi::radix_sort supports integers up to 64 bits only");

        constexpr detail::order_kind ord = detail::order_of_v<Compare, T>;
        static_assert(ord != detail::order_kind::custom,
            "nagi::radix_sort supports std::less / std::greater "
            "(including transparent std::less<> / std::greater<>); "
            "use nagi::sort for arbitrary comparators");

        (void)comp;
        const bool descending = (ord == detail::order_kind::descending);
        detail::checked_size(data.size());
        if (!detail::dispatch_radix(data, descending))
            throw std::logic_error("nagi::radix_sort: unsupported type");
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
