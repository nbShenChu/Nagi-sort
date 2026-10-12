// ============================================================
// ripple_sort_double.h - Floating-point sorting engine
// Version: v1.0.0
// ============================================================
// A drop-in double-sorting library based on the same idea
// as Ripple-Sort: pre-scan + radix sort + introsort fallback.
// Handles NaN, +0.0 / -0.0, positive and negative doubles.
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
#include <thread>
#include <type_traits>
#include <vector>

namespace ripple_sort_double
{
namespace detail
{

// ============================================================
// Compile-time constants
// ============================================================
constexpr int INSERTION_SORT_THRESHOLD = 64;
constexpr int NINTHER_THRESHOLD = 128;
constexpr int PRESCAN_THRESHOLD = 128;
constexpr int PARTIAL_INSERTION_LIMIT = 8;
constexpr int RADIX_SORT_THRESHOLD = 256;
constexpr int SORT_NETWORK_THRESHOLD = 5;
constexpr int PARALLEL_RADIX_MIN = 1000000;
constexpr int QUICK_REJECT_SAMPLE = 64;
constexpr int QUICK_REJECT_THRESHOLD = 16;

// ============================================================
// Insertion sort
// ============================================================
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
            if (comp(key, data[mid]))
                right = mid;
            else
                left = mid + 1;
        }
        for (int j = i; j > left; --j) data[j] = std::move(data[j - 1]);
        data[left] = std::move(key);
    }
}

template <typename T, typename Compare>
void unguarded_insertion_sort(std::vector<T>& data, int low, int high, Compare comp)
{
    for (int i = low + 1; i <= high; ++i)
    {
        T key = std::move(data[i]);
        int j = i - 1;
        while (comp(key, data[j]))
        {
            data[j + 1] = std::move(data[j]);
            --j;
        }
        data[j + 1] = std::move(key);
    }
}

template <typename T, typename Compare>
bool partial_insertion_sort(std::vector<T>& data, int low, int high, Compare comp, int limit = PARTIAL_INSERTION_LIMIT)
{
    if (low >= high) return true;
    int moves = 0;
    for (int i = low + 1; i <= high; ++i)
    {
        T key = data[i];
        int j = i - 1;
        while (j >= low && comp(key, data[j]))
        {
            data[j + 1] = data[j];
            --j;
            if (++moves > limit) return false;
        }
        data[j + 1] = key;
    }
    return true;
}

// ============================================================
// Sorting networks
// ============================================================
template <typename T, typename Compare>
inline void sort_network_3(T* data, Compare comp)
{
    auto cswap = [&](int i, int j)
    {
        if (comp(data[j], data[i])) std::swap(data[i], data[j]);
    };
    cswap(0, 2);
    cswap(0, 1);
    cswap(1, 2);
}

template <typename T, typename Compare>
inline void sort_network_4(T* data, Compare comp)
{
    auto cswap = [&](int i, int j)
    {
        if (comp(data[j], data[i])) std::swap(data[i], data[j]);
    };
    cswap(0, 2);
    cswap(1, 3);
    cswap(0, 1);
    cswap(2, 3);
    cswap(1, 2);
}

template <typename T, typename Compare>
inline void sort_network_5(T* data, Compare comp)
{
    auto cswap = [&](int i, int j)
    {
        if (comp(data[j], data[i])) std::swap(data[i], data[j]);
    };
    cswap(0, 2);
    cswap(1, 3);
    cswap(0, 1);
    cswap(2, 4);
    cswap(1, 2);
    cswap(3, 4);
    cswap(0, 1);
    cswap(2, 3);
    cswap(1, 2);
}

// ============================================================
// Median of three
// ============================================================
template <typename T, typename Compare>
inline const T& median_of_three(const T& a, const T& b, const T& c, Compare comp)
{
    if (comp(b, a))
    {
        if (comp(c, b)) return c;
        if (comp(c, a)) return a;
        return b;
    }
    else
    {
        if (comp(c, a)) return a;
        if (comp(c, b)) return c;
        return b;
    }
}

// ============================================================
// Double -> unsigned 64-bit total order mapping
// -0.0 and +0.0 both map to the same key (they are equal).
// NaN maps to the largest key (sorted to the end).
// ============================================================
inline unsigned long long double_to_sortable(double x)
{
    if (std::isnan(x)) return 0xFFFFFFFFFFFFFFFFull;

    unsigned long long u;
    std::memcpy(&u, &x, sizeof(double));

    // Normalize -0.0 to +0.0 so they map to the same key
    if ((u & 0x7FFFFFFFFFFFFFFFull) == 0ull) return 0x8000000000000000ull;

    // If sign bit is set (negative), flip all bits.
    // Otherwise, flip only the sign bit.
    if (u >> 63)
        return ~u;
    else
        return u | 0x8000000000000000ull;
}

inline double sortable_to_double(unsigned long long u)
{
    if (u == 0xFFFFFFFFFFFFFFFFull) return std::numeric_limits<double>::quiet_NaN();

    unsigned long long original;
    if (u >> 63)
        original = u & 0x7FFFFFFFFFFFFFFFull;
    else
        original = ~u;

    double x;
    std::memcpy(&x, &original, sizeof(double));
    return x;
}

// ============================================================
// 8-bit x 8-pass LSD radix sort for double
// ============================================================
inline bool radix_sort_double(std::vector<double>& data)
{
    const int n = static_cast<int>(data.size());
    if (n <= 1) return true;

    std::vector<unsigned long long> keys(n);
    for (int i = 0; i < n; ++i) keys[i] = double_to_sortable(data[i]);

    std::vector<unsigned long long> buffer(n);
    unsigned long long* src = keys.data();
    unsigned long long* dst = buffer.data();

    int count[256] = {0};
    int pos[256] = {0};

    for (int byte = 0; byte < 8; ++byte)
    {
        std::memset(count, 0, sizeof(count));
        const int shift = byte * 8;
        for (int i = 0; i < n; ++i) ++count[(src[i] >> shift) & 0xFF];

        pos[0] = 0;
        for (int i = 1; i < 256; ++i) pos[i] = pos[i - 1] + count[i - 1];

        for (int i = 0; i < n; ++i)
        {
            int idx = static_cast<int>((src[i] >> shift) & 0xFF);
            dst[pos[idx]++] = src[i];
        }

        std::swap(src, dst);
    }

    // After 8 passes, "src" holds the sorted keys
    for (int i = 0; i < n; ++i) data[i] = sortable_to_double(src[i]);

    return true;
}

// ============================================================
// Multi-threaded 8-bit radix sort for double
// ============================================================
inline bool radix_sort_double_parallel(std::vector<double>& data)
{
    const int n = static_cast<int>(data.size());
    if (n <= 1) return true;

    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;
    if (numThreads > 16) numThreads = 16;
    unsigned int maxByData = static_cast<unsigned int>(n / 500000);
    if (maxByData < 1) maxByData = 1;
    if (numThreads > maxByData) numThreads = maxByData;

    if (numThreads <= 1) return radix_sort_double(data);

    std::vector<unsigned long long> keys(n);
    for (int i = 0; i < n; ++i) keys[i] = double_to_sortable(data[i]);

    std::vector<unsigned long long> buffer(n);
    const int blockSize = (n + static_cast<int>(numThreads) - 1) / static_cast<int>(numThreads);

    for (int pass = 0; pass < 8; ++pass)
    {
        const int shift = pass * 8;
        const int mask = 0xFF;

        std::vector<std::array<int, 256>> localCount(numThreads);
        std::vector<std::thread> threads;
        threads.reserve(numThreads);

        for (unsigned int t = 0; t < numThreads; ++t)
        {
            threads.emplace_back(
                [&, t]()
                {
                    localCount[t].fill(0);
                    int start = static_cast<int>(t) * blockSize;
                    int end = std::min(start + blockSize, n);
                    for (int i = start; i < end; ++i) ++localCount[t][(keys[i] >> shift) & mask];
                });
        }
        for (auto& th : threads) th.join();
        threads.clear();

        std::array<int, 256> bucketStart = {};
        {
            int sum = 0;
            for (int b = 0; b < 256; ++b)
            {
                bucketStart[b] = sum;
                int total = 0;
                for (unsigned int t = 0; t < numThreads; ++t) total += localCount[t][b];
                sum += total;
            }
        }

        std::vector<std::array<int, 256>> threadPos(numThreads);
        for (unsigned int t = 0; t < numThreads; ++t)
        {
            for (int b = 0; b < 256; ++b)
            {
                int offset = bucketStart[b];
                for (unsigned int u = 0; u < t; ++u) offset += localCount[u][b];
                threadPos[t][b] = offset;
            }
        }

        for (unsigned int t = 0; t < numThreads; ++t)
        {
            threads.emplace_back(
                [&, t]()
                {
                    int start = static_cast<int>(t) * blockSize;
                    int end = std::min(start + blockSize, n);
                    auto pos = threadPos[t];
                    for (int i = start; i < end; ++i)
                    {
                        int idx = static_cast<int>((keys[i] >> shift) & mask);
                        buffer[pos[idx]++] = keys[i];
                    }
                });
        }
        for (auto& th : threads) th.join();

        keys.swap(buffer);
    }

    for (int i = 0; i < n; ++i) data[i] = sortable_to_double(keys[i]);

    return true;
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
// Pre-scan
// ============================================================
template <typename T, typename Compare>
bool try_prescan(std::vector<T>& data, int n, Compare comp)
{
    if (n > QUICK_REJECT_SAMPLE)
    {
        int quickInv = 0;
        for (int i = 1; i < QUICK_REJECT_SAMPLE; ++i)
            if (comp(data[i], data[i - 1])) ++quickInv;
        if (quickInv >= QUICK_REJECT_THRESHOLD) return false;
    }

    bool allSame = true;
    bool isSorted = true;
    bool isReversed = true;
    int invCount = 0;

    for (int i = 1; i < n; ++i)
    {
        bool lt = comp(data[i], data[i - 1]);
        bool gt = comp(data[i - 1], data[i]);

        if (lt)
        {
            isSorted = false;
            ++invCount;
        }
        if (gt)
        {
            isReversed = false;
        }
        if (lt || gt)
        {
            allSame = false;
        }
    }

    if (allSame) return true;
    if (isSorted) return true;
    if (isReversed)
    {
        std::reverse(data.begin(), data.end());
        return true;
    }
    if (invCount < 4 && n > 32)
    {
        insertion_sort(data, 0, n - 1, comp);
        return true;
    }

    return false;
}

// ============================================================
// Introsort core (for double)
// ============================================================
struct Range
{
    int low;
    int high;
    int depth;
    int bad_allowed;
};

template <typename T, typename Compare>
void intro_sort_core(std::vector<T>& data, Compare comp)
{
    int n = static_cast<int>(data.size());
    if (n <= 1) return;

    if (n >= PRESCAN_THRESHOLD)
        if (try_prescan(data, n, comp)) return;

    if (n > RADIX_SORT_THRESHOLD && n < 500000000)
    {
        if (n >= PARALLEL_RADIX_MIN)
        {
            if (radix_sort_double_parallel(data)) return;
        }
        else
        {
            if (radix_sort_double(data)) return;
        }
    }

    int max_depth = 0;
    while ((1LL << max_depth) < n) max_depth++;
    max_depth *= 2;

    std::vector<Range> stk;
    stk.reserve(static_cast<size_t>(max_depth) + 16);
    stk.push_back({0, n - 1, max_depth, std::max(1, max_depth / 4)});

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
                    switch (len)
                    {
                        case 1:
                            break;
                        case 2:
                            if (comp(data[high], data[low])) std::swap(data[low], data[high]);
                            break;
                        case 3:
                            sort_network_3(data.data() + low, comp);
                            break;
                        case 4:
                            sort_network_4(data.data() + low, comp);
                            break;
                        case 5:
                            sort_network_5(data.data() + low, comp);
                            break;
                        default:
                            insertion_sort(data, low, high, comp);
                            break;
                    }
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
                const T& m1 = median_of_three(data[low], data[low + 1], data[low + 2], comp);
                const T& m2 = median_of_three(data[mid - 1], data[mid], data[mid + 1], comp);
                const T& m3 = median_of_three(data[high - 2], data[high - 1], data[high], comp);
                pivot = median_of_three(m1, m2, m3, comp);
            }
            else
            {
                if (comp(data[mid], data[low])) std::swap(data[mid], data[low]);
                if (comp(data[high], data[low])) std::swap(data[high], data[low]);
                if (comp(data[high], data[mid])) std::swap(data[high], data[mid]);
                pivot = std::move(data[mid]);
            }

            int lt = low, i = low, gt = high;
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
                if (gt + 1 <= high) stk.push_back({gt + 1, high, depth - 1, bad_allowed});
                high = lt - 1;
            }
            else
            {
                if (low <= lt - 1) stk.push_back({low, lt - 1, depth - 1, bad_allowed});
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

// Sort a vector<double> in ascending order (default)
inline void sort(std::vector<double>& data)
{
    detail::intro_sort_core(data, std::less<>());
}

// Sort with custom comparator
template <typename Compare>
void sort(std::vector<double>& data, Compare comp)
{
    detail::intro_sort_core(data, comp);
}

// Sort a range of doubles (copies to vector, sorts, writes back)
template <typename Iter, typename Compare = std::less<double>>
void sort(Iter begin, Iter end, Compare comp = {})
{
    std::vector<double> data(begin, end);
    detail::intro_sort_core(data, comp);
    std::copy(data.begin(), data.end(), begin);
}

}  // namespace ripple_sort_double
