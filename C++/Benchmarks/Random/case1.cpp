// g++ -O3 -std=c++17 -march=native bench_random_multi.cpp -o bench_random_multi
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <utility>
#include <vector>

//--------------------------- DCE sink (avoid optimizer) -----------------------
static volatile uint64_t g_sink64 = 0;
template<typename Vec>
inline void consume(const Vec& v) {
    uint64_t s = 0; for (auto x : v) s += (uint64_t)x;
    g_sink64 ^= s;
}

//======================== LadderSort (flat tops + cached value) ===============
inline int binary_search_lad(const std::vector<int>& tops, int target) {
    int low = 0, high = (int)tops.size();
    while (low < high) {
        int mid = (low + high) / 2;
        if (tops[mid] > target) low = mid + 1;
        else high = mid;
    }
    return low;
}

struct HeapItem {
    int value, list_idx, elem_idx;
    bool operator>(const HeapItem& other) const { return value > other.value; }
};

static std::vector<int> merge_ladders(const std::vector<std::vector<int>>& lists) {
    size_t total = 0; for (const auto& v : lists) total += v.size();
    std::vector<int> merged; merged.reserve(total);

    std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<HeapItem>> heap;
    for (int i = 0; i < (int)lists.size(); ++i) {
        if (!lists[i].empty()) heap.push({lists[i][0], i, 0});
    }

    while (!heap.empty()) {
        auto top = heap.top(); heap.pop();
        merged.push_back(top.value);

        const auto& curr = lists[top.list_idx];
        int next_idx = top.elem_idx + 1;
        if (next_idx < (int)curr.size())
            heap.push({curr[next_idx], top.list_idx, next_idx});
    }
    return merged;
}

static std::vector<int> ladder_sort(const std::vector<int>& array) {
    if (array.empty()) return {};

    std::vector<std::vector<int>> lad; lad.reserve(64);
    std::vector<int> tops; tops.reserve(64);

    lad.push_back({array[0]});
    tops.push_back(array[0]);

    for (int i = 1, n = (int)array.size(); i < n; ++i) {
        int a = array[i];
        int idx = binary_search_lad(tops, a);
        if (idx == (int)lad.size()) {
            lad.emplace_back().emplace_back(a);
            tops.emplace_back(a);
        } else {
            lad[idx].emplace_back(a);
            tops[idx] = a;
        }
    }
    if (lad.size() == 1) return lad[0];
    return merge_ladders(lad);
}

//=========================== Patience Sort (for compare) ======================
static std::vector<int> patience_sort(const std::vector<int>& a) {
    if (a.empty()) return {};

    // Build piles: place x on the first pile with top >= x
    std::vector<std::vector<int>> piles;
    std::vector<int> tops; tops.reserve(64);
    piles.reserve(64);

    for (int x : a) {
        int low = 0, high = (int)tops.size();
        while (low < high) {
            int mid = (low + high) / 2;
            if (tops[mid] < x) low = mid + 1; else high = mid;
        }
        if (low == (int)piles.size()) {
            piles.emplace_back().push_back(x);
            tops.push_back(x);
        } else {
            piles[low].push_back(x);
            tops[low] = x;  // maintain nondecreasing tops
        }
    }

    // K-way merge of pile backs (each pile's sequence is nonincreasing)
    struct Node { int val; int p; }; // p = pile index
    struct Cmp { bool operator()(const Node& a, const Node& b) const { return a.val > b.val; } };

    std::priority_queue<Node, std::vector<Node>, Cmp> pq;
    for (int i = 0; i < (int)piles.size(); ++i) {
        pq.push({piles[i].back(), i});
        piles[i].pop_back();
    }

    std::vector<int> out; out.reserve(a.size());
    while (!pq.empty()) {
        auto t = pq.top(); pq.pop();
        out.push_back(t.val);
        int p = t.p;
        if (!piles[p].empty()) {
            pq.push({piles[p].back(), p});
            piles[p].pop_back();
        }
    }
    return out;
}
//====================== end Patience Sort section ============================

//--------------------------- Quicksort (3-way, iterative) ---------------------
static void quicksort3(std::vector<int>& a) {
    struct Range { int l, r; };
    std::vector<Range> st; st.reserve(64);
    st.push_back({0, (int)a.size()-1});
    while (!st.empty()) {
        auto [l,r] = st.back(); st.pop_back();
        while (l < r) {
            int m = l + (r-l)/2;
            int i1 = a[l], i2 = a[m], i3 = a[r];
            int pivot = std::max(std::min(i1,i2), std::min(std::max(i1,i2), i3)); // median-of-3
            int i=l, j=r, k=l;
            while (k <= j) {
                int v = a[k];
                if (v < pivot) std::swap(a[i++], a[k++]);
                else if (v > pivot) std::swap(a[k], a[j--]);
                else ++k;
            }
            if (i - l < r - j) { if (i-1 > l) st.push_back({l, i-1}); l = j+1; }
            else               { if (j+1 < r) st.push_back({j+1, r}); r = i-1; }
        }
    }
}

//--------------------------- Merge sort (stable, top-down) --------------------
static void mergesort_rec(std::vector<int>& a, std::vector<int>& buf, int l, int r) {
    if (r - l <= 32) {
        for (int i = l + 1; i <= r; ++i) {
            int x = a[i], j = i - 1;
            while (j >= l && a[j] > x) { a[j + 1] = a[j]; --j; }
            a[j + 1] = x;
        }
        return;
    }
    int m = l + (r - l) / 2;
    mergesort_rec(a, buf, l, m);
    mergesort_rec(a, buf, m + 1, r);
    int i = l, j = m + 1, k = l;
    while (i <= m && j <= r) buf[k++] = (a[i] <= a[j]) ? a[i++] : a[j++];
    while (i <= m) buf[k++] = a[i++];
    while (j <= r) buf[k++] = a[j++];
    for (i = l; i <= r; ++i) a[i] = buf[i];
}
static void mergesort_with_buf(std::vector<int>& a, std::vector<int>& buf) {
    if (a.empty()) return;
    if (buf.size() != a.size()) buf.assign(a.size(), 0);
    mergesort_rec(a, buf, 0, (int)a.size() - 1);
}

//--------------------------- Timsort-lite (reuse buf) -------------------------
namespace timsort_lite {
    static int minrun_for(size_t n) { int r=0; while (n>=64){ r|=(n&1U); n>>=1U; } return (int)n + r; }
    template<typename It>
    static void binary_insertion_sort(It first, It last) {
        for (It i = first+1; i<last; ++i) {
            auto x = *i;
            It lo = first, hi = i;
            while (lo < hi) { It mid = lo + (hi-lo)/2; if (*mid <= x) lo = mid+1; else hi = mid; }
            for (It j = i; j > lo; --j) *j = *(j-1);
            *lo = x;
        }
    }
    static void merge_runs(std::vector<int>& a, int l, int m, int r, std::vector<int>& buf) {
        int i=l, j=m, k=l;
        while (i<m && j<r) buf[k++] = (a[i]<=a[j]) ? a[i++] : a[j++];
        while (i<m) buf[k++] = a[i++];
        while (j<r) buf[k++] = a[j++];
        for (i=l;i<r;++i) a[i] = buf[i];
    }
    static void sort_with_buf(std::vector<int>& a, std::vector<int>& buf) {
        const int n = (int)a.size();
        if (n < 2) return;
        if ((int)buf.size() != n) buf.assign(n, 0);
        const int MINRUN = minrun_for(n);
        std::vector<std::pair<int,int>> runs; runs.reserve((n+MINRUN-1)/MINRUN);
        int i=0;
        while (i < n) {
            int j = i+1;
            if (j < n && a[j] < a[i]) {
                while (j < n && a[j] < a[j-1]) ++j;
                std::reverse(a.begin()+i, a.begin()+j);
            } else {
                while (j < n && a[j] >= a[j-1]) ++j;
            }
            int need = std::max(MINRUN, j - i);
            int end = std::min(i + need, n);
            binary_insertion_sort(a.begin()+i, a.begin()+end);
            runs.emplace_back(i, end);
            i = end;
        }
        while (runs.size() > 1) {
            std::vector<std::pair<int,int>> next;
            for (size_t k=0; k+1<runs.size(); k+=2) {
                auto [l1,r1] = runs[k];
                auto [l2,r2] = runs[k+1];
                merge_runs(a, l1, r1, r2, buf);
                next.emplace_back(l1, r2);
            }
            if (runs.size() & 1) next.push_back(runs.back());
            runs.swap(next);
        }
    }
}

//--------------------------- Helpers: random bases ----------------------------
static std::vector<int> make_random_vector(size_t n, uint64_t seed) {
    std::vector<int> v(n);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> dist(1, (int)n);
    for (size_t i = 0; i < n; ++i) v[i] = dist(rng);
    return v;
}

//--------------------------- Benchmark harness (time only) --------------------
struct Result {
    std::string name;
    double avg_seconds = 0.0;
    double std_seconds = 0.0;
    bool ok = true;
};

template<typename Fn>
static Result bench_algo_random(const std::string& name,
                                size_t n,
                                int rounds,
                                uint64_t seed_base,
                                Fn fn)
{
    Result res; res.name = name;

    // warm-up (not timed)
    {
        auto base = make_random_vector(n, seed_base + 0x9E3779B97F4A7C15ULL * 0);
        std::vector<int> v = base;
        fn(v);
        consume(v);
    }

    std::vector<double> times; times.reserve(rounds);
    for (int r = 0; r < rounds; ++r) {
        auto base = make_random_vector(n, seed_base + 0x9E3779B97F4A7C15ULL * (uint64_t)r);
        std::vector<int> v = base; // identical base for each algo at this (n, r)
        auto t0 = std::chrono::steady_clock::now();
        fn(v); // ALL allocations happen inside here
        auto t1 = std::chrono::steady_clock::now();
        consume(v);
        double dt = std::chrono::duration<double>(t1 - t0).count();
        times.push_back(dt);
        res.ok = res.ok && std::is_sorted(v.begin(), v.end());
    }

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / times.size();
    double acc = 0.0; for (double t : times) { double d = t - mean; acc += d*d; }
    double stdev = std::sqrt(acc / times.size());
    res.avg_seconds = mean; res.std_seconds = stdev;
    return res;
}

static void print_result(const Result& r) {
    std::cout << std::left << std::setw(14) << r.name
              << " avg: " << std::fixed << std::setprecision(6) << r.avg_seconds
              << " s  (±" << std::setprecision(6) << r.std_seconds << ")"
              << (r.ok ? "" : "  (! not sorted)") << "\n";
}

//----------------------------------- Main -------------------------------------
struct BenchCase { size_t n; int rounds; };

int main() {
    using namespace std;
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << std::unitbuf;

    // Now: 10 rounds for EVERY size (including 100M)
    const vector<BenchCase> cases = {
        {1'000'000ULL,   10},
        {10'000'000ULL,  10},
        {100'000'000ULL, 10}
    };

    for (const auto& C : cases) {
        const size_t n = C.n;
        const int rounds = C.rounds;
        const uint64_t seed_base = 0x123456789ABCDEF0ULL ^ (uint64_t)n; // same across algos

        std::cout << "\n=== Random array, n=" << n << ", rounds=" << rounds << " ===\n\n";

        auto r_ladder = bench_algo_random("LadderSort", n, rounds, seed_base, [](std::vector<int>& v){
            v = ladder_sort(v);                          // allocs counted
        });

        auto r_pat = bench_algo_random("PatienceSort", n, rounds, seed_base, [](std::vector<int>& v){
            v = patience_sort(v);                        // allocs counted
        });

        auto r_tims = bench_algo_random("Timsort-lite", n, rounds, seed_base, [](std::vector<int>& v){
            std::vector<int> buf;                        // alloc inside timing
            timsort_lite::sort_with_buf(v, buf);
        });

        auto r_quick = bench_algo_random("Quicksort", n, rounds, seed_base, [](std::vector<int>& v){
            quicksort3(v);                               // in-place
        });

        auto r_intro = bench_algo_random("Introsort", n, rounds, seed_base, [](std::vector<int>& v){
            std::sort(v.begin(), v.end());               // in-place
        });

        auto r_stable = bench_algo_random("StableSort", n, rounds, seed_base, [](std::vector<int>& v){
            std::stable_sort(v.begin(), v.end());        // library allocs counted
        });

        auto r_merge = bench_algo_random("MergeSort", n, rounds, seed_base, [](std::vector<int>& v){
            std::vector<int> buf;                        // alloc inside timing
            mergesort_with_buf(v, buf);
        });

        std::cout << "Results (Random only):\n";
        print_result(r_ladder);
        print_result(r_pat);
        print_result(r_tims);
        print_result(r_quick);
        print_result(r_intro);
        print_result(r_stable);
        print_result(r_merge);
    }

    if (g_sink64 == 0xdeadbeefULL) std::cerr << "";
    return 0;
}