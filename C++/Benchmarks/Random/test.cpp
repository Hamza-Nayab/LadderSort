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

//======================== LadderSort (optimized merge) ========================
// place into the leftmost ladder whose tail <= target
inline int find_ladder_index(const std::vector<int>& tops, int target) {
    int low = 0, high = (int)tops.size();
    while (low < high) {
        int mid = (low + high) / 2;
        if (tops[mid] > target) low = mid + 1; else high = mid;
    }
    return low;
}

struct Run { const int* cur; const int* end; };
struct RunGreater { bool operator()(const Run& a, const Run& b) const { return *a.cur > *b.cur; } };

static std::vector<int> merge_ladders(const std::vector<std::vector<int>>& lists) {
    size_t total = 0; for (const auto& v : lists) total += v.size();
    std::vector<int> out; out.reserve(total);
    std::vector<Run> heap_store; heap_store.reserve(lists.size());
    std::priority_queue<Run, std::vector<Run>, RunGreater> heap(RunGreater{}, std::move(heap_store));
    for (const auto& v : lists) if (!v.empty()) heap.push(Run{v.data(), v.data()+v.size()});
    while (!heap.empty()) {
        Run r = heap.top(); heap.pop();
        out.push_back(*r.cur++);
        if (r.cur != r.end) heap.push(r);
    }
    return out;
}

static std::vector<int> ladder_sort(const std::vector<int>& a) {
    if (a.empty()) return {};
    std::vector<std::vector<int>> lad; lad.reserve(64);
    std::vector<int> tops; tops.reserve(64);
    lad.push_back({a[0]}); tops.push_back(a[0]);
    for (int i = 1, n = (int)a.size(); i < n; ++i) {
        int x = a[i];
        int idx = find_ladder_index(tops, x);
        if (idx == (int)lad.size()) { lad.emplace_back().emplace_back(x); tops.emplace_back(x); }
        else { lad[idx].emplace_back(x); tops[idx] = x; }
    }
    if (lad.size() == 1) return lad[0];
    return merge_ladders(lad);
}

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

    // warm-up (not timed) with round 0 base
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
        fn(v);
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
    cout << std::unitbuf; // show progress line-by-line

    // Cases: 1M (10), 10M (5), 100M (3)
    const vector<BenchCase> cases = {
        {1'000'000ULL,   10},
        {10'000'000ULL,   5},
        {100'000'000ULL,  3}
    };

    // reusable scratch buffers (so we don't pay alloc every call)
    static std::vector<int> g_merge_buf;
    static std::vector<int> g_tims_buf;

    for (const auto& C : cases) {
        const size_t n = C.n;
        const int rounds = C.rounds;
        const uint64_t seed_base = 0x123456789ABCDEF0ULL ^ (uint64_t)n; // same across algos

        std::cout << "\n=== Random array, n=" << n << ", rounds=" << rounds << " ===\n\n";

        auto r_ladder = bench_algo_random("LadderSort", n, rounds, seed_base, [](std::vector<int>& v){
            v = ladder_sort(v);
        });

        auto r_tims = bench_algo_random("Timsort-lite", n, rounds, seed_base, [&](std::vector<int>& v){
            timsort_lite::sort_with_buf(v, g_tims_buf);
        });

        auto r_quick = bench_algo_random("Quicksort", n, rounds, seed_base, [](std::vector<int>& v){
            quicksort3(v);
        });

        auto r_intro = bench_algo_random("Introsort", n, rounds, seed_base, [](std::vector<int>& v){
            std::sort(v.begin(), v.end());
        });

        auto r_stable = bench_algo_random("StableSort", n, rounds, seed_base, [](std::vector<int>& v){
            std::stable_sort(v.begin(), v.end());
        });

        auto r_merge = bench_algo_random("MergeSort", n, rounds, seed_base, [&](std::vector<int>& v){
            mergesort_with_buf(v, g_merge_buf);
        });

        std::cout << "Results (Random only):\n";
        print_result(r_ladder);
        print_result(r_tims);
        print_result(r_quick);   // Quicksort measured in every scenario, as requested
        print_result(r_intro);
        print_result(r_stable);
        print_result(r_merge);
    }

    if (g_sink64 == 0xdeadbeefULL) std::cerr << "";
    return 0;
}
