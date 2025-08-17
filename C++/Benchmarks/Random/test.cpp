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
#include <climits>

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
            tops[low] = x;
        }
    }

    struct Node { int val; int p; };
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

//=========================== Adaptive Quicksort (3-way) =======================
static inline int median3(int a, int b, int c) {
    if (a < b) { if (b < c) return b; return (a < c) ? c : a; }
    else { if (a < c) return a; return (b < c) ? c : b; }
}
static inline int tukeys_ninther(const std::vector<int>& A, int l, int r) {
    int n = r - l + 1;
    int step = std::max(1, n / 8);
    int a1 = A[l],               a2 = A[l + step],          a3 = A[std::min(l + 2*step, r)];
    int b1 = A[l + n/2 - step],  b2 = A[l + n/2],           b3 = A[std::min(l + n/2 + step, r)];
    int c1 = A[std::max(r - 2*step, l)], c2 = A[std::max(r - step, l)], c3 = A[r];
    int m1 = median3(a1,a2,a3), m2 = median3(b1,b2,b3), m3 = median3(c1,c2,c3);
    return median3(m1,m2,m3);
}
static inline void insertion_sort_range(std::vector<int>& a, int l, int r) {
    for (int i = l + 1; i <= r; ++i) {
        int x = a[i], j = i - 1;
        while (j >= l && a[j] > x) { a[j+1] = a[j]; --j; }
        a[j+1] = x;
    }
}
static void quicksort3(std::vector<int>& a) {
    const int SMALL = 32;
    struct Range { int l, r; };
    std::vector<Range> st; st.reserve(64);
    if (a.empty()) return;
    st.push_back({0, (int)a.size()-1});
    while (!st.empty()) {
        auto [l,r] = st.back(); st.pop_back();
        while (l < r) {
            if (r - l + 1 <= SMALL) { insertion_sort_range(a, l, r); break; }
            int n = r - l + 1;
            int pivot = (n >= 128) ? tukeys_ninther(a, l, r)
                                   : median3(a[l], a[l + (n>>1)], a[r]);
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

//================----------- Merge sort (stable, top-down) --------------------
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
    std::copy(buf.begin() + l, buf.begin() + r + 1, a.begin() + l);
}
static void mergesort_with_buf(std::vector<int>& a, std::vector<int>& buf) {
    if (a.empty()) return;
    if ((int)buf.size() != (int)a.size()) buf.assign(a.size(), 0);
    mergesort_rec(a, buf, 0, (int)a.size() - 1);
}

//=========================== Proper Timsort (simple, stable) ==================
namespace timsort {

struct Run { int base; int len; };

static inline int minrun_for(size_t n) {
    int r = 0;
    while (n >= 64) { r |= (n & 1U); n >>= 1U; }
    return (int)n + r;
}

static int count_run_and_make_ascending(std::vector<int>& a, int lo, int hi) {
    int run_hi = lo + 1;
    if (run_hi == hi) return 1;
    if (a[run_hi++] < a[lo]) {
        while (run_hi < hi && a[run_hi] < a[run_hi - 1]) ++run_hi;
        std::reverse(a.begin()+lo, a.begin()+run_hi);
    } else {
        while (run_hi < hi && a[run_hi] >= a[run_hi - 1]) ++run_hi;
    }
    return run_hi - lo;
}

template<typename It>
static inline void binary_insert(It first, It last) {
    for (It i = first + 1; i < last; ++i) {
        auto x = *i;
        It lo = first, hi = i;
        while (lo < hi) {
            It mid = lo + (hi - lo) / 2;
            if (*mid <= x) lo = mid + 1; else hi = mid;
        }
        for (It j = i; j > lo; --j) *j = *(j - 1);
        *lo = x;
    }
}

static inline void ensure_capacity(std::vector<int>& buf, int need) {
    if ((int)buf.size() < need) buf.resize(need);
}

static void merge_lo(std::vector<int>& a, int base1, int len1,
                     int base2, int len2, std::vector<int>& buf)
{
    ensure_capacity(buf, len1);
    std::copy(a.begin()+base1, a.begin()+base1+len1, buf.begin());
    int i = 0, j = base2, k = base1;
    while (i < len1 && j < base2 + len2) {
        if (buf[i] <= a[j]) a[k++] = buf[i++];
        else                a[k++] = a[j++];
    }
    if (i < len1) std::copy(buf.begin()+i, buf.begin()+len1, a.begin()+k);
}

static void merge_hi(std::vector<int>& a, int base1, int len1,
                     int base2, int len2, std::vector<int>& buf)
{
    ensure_capacity(buf, len2);
    std::copy(a.begin()+base2, a.begin()+base2+len2, buf.begin());
    int i = base1 + len1 - 1;
    int j = len2 - 1;
    int k = base2 + len2 - 1;
    while (i >= base1 && j >= 0) {
        if (a[i] > buf[j]) a[k--] = a[i--];
        else               a[k--] = buf[j--];
    }
    if (j >= 0) std::copy(buf.begin(), buf.begin()+j+1, a.begin() + (k - j));
}

static void merge_at(std::vector<int>& a, std::vector<Run>& s, int i, std::vector<int>& buf) {
    int base1 = s[i].base, len1 = s[i].len;
    int base2 = s[i+1].base, len2 = s[i+1].len;

    // Remember full concatenated range for the run stack update:
    const int orig_base = base1;
    const int orig_len  = len1 + len2;

    // Trim borders (cheap scans, stable)
    int j = 0;
    while (j < len1 && a[base1 + j] <= a[base2]) ++j;
    base1 += j; len1 -= j;
    if (len1 == 0) { s[i] = { orig_base, orig_len }; s.erase(s.begin() + i + 1); return; }

    int t = len2 - 1;
    while (t >= 0 && a[base2 + t] >= a[base1 + len1 - 1]) --t;
    len2 = t + 1;
    if (len2 == 0) { s[i] = { orig_base, orig_len }; s.erase(s.begin() + i + 1); return; }

    if (len1 <= len2) merge_lo(a, base1, len1, base2, len2, buf);
    else              merge_hi(a, base1, len1, base2, len2, buf);

    // The merged run covers the *entire* concatenation
    s[i] = { orig_base, orig_len };
    s.erase(s.begin() + i + 1);
}

static bool collapse_needed(const std::vector<Run>& s) {
    int n = (int)s.size();
    if (n <= 1) return false;
    if (n == 2) return s[n-2].len <= s[n-1].len;
    int A = s[n-3].len, B = s[n-2].len, C = s[n-1].len;
    return (A <= B + C) || (B <= C);
}
static int pick_merge_idx(const std::vector<Run>& s) {
    int n = (int)s.size();
    if (n >= 3) {
        int A = s[n-3].len, B = s[n-2].len, C = s[n-1].len;
        return (A < C) ? n-3 : n-2;
    }
    return n-2;
}

static void force_collapse(std::vector<int>& a, std::vector<Run>& s, std::vector<int>& buf) {
    while (s.size() > 1) {
        int i = (s.size() >= 3 && s[s.size()-3].len < s[s.size()-1].len) ? (int)s.size()-3 : (int)s.size()-2;
        merge_at(a, s, i, buf);
    }
}

static void sort_with_buf(std::vector<int>& a, std::vector<int>& buf) {
    const int n = (int)a.size();
    if (n < 2) return;

    int minrun = minrun_for(n);
    std::vector<Run> runs; runs.reserve((n + minrun - 1) / minrun);

    int lo = 0;
    while (lo < n) {
        int run_len = count_run_and_make_ascending(a, lo, n);
        int need = (run_len < minrun) ? std::min(minrun, n - lo) : run_len;
        binary_insert(a.begin()+lo, a.begin()+lo+need);
        runs.push_back({lo, need});
        lo += need;

        while (collapse_needed(runs)) {
            int i = pick_merge_idx(runs);
            merge_at(a, runs, i, buf);
        }
    }
    force_collapse(a, runs, buf);
}

} // namespace timsort

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
            v = ladder_sort(v);
        });

        auto r_pat = bench_algo_random("PatienceSort", n, rounds, seed_base, [](std::vector<int>& v){
            v = patience_sort(v);
        });

        auto r_tims = bench_algo_random("Timsort", n, rounds, seed_base, [](std::vector<int>& v){
            std::vector<int> buf; // alloc inside timing
            timsort::sort_with_buf(v, buf);
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

        auto r_merge = bench_algo_random("MergeSort", n, rounds, seed_base, [](std::vector<int>& v){
            std::vector<int> buf; // alloc inside timing
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
