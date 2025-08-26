// g++ -O3 -march=native -flto -funroll-loops -fomit-frame-pointer -DNDEBUG -std=c++17 bench_postinsert_multi.cpp -o bench_postinsert_multi
#include <algorithm>
#include <chrono>
#include <gfx/timsort.hpp>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include <climits>
#include <iterator>

using std::size_t;

//--------------------------- DCE sink (avoid optimizer) -----------------------
static volatile uint64_t g_sink64 = 0;
template<typename Vec>
inline void consume(const Vec& v) {
    uint64_t s = 0; for (auto x : v) s += (uint64_t)x;
    g_sink64 ^= s;
}

//======================== LadderSort workspaces (reused) ======================
static std::vector<std::vector<int>> g_lad_ws;        // runs
static std::vector<int>              g_tops_ws;       // run tails (non-increasing)
static std::vector<int>              g_ladder_out_ws; // output buffer

//------------------------ Hinted search over flat `tops` ----------------------
// We want the RIGHTMOST index i with tops[i] <= x  (tops is non-increasing).
inline int hinted_rightmost_le(const std::vector<int>& tops, int x, int last_idx_hint) {
    int n = (int)tops.size();
    if (n == 0) return 0; // place at 0 (will become new run)

    int i = last_idx_hint;
    if (i < 0) i = 0;
    if (i >= n) i = n - 1;

    int lo, hi;

    if (tops[i] <= x) {
        // gallop right over region with <= x
        lo = i; hi = lo + 1;
        int step = 1;
        while (hi < n && tops[hi] <= x) {
            lo = hi;
            hi = lo + step;
            step <<= 1;
        }
        if (hi > n) hi = n;

        // binary search last <= x in [lo, hi)
        while (lo + 1 < hi) {
            int mid = (lo + hi) / 2;
            if (tops[mid] <= x) lo = mid; else hi = mid;
        }
        return lo;
    } else {
        // need the first index where tops[idx] <= x (to the right)
        lo = i + 1; hi = lo;
        int step = 1;
        while (hi < n && tops[hi] > x) {
            lo = hi;
            hi = lo + step;
            step <<= 1;
        }
        if (hi > n) hi = n;
        // now find first <= x in [lo, hi)
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (tops[mid] > x) lo = mid + 1; else hi = mid;
        }
        // we want the RIGHTMOST <= x; extend if there are equals to the right
        int idx = lo;
        while (idx + 1 < n && tops[idx + 1] <= x) ++idx;
        return idx;
    }
}

//======================== Merge primitives (for LadderSort) ===================
// Gallop helpers
static inline size_t gallop_right_le(const std::vector<int>& V, size_t lo, int key) {
    const size_t n = V.size();
    size_t step = 1, hi = lo;
    while (hi + step < n && V[hi + step] <= key) step <<= 1;
    size_t L = lo, R = std::min(hi + step, n);
    while (L < R) { size_t m = (L + R) / 2; if (V[m] <= key) L = m + 1; else R = m; }
    return L;
}
static inline size_t gallop_left_lt(const std::vector<int>& V, size_t lo, int key) {
    const size_t n = V.size();
    size_t step = 1, hi = lo;
    while (hi + step < n && V[hi + step] < key) step <<= 1;
    size_t L = lo, R = std::min(hi + step, n);
    while (L < R) { size_t m = (L + R) / 2; if (V[m] < key) L = m + 1; else R = m; }
    return L;
}

// 2-way merge with adaptive galloping
static void merge_two_gallop(const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& out) {
    out.clear();
    out.reserve(A.size() + B.size());
    size_t i = 0, j = 0;
    int winA = 0, winB = 0, GALLOP = 7, calm = 0;

    while (i < A.size() && j < B.size()) {
        if (A[i] <= B[j]) {
            out.push_back(A[i++]);
            if (++winA >= GALLOP && i < A.size()) {
                size_t ni = gallop_right_le(A, i, B[j]);
                size_t took = ni - i;
                if (took) {
                    out.insert(out.end(), A.begin()+i, A.begin()+ni);
                    i = ni; winA = winB = 0;
                    if (took >= 8) { GALLOP = std::max(4, GALLOP - 1); calm = 0; }
                    else if (++calm > 64) { GALLOP = std::min(16, GALLOP + 1); calm = 0; }
                }
            }
        } else {
            out.push_back(B[j++]);
            if (++winB >= GALLOP && j < B.size()) {
                size_t nj = gallop_left_lt(B, j, (i < A.size() ? A[i] : INT_MAX));
                size_t took = nj - j;
                if (took) {
                    out.insert(out.end(), B.begin()+j, B.begin()+nj);
                    j = nj; winA = winB = 0;
                    if (took >= 8) { GALLOP = std::max(4, GALLOP - 1); calm = 0; }
                    else if (++calm > 64) { GALLOP = std::min(16, GALLOP + 1); calm = 0; }
                }
            }
        }
    }
    if (i < A.size()) out.insert(out.end(), A.begin()+i, A.end());
    if (j < B.size()) out.insert(out.end(), B.begin()+j, B.end());
}

// Balanced pairwise merges with galloping (good for small K)
static void merge_pairwise(std::vector<std::vector<int>> runs, std::vector<int>& out) {
    if (runs.empty()) { out.clear(); return; }
    if (runs.size() == 1) { out = std::move(runs[0]); return; }
    std::vector<std::vector<int>> next;
    std::vector<int> tmp;
    while (runs.size() > 1) {
        next.clear();
        next.reserve((runs.size() + 1) / 2);
        for (size_t i = 0; i < runs.size(); i += 2) {
            if (i + 1 < runs.size()) {
                merge_two_gallop(runs[i], runs[i + 1], tmp);
                next.emplace_back().swap(tmp);
            } else {
                next.emplace_back(std::move(runs[i]));
            }
        }
        runs.swap(next);
    }
    out = std::move(runs[0]);
}

// Robust k-way merge via a stable min-heap (fixes dup-heavy correctness)
struct HeapItem {
    int val;
    int run;
    size_t idx;
    bool operator>(const HeapItem& o) const {
        if (val != o.val) return val > o.val;
        return run > o.run; // stable tie: lower run id wins
    }
};

static void merge_k_heap(const std::vector<std::vector<int>>& runs, std::vector<int>& out) {
    size_t total = 0; for (auto& r : runs) total += r.size();
    out.clear(); out.reserve(total);
    std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<HeapItem>> pq;
    for (int r = 0; r < (int)runs.size(); ++r)
        if (!runs[r].empty()) pq.push({runs[r][0], r, 0});

    while (!pq.empty()) {
        auto it = pq.top(); pq.pop();
        out.push_back(it.val);
        size_t ni = it.idx + 1;
        const auto& R = runs[it.run];
        if (ni < R.size()) pq.push({R[ni], it.run, ni});
    }
}

//============================= LadderSort (into buffer) =======================
// Natural-run chunking + adaptive merge-engine choice + rightmost placement
static void ladder_sort_into(const std::vector<int>& a, std::vector<int>& out) {
    if (a.empty()) { out.clear(); return; }

    auto& lad  = g_lad_ws;  lad.clear();  lad.reserve(256);
    auto& tops = g_tops_ws; tops.clear(); tops.reserve(256);

    int last_idx = 0;
    const int n = (int)a.size();

    int i = 0;
    while (i < n) {
        // detect a maximal non-decreasing chunk [i .. j)
        int j = i + 1;
        while (j < n && a[j] >= a[j - 1]) ++j;

        int x0  = a[i];
        int idx = hinted_rightmost_le(tops, x0, last_idx);
        if (idx == (int)lad.size()) {
            lad.emplace_back();
            tops.emplace_back(INT_MAX); // placeholder; will be overwritten
        }
        auto& R = lad[idx];
        R.insert(R.end(), a.begin() + i, a.begin() + j);
        tops[idx] = R.back(); // non-increasing invariant on tops

        last_idx = idx;
        i = j;
    }

    const size_t K = lad.size();
    if (K == 1) { out = lad[0]; return; }
    if (K == 2) { merge_two_gallop(lad[0], lad[1], out); return; }
    if (K <= 8) { merge_pairwise(lad, out); return; }
    merge_k_heap(lad, out); // robust for dup-heavy & large K
}

//=========================== Quicksort / Introsort / Mergesort ================
static inline int median3(int a, int b, int c) {
    if (a < b) { if (b < c) return b; return (a < c) ? c : a; }
    else { if (a < c) return a; return (b < c) ? c : b; }
}
static inline int tukeys_ninther(const std::vector<int>& A, int l, int r) {
    int n = r - l + 1;
    int step = n / 8;
    int a1 = A[l],               a2 = A[l + step],          a3 = A[l + 2*step];
    int b1 = A[l + n/2 - step],  b2 = A[l + n/2],           b3 = A[l + n/2 + step];
    int c1 = A[r - 2*step],      c2 = A[r - step],          c3 = A[r];
    int m1 = median3(a1,a2,a3);
    int m2 = median3(b1,b2,b3);
    int m3 = median3(c1,c2,c3);
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
    st.push_back({0, (int)a.size()-1});
    while (!st.empty()) {
        auto [l,r] = st.back(); st.pop_back();
        while (l < r) {
            if (r - l + 1 <= SMALL) { insertion_sort_range(a, l, r); break; }
            int n = r - l + 1;
            int pivot = (n >= 128) ? tukeys_ninther(a, l, r)
                                   : median3(a[l], a[l + (n>>1)], a[r]);
            int i = l, j = r, k = l;
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

//=========================== Minimal (faithful) Timsort =======================
namespace timsort {
inline void timsort(std::vector<int>& a) { gfx::timsort(a.begin(), a.end()); }
}

//=========================== Dataset (dup-heavy) ==============================
static std::vector<int> generate_dataset(size_t N, uint64_t seed) {
    const int M = 1000; // cardinality
    std::vector<int> out; out.reserve(N);
    uint64_t x = seed ? seed : 0x9E3779B97F4A7C15ULL;
    auto rnd = [&]() -> uint64_t {
        x ^= x << 7;  x ^= x >> 9;  x *= 0x2545F4914F6CDD1ULL;
        return x;
    };
    for (size_t i = 0; i < N; ++i) out.push_back((int)(rnd() % M));
    return out;
}

//=========================== Benchmark harness (time only) ====================
struct Result {
    std::string name;
    double avg_seconds = 0.0;
    double std_seconds = 0.0;
    bool ok = true;
};
template<typename Fn>
static Result bench_algo_postinsert(const std::string& name,
                                    const std::vector<int>& base,
                                    int rounds,
                                    Fn fn)
{
    Result res; res.name = name;
    { std::vector<int> v = base; v.push_back(-1); fn(v); consume(v); } // warm-up

    std::vector<double> times; times.reserve(rounds);
    for (int r = 0; r < rounds; ++r) {
        std::vector<int> v = base;
        v.push_back(-1);
        auto t0 = std::chrono::steady_clock::now();
        fn(v);
        auto t1 = std::chrono::steady_clock::now();
        consume(v);
        times.push_back(std::chrono::duration<double>(t1 - t0).count());
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
    static std::vector<int> g_merge_buf;

    for (const auto& C : cases) {
        const size_t n = C.n, rounds = C.rounds;
        std::vector<int> base = generate_dataset(n, 0xC0FFEEULL);

        std::cout << "\n=== Post-insert case (dup-heavy + push_back(-1)), n=" << n
                  << ", rounds=" << rounds << " ===\n\n";

        auto r_ladder = bench_algo_postinsert("LadderSort", base, rounds, [](std::vector<int>& v){
            ladder_sort_into(v, g_ladder_out_ws);
            v.swap(g_ladder_out_ws);
        });
        auto r_tims = bench_algo_postinsert("Timsort", base, rounds, [&](std::vector<int>& v){
            timsort::timsort(v);
        });
        auto r_quick = bench_algo_postinsert("Quicksort", base, rounds, [](std::vector<int>& v){
            quicksort3(v);
        });
        auto r_intro = bench_algo_postinsert("Introsort", base, rounds, [](std::vector<int>& v){
            std::sort(v.begin(), v.end());
        });
        auto r_stable = bench_algo_postinsert("StableSort", base, rounds, [](std::vector<int>& v){
            std::stable_sort(v.begin(), v.end());
        });
        auto r_merge = bench_algo_postinsert("MergeSort", base, rounds, [&](std::vector<int>& v){
            mergesort_with_buf(v, g_merge_buf);
        });

        std::cout << "Results (Post-insert only):\n";
        print_result(r_ladder);
        print_result(r_tims);
        print_result(r_quick);
        print_result(r_intro);
        print_result(r_stable);
        print_result(r_merge);
    }

    if (g_sink64 == 0xdeadbeefULL) std::cerr << "";
    return 0;
}
