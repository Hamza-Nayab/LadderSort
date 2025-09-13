// g++ -O3 -std=c++17 -march=native bench_postinsert_multi.cpp -o bench_postinsert_multi
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
static std::vector<std::vector<int>> g_lad_ws;   // runs
static std::vector<int>              g_tops_ws;  // run tails (non-increasing)
static std::vector<int>              g_ladder_out_ws; // output buffer

//------------------------ Hinted search over flat `tops` ----------------------
// Find first i with tops[i] <= x, given tops is NON-INCREASING.
// Uses a local ±1 probe, then exponential gallop, then binary search.
// Robust on giant equal plateaus; stable on ties.
inline int hinted_lower_bound_lad(const std::vector<int>& tops, int x, int hint) {
    int n = (int)tops.size();
    if (n == 0) return 0;

    int i = hint;
    if (i < 0)   i = 0;
    if (i >= n)  i = n - 1;

    // Fast paths around the hint
    if ((i == 0 || tops[i - 1] > x) && tops[i] <= x) return i;               // already correct
    if (i + 1 < n && tops[i] > x && tops[i + 1] <= x) return i + 1;          // moved right by 1
    if (i > 0   && tops[i - 1] <= x && (i == 1 || tops[i - 2] > x)) return i - 1; // moved left by 1

    if (tops[i] > x) {
        // Need a larger index (move right)
        int last = i, ofs = 1;
        while (i + ofs < n && tops[i + ofs] > x) { last = i + ofs; ofs = (ofs << 1) + 1; }
        int lo = last + 1;
        int hi = std::min(i + ofs, n - 1);
        while (lo <= hi) { // first j with tops[j] <= x
            int mid = lo + ((hi - lo) >> 1);
            if (tops[mid] > x) lo = mid + 1; else hi = mid - 1;
        }
        return lo; // may be n
    } else {
        // tops[i] <= x : move left to find the first such position
        int last = i, ofs = 1;
        while (i - ofs >= 0 && tops[i - ofs] <= x) { last = i - ofs; ofs <<= 1; }
        int lo = std::max(0, i - ofs);
        int hi = last;
        while (lo <= hi) {
            int mid = lo + ((hi - lo) >> 1);
            if (tops[mid] > x) lo = mid + 1; else hi = mid - 1;
        }
        return lo;
    }
}

//======================== Merge primitives (for LadderSort) ===================
// 2-way merge with simple galloping
static void merge_two_gallop(const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& out) {
    out.clear();
    out.reserve(A.size() + B.size());
    size_t i = 0, j = 0;
    int winA = 0, winB = 0, GALLOP = 8;

    auto gallop_right = [](const std::vector<int>& V, size_t lo, int key) {
        size_t n = V.size();
        size_t step = 1, hi = lo;
        while (hi + step < n && V[hi + step] <= key) step <<= 1;
        size_t L = lo, R = std::min(hi + step, n);
        while (L < R) { size_t m = (L + R) / 2; if (V[m] <= key) L = m + 1; else R = m; }
        return L;
    };

    while (i < A.size() && j < B.size()) {
        if (A[i] <= B[j]) {
            out.push_back(A[i++]);
            if (++winA >= GALLOP) {
                size_t ni = gallop_right(A, i, B[j]);
                out.insert(out.end(), A.begin()+i, A.begin()+ni);
                i = ni; winA = winB = 0;
            }
        } else {
            out.push_back(B[j++]);
            if (++winB >= GALLOP) {
                size_t step = 1, n = B.size();
                size_t nj = j;
                while (nj + step < n && B[nj + step] < A[i]) step <<= 1;
                size_t L = j, R = std::min(nj + step, n);
                while (L < R) { size_t m = (L + R) / 2; if (B[m] < A[i]) L = m + 1; else R = m; }
                out.insert(out.end(), B.begin()+j, B.begin()+L);
                j = L; winA = winB = 0;
            }
        }
    }
    if (i < A.size()) out.insert(out.end(), A.begin()+i, A.end());
    if (j < B.size()) out.insert(out.end(), B.begin()+j, B.end());
}

//=========================== FIXED Loser-tree for k-way merge =================
struct LoserTree {
    int k;
    std::vector<int> tree;                // losers; size k
    std::vector<int> key;                 // current keys per run
    std::vector<const int*> cur, end;     // cursors
    std::vector<char> alive;              // run has remaining data?

    explicit LoserTree(const std::vector<std::vector<int>>& runs) {
        k = (int)runs.size();
        tree.assign(k, -1);
        key.resize(k);
        cur.resize(k);
        end.resize(k);
        alive.assign(k, 0);

        for (int i = 0; i < k; ++i) {
            cur[i] = runs[i].data();
            end[i] = runs[i].data() + runs[i].size();
            if (cur[i] < end[i]) {
                key[i] = *cur[i];
                alive[i] = 1;
            }
        }
        for (int i = 0; i < k; ++i) if (alive[i]) adjust(i);
    }

    // stable <= with deterministic tie-break on run id
    inline bool less_eq(int a, int b) const {
        if (!alive[a]) return false;
        if (!alive[b]) return true;
        if (key[a] != key[b]) return key[a] < key[b];
        return a < b;
    }

    inline void adjust(int s) {
        int t = s; // winner candidate
        for (int parent = (s + k) >> 1; parent > 0; parent >>= 1) {
            int &los = tree[parent - 1];
            if (los < 0) {
                // fill empty slot; keep current winner bubbling upward
                los = t;
            } else if (!less_eq(t, los)) {
                // t loses -> store it as the loser; keep current winner in t
                std::swap(t, los);
            }
        }
        tree[0] = t; // final winner
    }

    inline int pop_and_advance() {
        int s = tree[0];
        int v = key[s];
        if (++cur[s] < end[s]) {
            key[s] = *cur[s];
        } else {
            alive[s] = 0;    // run exhausted
        }
        adjust(s);
        return v;
    }
};

static void merge_k_loser_tree(const std::vector<std::vector<int>>& runs, std::vector<int>& out) {
    out.clear();
    size_t total = 0; for (const auto& r : runs) total += r.size();
    out.reserve(total);
    if (runs.empty()) return;
    if (runs.size() == 1) { out = runs[0]; return; }

    LoserTree lt(runs);
    for (size_t t = 0; t < total; ++t) out.push_back(lt.pop_and_advance());
}

//============================= LadderSort (into buffer) =======================
static void ladder_sort_into(const std::vector<int>& a, std::vector<int>& out) {
    if (a.empty()) { out.clear(); return; }

    auto& lad  = g_lad_ws;   lad.clear();  lad.reserve(64);
    auto& tops = g_tops_ws;  tops.clear(); tops.reserve(64);

    lad.push_back({a[0]});
    tops.push_back(a[0]);            // tops is kept NON-INCREASING

    int last_idx = 0;
    for (int i = 1, n = (int)a.size(); i < n; ++i) {
        int x = a[i];
        int idx = hinted_lower_bound_lad(tops, x, last_idx); // first j with tops[j] <= x
        if (idx == (int)lad.size()) {
            lad.emplace_back().emplace_back(x);
            tops.emplace_back(x);
        } else {
            lad[idx].emplace_back(x);
            tops[idx] = x;           // update run tail
        }
        last_idx = idx;              // very good hint on smooth inputs
    }

    if (lad.size() == 1) { out = lad[0]; return; }
    if (lad.size() == 2) { merge_two_gallop(lad[0], lad[1], out); return; }
    merge_k_loser_tree(lad, out);
}

//=========================== Quicksort (adaptive pivot) =======================
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

//=========================== Merge sort (stable, top-down) ====================
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

//=========================== Timsort wrapper (gfx) ============================
namespace timsort {
inline void timsort(std::vector<int>& a) { gfx::timsort(a.begin(), a.end()); }
}

//=========================== Datasets ========================================

// CASE: Nearly-sorted with sparse errors (random swaps)
// Kafka/Kinesis multi-partition fan-in (small K, sticky bursts, bounded jitter)
// Consolidated market data (multi-venue feeds)
// - K venues (10..20), each locally time/seq-sorted (non-decreasing).
// - Bursty, sticky interleave to mimic transport jitter.
// - Heavy equality clusters from coarse timestamp buckets.
// - Stability on equal keys matters; your LadderSort (stable k-way merge)
//   will preserve venue tie-order on ties.

// Social feed assembly (per-user fan-in)
// K followees, each locally sorted; bursty, sticky interleave.
// Many equal timestamps create large tie plateaus across runs.

// IoT / telemetry fan-in (dozens of sensors with bounded skew)
// K sensors; per-round base time increases by 1. Each round, a random subset
// of sensors emits; each sensor's timestamp = max(last_ts + inc, base + jitter)
// so per-sensor sequences are non-decreasing. Small jitter => many ties.

// Search-engine partial index merges (few segments, equality-heavy)
// Model: K segments each contain a sorted posting list of docIDs drawn from a
// shared universe U (so segments overlap -> many equal docIDs across runs).
// We then interleave the K sorted runs in sticky bursts to mimic fan-in.
// All values are non-negative so your push_back(-1) is still the global min.

static std::vector<int> generate_dataset(size_t N, uint64_t seed) {
    const int K          = 16;
    const double overlap = 0.5;
    const size_t U       = std::max<size_t>(1, (size_t)(N * overlap));
    const int BURST_MAX  = 32;
    const int STAY_PCT   = 60;
    const int RIGHT_PCT  = 20;
    std::vector<size_t> need(K);
    { size_t q = N / K, r = N % K; for (int k = 0; k < K; ++k) need[k] = q + (k < (int)r ? 1 : 0); }
    uint64_t x = seed ? seed : 0x9E3779B97F4A7C15ULL;
    auto rnd  = [&]() -> uint64_t { x ^= x << 7; x ^= x >> 9; x *= 0x2545F4914F6CDD1DULL; return x; };
    std::vector<std::vector<int>> runs(K);
    for (int k = 0; k < K; ++k) {
        runs[k].reserve(need[k]);
        size_t left = need[k];
        size_t pos  = (size_t)(rnd() % std::max<size_t>(1, U / K));
        while (left--) {
            uint64_t r = rnd();
            size_t gap = 1 + (size_t)(r % 4);
            pos += gap;
            if (pos >= U) pos = (size_t)(rnd() % (U / 2 + 1));
            runs[k].push_back((int)pos);
        }
    }
    std::vector<int> out; out.reserve(N);
    std::vector<size_t> cur(K, 0);
    auto has_more = [&](int s) -> bool { return cur[s] < runs[s].size(); };
    int s = (int)(rnd() % K);
    while (out.size() < N) {
        if (!has_more(s)) {
            int tries = 0;
            while (tries < K && !has_more(s)) { s = (s + 1) % K; ++tries; }
            if (tries == K) break;
        }
        int burst = 1 + (int)(rnd() % BURST_MAX);
        while (burst-- > 0 && out.size() < N && has_more(s)) {
            out.push_back(runs[s][cur[s]++]);
        }
        int toss = (int)(rnd() % 100);
        if      (toss < STAY_PCT) { /* stay */ }
        else if (toss < STAY_PCT + RIGHT_PCT) s = (s + 1) % K;
        else                                  s = (s + K - 1) % K;
    }
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
                                    const std::vector<int>& base, // size n
                                    int rounds,
                                    Fn fn)
{
    Result res; res.name = name;

    // Warm-up (not timed): copy base, then push_back(-1)
    {
        std::vector<int> v = base;
        v.push_back(-1);
        fn(v);
        consume(v);
    }

    std::vector<double> times; times.reserve(rounds);
    for (int r = 0; r < rounds; ++r) {
        std::vector<int> v = base;
        v.push_back(-1); // late out-of-order minimum
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
    cout << std::unitbuf;

    const vector<BenchCase> cases = {
        {1'000'000ULL,   10},
        {10'000'000ULL,  10},
        {100'000'000ULL, 10}
    };

    static std::vector<int> g_merge_buf; // (kept for mergesort variant)

    for (const auto& C : cases) {
        const size_t n = C.n;
        const int rounds = C.rounds;

        std::vector<int> base = generate_dataset(n, 0xC0FFEEULL);

        std::cout << "\n=== Post-insert case (high-duplication + push_back(-1)), n="
                  << n << ", rounds=" << rounds << " ===\n\n";

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
