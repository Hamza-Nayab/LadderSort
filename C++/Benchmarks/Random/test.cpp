// g++ -O3 -std=c++17 -march=native bench_postinsert_multi.cpp -o bench_postinsert_multi
#include <algorithm>
#include <chrono>
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
static std::vector<int>              g_tops_ws;  // run tails
static std::vector<int>              g_ladder_out_ws; // output buffer

//------------------------ Hinted search over flat `tops` ----------------------
// Hinted lower_bound with exponential "gallop" around the hint.
// Fast when the index moves slowly (your best cases), but still O(log K)
// on random inputs.
inline int hinted_lower_bound_lad(const std::vector<int>& tops, int x, int last_idx_hint) {
    int n = (int)tops.size();
    if (n == 0) return 0;

    // Clamp the hint
    int i = last_idx_hint;
    if (i < 0) i = 0;
    if (i >= n) i = n - 1;

    // If the hint is already correct, use it.
    if ((i == 0 || tops[i - 1] > x) && tops[i] <= x) return i;

    int lo, hi;

    if (tops[i] > x) {
        // Need a larger index -> gallop to the right
        lo = i + 1;
        hi = lo;
        int step = 1;
        while (hi < n && tops[hi] > x) {
            lo = hi;
            hi = lo + step;
            step <<= 1;
        }
        if (hi > n) hi = n;
    } else {
        // Need a smaller index -> gallop to the left
        hi = i + 1;
        lo = i;
        int step = 1;
        while (lo > 0 && tops[lo - 1] <= x) {
            hi = lo;
            lo -= step;
            if (lo < 0) { lo = 0; break; }
            step <<= 1;
        }
        if (lo < 0) lo = 0;
    }

    // Finish with binary search in the narrowed window [lo, hi)
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (tops[mid] > x) lo = mid + 1;
        else               hi = mid;
    }
    return lo;
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
    std::vector<int> key;                 // current keys per stream
    std::vector<const int*> cur, end;     // cursors

    explicit LoserTree(const std::vector<std::vector<int>>& runs) {
        k = (int)runs.size();
        tree.assign(k, -1);
        key.resize(k);
        cur.resize(k);
        end.resize(k);
        for (int i = 0; i < k; ++i) {
            cur[i] = runs[i].data();
            end[i] = runs[i].data() + runs[i].size();
            key[i] = (cur[i] < end[i]) ? *cur[i] : INT_MAX;
        }
        for (int i = 0; i < k; ++i) adjust(i);
    }
// --- REPLACE your LoserTree::adjust with this ---
inline void adjust(int s) {
    int t = s; // current winner candidate (valid index)
    for (int parent = (s + k) >> 1; parent > 0; parent >>= 1) {
        int& l = tree[parent - 1];   // stores the loser at this node
        if (l < 0) {
            // empty slot: just place current competitor here; keep bubbling the same winner 't'
            l = t;
        } else if (key[t] >= key[l]) {
            // 't' loses (>= to keep stability on ties), store loser and keep bubbling the winner
            std::swap(t, l);
        }
        // else: t < l  -> 'l' loses and is already stored; keep 't' as winner bubbling up
    }
    tree[0] = t; // final winner index (never -1)
}

inline int pop_and_advance() {
    int s = tree[0];
    if (s < 0) return INT_MAX; // defensive guard
    int v = key[s];
    if (++cur[s] < end[s]) key[s] = *cur[s]; else key[s] = INT_MAX;
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
    tops.push_back(a[0]);

    int last_idx = 0;
    for (int i = 1, n = (int)a.size(); i < n; ++i) {
        int x = a[i];
        int idx = hinted_lower_bound_lad(tops, x, last_idx);
        if (idx == (int)lad.size()) {
            lad.emplace_back().emplace_back(x);
            tops.emplace_back(x);
        } else {
            lad[idx].emplace_back(x);
            tops[idx] = x;
        }
        last_idx = idx;
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

//=========================== Minimal (faithful) Timsort =======================
//=========================== MIAU TimSort (faithful) =======================
//=========================== Proper TimSort (fixed, faithful) =================
//=========================== Proper TimSort (robust & stable) =================
namespace timsort {

struct Run { int base; int len; };

// Tim Peters' minrun calculation
static inline std::size_t minrun_len(std::size_t n) {
    std::size_t r = 0;
    while (n >= 64) { r |= (n & 1u); n >>= 1u; }
    return n + r; // in [32,64]
}

// Detect a natural run starting at lo, reverse if descending, return length.
static int count_run_and_make_ascending(std::vector<int>& a, int lo, int hi) {
    int run_hi = lo + 1;
    if (run_hi >= hi) return 1;

    if (a[run_hi] < a[lo]) { // strictly descending
        while (++run_hi < hi && a[run_hi] < a[run_hi - 1]) {}
        std::reverse(a.begin() + lo, a.begin() + run_hi);
    } else { // nondecreasing (stable on ties)
        while (++run_hi < hi && a[run_hi] >= a[run_hi - 1]) {}
    }
    return run_hi - lo;
}

// Stable binary insertion sort on [first,last), assuming [first,start) is sorted.
static inline void binary_insertion_sort(std::vector<int>& a, int first, int last, int start) {
    if (first == start) ++start;
    for (int i = start; i < last; ++i) {
        int x = a[i];
        int lo = first, hi = i;
        while (lo < hi) {
            int mid = lo + ((hi - lo) >> 1);
            if (a[mid] <= x) lo = mid + 1; else hi = mid;  // stable
        }
        for (int j = i; j > lo; --j) a[j] = a[j - 1];
        a[lo] = x;
    }
}

// Merge when len1 <= len2, adjacent runs: [b1,b1+len1) and [b2,b2+len2)
static void merge_lo(std::vector<int>& a, int b1, int len1, int b2, int len2) {
    std::vector<int> tmp(len1);
    std::copy(a.begin() + b1, a.begin() + b1 + len1, tmp.begin());

    int i = 0;      // tmp (left)
    int j = b2;     // a   (right)
    int k = b1;     // dest

    while (i < len1 && j < b2 + len2) {
        if (a[j] < tmp[i]) a[k++] = a[j++];   // right strictly smaller first
        else               a[k++] = tmp[i++]; // left (or equal) to keep stability
    }
    if (i < len1) std::copy(tmp.begin() + i, tmp.end(), a.begin() + k);
    // if right remains: already in place
}

// Merge when len1 > len2, adjacent runs: merge from the back
static void merge_hi(std::vector<int>& a, int b1, int len1, int b2, int len2) {
    std::vector<int> tmp(len2);
    std::copy(a.begin() + b2, a.begin() + b2 + len2, tmp.begin());

    int i = b1 + len1 - 1; // left end
    int j = len2 - 1;      // tmp end
    int k = b2 + len2 - 1; // dest end

    while (i >= b1 && j >= 0) {
        if (tmp[j] < a[i]) a[k--] = a[i--];
        else               a[k--] = tmp[j--]; // right (or equal) wins to keep stability
    }
    if (j >= 0) std::copy(tmp.begin(), tmp.begin() + (j + 1), a.begin() + (k - j));
}

// Merge adjacent runs [base1, base1+len1) and [base2, base2+len2), with conservative trimming.
// NOTE: This function trims the *work* but the caller must keep the *original* total length.
static void merge_at(std::vector<int>& a, int base1, int len1, int base2, int len2) {
    if (len1 == 0 || len2 == 0) return;

    // Trim from left end of left run (skip <= first right) — stable
    {
        int k = int(std::upper_bound(a.begin() + base1,
                                     a.begin() + base1 + len1,
                                     a[base2]) - (a.begin() + base1));
        base1 += k; len1 -= k;
        if (len1 == 0) return;
    }
    // Trim from right end of right run (keep only < last left) — stable
    {
        int j = int(std::lower_bound(a.begin() + base2,
                                     a.begin() + base2 + len2,
                                     a[base1 + len1 - 1]) - (a.begin() + base2));
        len2 = j;
        if (len2 == 0) return;
    }

    if (len1 <= len2) merge_lo(a, base1, len1, base2, len2);
    else              merge_hi(a, base1, len1, base2, len2);
}

// TimSort stack invariants (CPython/Java)
static inline bool collapse_needed(const std::vector<Run>& s) {
    int n = (int)s.size();
    if (n <= 1) return false;
    if (n == 2) return s[0].len <= s[1].len;
    int A = s[n - 3].len, B = s[n - 2].len, C = s[n - 1].len;
    return (A <= B + C) || (B <= C);
}

static inline int pick_merge_idx(const std::vector<Run>& s) {
    int n = (int)s.size();
    if (n >= 3) {
        int A = s[n - 3].len, C = s[n - 1].len;
        return (A < C) ? (n - 3) : (n - 2);
    }
    return n - 2;
}

static void merge_stack_top(std::vector<int>& a, std::vector<Run>& st, int idx) {
    int base1 = st[idx].base;
    int len1  = st[idx].len;
    int base2 = st[idx + 1].base;
    int len2  = st[idx + 1].len;

    // DO THE WORK on trimmed subranges…
    merge_at(a, base1, len1, base2, len2);

    // …but KEEP the full merged span on the stack:
    // base stays at original base1; length is the *original* len1+len2.
    st[idx].len  = st[idx].len + st[idx + 1].len;  // (not the trimmed len1+len2)
    // st[idx].base unchanged
    st.erase(st.begin() + idx + 1);
}

static void merge_collapse(std::vector<int>& a, std::vector<Run>& st) {
    while (collapse_needed(st)) {
        int i = pick_merge_idx(st);
        merge_stack_top(a, st, i);
    }
}

static void merge_force_collapse(std::vector<int>& a, std::vector<Run>& st) {
    while (st.size() > 1) {
        int n = (int)st.size();
        int i = (n >= 3 && st[n - 3].len < st[n - 1].len) ? (n - 3) : (n - 2);
        merge_stack_top(a, st, i);
    }
}

static void timsort(std::vector<int>& a) {
    const int n = (int)a.size();
    if (n < 2) return;

    const int minrun = (int)minrun_len((std::size_t)n);
    std::vector<Run> st; st.reserve((n + minrun - 1) / minrun);

    int lo = 0;
    while (lo < n) {
        int run_len = count_run_and_make_ascending(a, lo, n);
        int need = (run_len < minrun) ? std::min(minrun, n - lo) : run_len;
        binary_insertion_sort(a, lo, lo + need, lo + run_len); // extend to minrun
        st.push_back({lo, need});
        lo += need;
        merge_collapse(a, st);
    }
    merge_force_collapse(a, st);
}

// Iterator overload: sort into a temp then move back (keeps API identical)
template<class RandomIt>
void timsort(RandomIt first, RandomIt last) {
    using T = typename std::iterator_traits<RandomIt>::value_type;
    std::vector<T> tmp(first, last);
    timsort(tmp);
    std::move(tmp.begin(), tmp.end(), first);
}

} // namespace timsort

// Sticky K-way interleaving (bursty sources, small K)
// Uniform random integers (i.i.d., duplicates allowed)
static std::vector<int> generate_dataset(size_t N, uint64_t seed) {
    std::vector<int> out; out.reserve(N);

    // tiny xorshift64* PRNG (no <random> dependency)
    uint64_t x = seed ? seed : 0x9E3779B97F4A7C15ULL;
    auto rnd = [&]() -> uint64_t {
        x ^= x << 7;  x ^= x >> 9;  x *= 0x2545F4914F6CDD1DULL;
        return x;
    };

    for (size_t i = 0; i < N; ++i) {
        // keep values >= 0 so your push_back(-1) is the new minimum
        out.push_back((int)(rnd() & 0x7fffffffULL)); // 0 .. 2^31-1
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
                                    const std::vector<int>& base, // size n (not necessarily sorted)
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
        v.push_back(-1); // post-insert smaller element at the end
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

    // 10 rounds for EVERY size (1M, 10M, 100M)
    const vector<BenchCase> cases = {
        {1'000'000ULL,   10},
        {10'000'000ULL,  10},
        {100'000'000ULL, 10}
    };

    static std::vector<int> g_merge_buf; // (kept for mergesort variant)

    for (const auto& C : cases) {
        const size_t n = C.n;
        const int rounds = C.rounds;

        // Base dataset: Strided-by-K (K=64) then post-insert -1 per trial
        std::vector<int> base = generate_dataset(n, 0xC0FFEEULL);

        std::cout << "\n=== Post-insert case (Strided-by-64 + push_back(-1)), n=" << n
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