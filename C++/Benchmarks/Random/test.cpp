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
inline int hinted_lower_bound_lad(const std::vector<int>& tops, int x, int last_idx_hint) {
    int n = (int)tops.size();
    if (n == 0) return 0;
    int idx = last_idx_hint;
    if (idx < 0) idx = 0;
    if (idx >= n) idx = n - 1;

    if (tops[idx] > x) {
        while (idx + 1 < n && tops[idx + 1] > x) ++idx; // move right
    } else {
        while (idx > 0 && tops[idx - 1] <= x) --idx;     // move left
    }
    // Validate; else fallback to binary search
    if (!((idx == 0 || tops[idx - 1] > x) && tops[idx] <= x)) {
        int low = 0, high = n;
        while (low < high) {
            int mid = (low + high) / 2;
            if (tops[mid] > x) low = mid + 1; else high = mid;
        }
        idx = low;
    }
    return idx;
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

// Loser-tree for k-way merge (k >= 3)
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

    inline void adjust(int s) {
        int t = s;
        for (int parent = (s + k) >> 1; parent > 0; parent >>= 1) {
            int& loser = tree[parent - 1];
            if (loser < 0 || key[t] >= key[loser]) std::swap(t, loser);
        }
        tree[0] = t; // winner at root
    }

    inline int pop_and_advance() {
        int s = tree[0];
        int v = key[s];
        if (++cur[s] < end[s]) key[s] = *cur[s]; else key[s] = INT_MAX;
        adjust(s);
        return v;
    }
};

static void merge_k_loser_tree(const std::vector<std::vector<int>>& runs, std::vector<int>& out) {
    size_t total = 0; for (const auto& r : runs) total += r.size();
    out.clear(); out.reserve(total);
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
        // (binary) insertion sort base case
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

    // No early-out here; always merge
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

//=========================== Proper Timsort (stable) ==========================
namespace timsort {
    struct Run { int base; int len; };

    static inline int minrun_for(size_t n) {
        int r = 0;
        while (n >= 64) { r |= (n & 1U); n >>= 1U; }
        return (int)n + r;
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

    // Galloping helpers (return insertion index)
    static int gallop_left(int key, const std::vector<int>& a, int base, int len, int hint) {
        int last_ofs = 0, ofs = 1;
        if (key > a[base + hint]) {
            int max_ofs = len - hint;
            while (ofs < max_ofs && key > a[base + hint + ofs]) { last_ofs = ofs; ofs = (ofs << 1) + 1; }
            if (ofs > max_ofs) ofs = max_ofs;
            last_ofs += hint; ofs += hint;
        } else {
            int max_ofs = hint + 1;
            while (ofs < max_ofs && key <= a[base + hint - ofs]) { last_ofs = ofs; ofs = (ofs << 1) + 1; }
            if (ofs > max_ofs) ofs = max_ofs;
            int tmp = last_ofs; last_ofs = hint - ofs; ofs = hint - tmp;
        }
        ++last_ofs;
        while (last_ofs < ofs) {
            int m = last_ofs + ((ofs - last_ofs) >> 1);
            if (key > a[base + m]) last_ofs = m + 1;
            else ofs = m;
        }
        return ofs;
    }

    static int gallop_right(int key, const std::vector<int>& a, int base, int len, int hint) {
        int last_ofs = 0, ofs = 1;
        if (key < a[base + hint]) {
            int max_ofs = hint + 1;
            while (ofs < max_ofs && key < a[base + hint - ofs]) { last_ofs = ofs; ofs = (ofs << 1) + 1; }
            if (ofs > max_ofs) ofs = max_ofs;
            int tmp = last_ofs; last_ofs = hint - ofs; ofs = hint - tmp;
        } else {
            int max_ofs = len - hint;
            while (ofs < max_ofs && key >= a[base + hint + ofs]) { last_ofs = ofs; ofs = (ofs << 1) + 1; }
            if (ofs > max_ofs) ofs = max_ofs;
            last_ofs += hint; ofs += hint;
        }
        ++last_ofs;
        while (last_ofs < ofs) {
            int m = last_ofs + ((ofs - last_ofs) >> 1);
            if (key < a[base + m]) ofs = m;
            else last_ofs = m + 1;
        }
        return ofs;
    }

    static int min_gallop = 7;

    static void ensure_capacity(std::vector<int>& buf, int need) {
        if ((int)buf.size() < need) buf.resize(need);
    }

    static void merge_lo(std::vector<int>& a, int base1, int len1,
                         int base2, int len2, std::vector<int>& buf) {
        ensure_capacity(buf, len1);
        std::copy(a.begin()+base1, a.begin()+base1+len1, buf.begin());
        int i = 0, j = base2, k = base1;
        if (len1 == 0 || len2 == 0) return;

        int mg = min_gallop;
        while (true) {
            int count1 = 0, count2 = 0;
            // pairwise mode
            do {
                if (a[j] < buf[i]) { a[k++] = a[j++]; ++count2; count1 = 0; if (--len2 == 0) goto copy_left; }
                else               { a[k++] = buf[i++]; ++count1; count2 = 0; if (--len1 == 0) return;      }
            } while ( (count1 | count2) < mg );

            // gallop mode
            while (true) {
                int right_idx = gallop_right(a[j], buf, 0, len1, i);
                int take1 = right_idx - i;
                if (take1) {
                    std::copy(buf.begin()+i, buf.begin()+right_idx, a.begin()+k);
                    k += take1; i = right_idx; len1 -= take1;
                    if (len1 == 0) return;
                }
                a[k++] = a[j++]; if (--len2 == 0) goto copy_left;

                int left_cnt = gallop_left(buf[i], a, j, len2, 0);
                int take2 = left_cnt;
                if (take2) {
                    std::copy(a.begin()+j, a.begin()+j+take2, a.begin()+k);
                    k += take2; j += take2; len2 -= take2;
                    if (len2 == 0) goto copy_left;
                }
                a[k++] = buf[i++]; if (--len1 == 0) return;

                if (take1 < mg && take2 < mg) break; // leave gallop
                if (mg > 1) --mg;
            }
            mg += 2;
        }
    copy_left:
        std::copy(buf.begin()+i, buf.begin()+i+len1, a.begin()+k);
    }

    static void merge_hi(std::vector<int>& a, int base1, int len1,
                         int base2, int len2, std::vector<int>& buf) {
        ensure_capacity(buf, len2);
        std::copy(a.begin()+base2, a.begin()+base2+len2, buf.begin());
        int i = base1 + len1 - 1, j = len2 - 1, k = base2 + len2 - 1;
        if (len1 == 0 || len2 == 0) return;

        int mg = min_gallop;
        while (true) {
            int count1 = 0, count2 = 0;
            // pairwise mode (from the right)
            do {
                if (buf[j] < a[i]) { a[k--] = a[i--]; ++count1; count2 = 0; if (--len1 == 0) { std::copy(buf.begin(), buf.begin()+len2, a.begin() + (k - len2 + 1)); return; } }
                else               { a[k--] = buf[j--]; ++count2; count1 = 0; if (--len2 == 0) return; }
            } while ( (count1 | count2) < mg );

            // gallop mode
            while (true) {
                int gr = len1 - gallop_right(buf[j], a, base1, len1, i - base1);
                int take1 = gr;
                if (take1) {
                    k -= take1; i -= take1;
                    std::copy_backward(a.begin()+i+1, a.begin()+i+1+take1, a.begin()+k+1+take1);
                    len1 -= take1;
                    if (len1 == 0) { std::copy(buf.begin(), buf.begin()+len2, a.begin() + (k - len2 + 1)); return; }
                }
                a[k--] = buf[j--]; if (--len2 == 0) return;

                int gl = len2 - gallop_left(a[i], buf, 0, len2, j);
                int take2 = gl;
                if (take2) {
                    k -= take2; j -= take2;
                    std::copy(buf.begin()+j+1, buf.begin()+j+1+take2, a.begin()+k+1);
                    len2 -= take2;
                    if (len2 == 0) return;
                }
                a[k--] = a[i--]; if (--len1 == 0) { std::copy(buf.begin(), buf.begin()+len2, a.begin() + (k - len2 + 1)); return; }

                if (take1 < mg && take2 < mg) break; // leave gallop
                if (mg > 1) --mg;
            }
            mg += 2;
        }
    }

    static int count_run_and_make_ascending(std::vector<int>& a, int lo, int hi) {
        int run_hi = lo + 1;
        if (run_hi == hi) return 1;
        if (a[run_hi++] < a[lo]) { // descending
            while (run_hi < hi && a[run_hi] < a[run_hi - 1]) ++run_hi;
            std::reverse(a.begin()+lo, a.begin()+run_hi);
        } else { // ascending
            while (run_hi < hi && a[run_hi] >= a[run_hi - 1]) ++run_hi;
        }
        return run_hi - lo;
    }

    static void push_run(std::vector<Run>& stack, int base, int len) {
        stack.push_back({base, len});
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

    static void merge_at(std::vector<int>& a, std::vector<Run>& s, int i, std::vector<int>& buf) {
        int base1 = s[i].base, len1 = s[i].len;
        int base2 = s[i+1].base, len2 = s[i+1].len;

        // Keep original concatenation for the run stack update
        const int orig_base = base1;
        const int orig_len  = len1 + len2;

        // Gallop/trim borders (stable)
        int k = gallop_right(a[base2], a, base1, len1, 0);
        base1 += k; len1 -= k;
        if (len1 == 0) { s[i] = { orig_base, orig_len }; s.erase(s.begin() + i + 1); return; }

        len2 = gallop_left(a[base1 + len1 - 1], a, base2, len2, len2 - 1);
        if (len2 == 0) { s[i] = { orig_base, orig_len }; s.erase(s.begin() + i + 1); return; }

        if (len1 <= len2) merge_lo(a, base1, len1, base2, len2, buf);
        else              merge_hi(a, base1, len1, base2, len2, buf);

        // Merged run covers the entire concatenation
        s[i] = { orig_base, orig_len };
        s.erase(s.begin() + i + 1);
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
            binary_insert(a.begin()+lo, a.begin()+lo+need); // pad to minrun
            push_run(runs, lo, need);
            lo += need;

            while (collapse_needed(runs)) {
                int i = pick_merge_idx(runs);
                merge_at(a, runs, i, buf);
            }
        }
        force_collapse(a, runs, buf);
    }
} // namespace timsort

//=========================== Benchmark harness (time only) ====================
struct Result {
    std::string name;
    double avg_seconds = 0.0;
    double std_seconds = 0.0;
    bool ok = true;
};

template<typename Fn>
static Result bench_algo_postinsert(const std::string& name,
                                    const std::vector<int>& sorted_base, // size n
                                    int rounds,
                                    Fn fn)
{
    Result res; res.name = name;

    // Warm-up (not timed): copy base, then push_back(-1)
    {
        std::vector<int> v = sorted_base;
        v.push_back(-1);
        fn(v);
        consume(v);
    }

    std::vector<double> times; times.reserve(rounds);
    for (int r = 0; r < rounds; ++r) {
        std::vector<int> v = sorted_base;
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

    static std::vector<int> g_merge_buf;
    static std::vector<int> g_tims_buf;

    for (const auto& C : cases) {
        const size_t n = C.n;
        const int rounds = C.rounds;

        // base sorted array of size n: 1..n
        std::vector<int> sorted_base(n);
        std::iota(sorted_base.begin(), sorted_base.end(), 1);

        std::cout << "\n=== Post-insert case (sorted + push_back(-1)), n=" << n
                  << ", rounds=" << rounds << " ===\n\n";

        auto r_ladder = bench_algo_postinsert("LadderSort", sorted_base, rounds, [](std::vector<int>& v){
            ladder_sort_into(v, g_ladder_out_ws); // write into reusable out buffer
            v.swap(g_ladder_out_ws);              // deliver result in-place
        });

        auto r_tims = bench_algo_postinsert("Timsort", sorted_base, rounds, [&](std::vector<int>& v){
            timsort::sort_with_buf(v, g_tims_buf);        // stable; reusable buffer
        });

        auto r_quick = bench_algo_postinsert("Quicksort", sorted_base, rounds, [](std::vector<int>& v){
            quicksort3(v);                                 // adaptive pivot; in-place
        });

        auto r_intro = bench_algo_postinsert("Introsort", sorted_base, rounds, [](std::vector<int>& v){
            std::sort(v.begin(), v.end());                 // in-place
        });

        auto r_stable = bench_algo_postinsert("StableSort", sorted_base, rounds, [](std::vector<int>& v){
            std::stable_sort(v.begin(), v.end());          // library allocs counted
        });

        auto r_merge = bench_algo_postinsert("MergeSort", sorted_base, rounds, [&](std::vector<int>& v){
            mergesort_with_buf(v, g_merge_buf);            // reusable buf
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
