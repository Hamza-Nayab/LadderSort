// g++ -O3 -std=c++17 -march=native bench_blocks_ladder.cpp -o bench_blocks_ladder
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

using std::size_t;

// ------------------------------- Config --------------------------------------
static constexpr size_t NUM_ELEMENTS   = 10'000'000; // total elements
static constexpr int    NUM_BLOCKS     = 64;         // number of locally-sorted blocks
static constexpr bool   SHUFFLE_BLOCKS = true;       // shuffle block order to make it non-trivial
static constexpr int    ROUNDS         = 10;         // timed rounds

//--------------------------- DCE sink (avoid optimizer) -----------------------
static volatile uint64_t g_sink64 = 0;
template<typename Vec>
inline void consume(const Vec& v) {
    uint64_t s = 0; for (auto x : v) s += (uint64_t)x;
    g_sink64 ^= s;
}

//======================== LadderSort workspaces (reused) ======================
static std::vector<std::vector<int>> g_lad_ws;        // runs
static std::vector<int>              g_tops_ws;       // run tails (flat)
static std::vector<int>              g_ladder_out_ws; // output buffer

//------------------------ Hinted search over flat `tops` ----------------------
inline int hinted_lower_bound_lad(const std::vector<int>& tops, int x, int last_idx_hint) {
    const int n = (int)tops.size();
    if (n == 0) return 0;
    int idx = last_idx_hint;
    if (idx < 0) idx = 0;
    if (idx >= n) idx = n - 1;

    // local walk (tops maintained non-increasing)
    if (tops[idx] > x) {
        while (idx + 1 < n && tops[idx + 1] > x) ++idx;
    } else {
        while (idx > 0 && tops[idx - 1] <= x) --idx;
    }
    // validate neighborhood; else binary search
    if (!((idx == 0 || tops[idx - 1] > x) && tops[idx] <= x)) {
        int lo = 0, hi = n;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (tops[mid] > x) lo = mid + 1; else hi = mid;
        }
        idx = lo;
    }
    return idx;
}

//======================== Merge primitives (for LadderSort) ===================
// 2-way merge with simple galloping (stable)
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
                // take as many from B strictly less than A[i]
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

//==================== Build “Block-Sorted Concatenated” input =================
static std::vector<int> build_block_sorted_concat(size_t N, int blocks, uint64_t seed, bool shuffle_blocks) {
    std::mt19937_64 rng(seed);
    std::vector<std::vector<int>> blks; blks.reserve(blocks);

    size_t used = 0;
    size_t base_block = N / (size_t)blocks;

    // Generate each block as a sorted sequence. We’ll use consecutive integers
    // to avoid extra RNG cost and keep blocks perfectly sorted internally.
    // To make the global array *not* sorted (interesting case), we shuffle the
    // order of blocks before concatenation.
    int next_val = -(int)N / 2;
    for (int b = 0; b < blocks; ++b) {
        size_t need = (b == blocks - 1) ? (N - used) : base_block;
        std::vector<int> blk; blk.reserve(need);
        for (size_t i = 0; i < need; ++i) blk.push_back(next_val++);
        // already nondecreasing; keep it sorted
        blks.push_back(std::move(blk));
        used += need;
    }
    if (shuffle_blocks) std::shuffle(blks.begin(), blks.end(), rng);

    std::vector<int> out; out.reserve(N);
    for (auto& blk : blks) out.insert(out.end(), blk.begin(), blk.end());
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
static Result bench_algo_blocks(const std::string& name,
                                size_t N,
                                int blocks,
                                int rounds,
                                uint64_t seed,
                                Fn fn)
{
    Result res; res.name = name;

    // Warm-up (not timed)
    {
        auto base = build_block_sorted_concat(N, blocks, seed, SHUFFLE_BLOCKS);
        std::vector<int> v = base;
        fn(v);
        consume(v);
    }

    std::vector<double> times; times.reserve(rounds);
    for (int r = 0; r < rounds; ++r) {
        auto base = build_block_sorted_concat(N, blocks, seed + 0x9E3779B97F4A7C15ULL * r, SHUFFLE_BLOCKS);
        std::vector<int> v = base;
        auto t0 = std::chrono::steady_clock::now();
        fn(v); // sorting happens here
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
    std::cout << std::left << std::setw(24) << r.name
              << " avg: " << std::fixed << std::setprecision(6) << r.avg_seconds
              << " s  (±" << std::setprecision(6) << r.std_seconds << ")"
              << (r.ok ? "" : "  (! not sorted)") << "\n";
}

//----------------------------------- Main -------------------------------------
int main() {
    using namespace std;
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << std::unitbuf;

    cout << "=== Block-sorted concatenated: N=" << NUM_ELEMENTS
         << ", blocks=" << NUM_BLOCKS
         << ", rounds=" << ROUNDS
         << ", shuffle=" << (SHUFFLE_BLOCKS ? "yes" : "no") << " ===\n\n";

    auto r_ladder = bench_algo_blocks("LadderSort (hinted + loser-tree)",
                                      NUM_ELEMENTS, NUM_BLOCKS, ROUNDS,
                                      0xBADC0FFEEULL,
                                      [](std::vector<int>& v){
                                          ladder_sort_into(v, g_ladder_out_ws);
                                          v.swap(g_ladder_out_ws);
                                      });

    cout << "Results:\n";
    print_result(r_ladder);

    if (g_sink64 == 0xdeadbeefULL) std::cerr << "";
    return 0;
}
