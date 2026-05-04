#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

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
inline int hinted_lower_bound_lad(const std::vector<int>& tops, int x, int hint) {
    int n = (int)tops.size();
    if (n == 0) return 0;
    int i = std::clamp(hint, 0, n - 1);
    if ((i == 0 || tops[i - 1] > x) && tops[i] <= x) return i;
    if (i + 1 < n && tops[i] > x && tops[i + 1] <= x) return i + 1;
    if (i > 0 && tops[i - 1] <= x && (i == 1 || tops[i - 2] > x)) return i - 1;
    if (tops[i] > x) {
        int last = i, ofs = 1;
        while (i + ofs < n && tops[i + ofs] > x) { last = i + ofs; ofs = (ofs << 1) + 1; }
        int lo = last + 1, hi = std::min(i + ofs, n - 1);
        while (lo <= hi) {
            int mid = lo + ((hi - lo) >> 1);
            if (tops[mid] > x) lo = mid + 1; else hi = mid - 1;
        }
        return lo;
    } else {
        int last = i, ofs = 1;
        while (i - ofs >= 0 && tops[i - ofs] <= x) { last = i - ofs; ofs <<= 1; }
        int lo = std::max(0, i - ofs), hi = last;
        while (lo <= hi) {
            int mid = lo + ((hi - lo) >> 1);
            if (tops[mid] > x) lo = mid + 1; else hi = mid - 1;
        }
        return lo;
    }
}

//======================== Merge primitives (for LadderSort) ===================
static void merge_two_gallop(const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& out) {
    out.clear();
    out.reserve(A.size() + B.size());
    size_t i = 0, j = 0;
    int winA = 0, winB = 0, GALLOP = 8;

    auto gallop_right = [](const std::vector<int>& V, size_t lo, int key) {
        size_t n = V.size(), step = 1, hi = lo;
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
                size_t step = 1, n = B.size(), nj = j;
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

//=========================== LoserTree for k-way merge ========================
struct LoserTree {
    int k;
    std::vector<int> tree, key;
    std::vector<const int*> cur, end;
    std::vector<char> alive;

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
            if (cur[i] < end[i]) { key[i] = *cur[i]; alive[i] = 1; }
        }
        for (int i = 0; i < k; ++i) if (alive[i]) adjust(i);
    }

    inline bool less_eq(int a, int b) const {
        if (!alive[a]) return false;
        if (!alive[b]) return true;
        if (key[a] != key[b]) return key[a] < key[b];
        return a < b;
    }

    inline void adjust(int s) {
        int t = s;
        for (int parent = (s + k) >> 1; parent > 0; parent >>= 1) {
            int &los = tree[parent - 1];
            if (los < 0) los = t;
            else if (!less_eq(t, los)) std::swap(t, los);
        }
        tree[0] = t;
    }

    inline int pop_and_advance() {
        int s = tree[0], v = key[s];
        if (++cur[s] < end[s]) key[s] = *cur[s];
        else alive[s] = 0;
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

    auto& lad  = g_lad_ws; lad.clear(); lad.reserve(64);
    auto& tops = g_tops_ws; tops.clear(); tops.reserve(64);

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

    if (lad.size() == 1) out = lad[0];
    else if (lad.size() == 2) merge_two_gallop(lad[0], lad[1], out);
    else merge_k_loser_tree(lad, out);
}

//=========================== Dataset generator ================================
static std::vector<int> generate_dataset(size_t N, uint64_t seed) {
    const int K = 16;
    const double overlap = 0.5;
    const size_t U = std::max<size_t>(1, (size_t)(N * overlap));
    const int BURST_MAX = 32, STAY_PCT = 60, RIGHT_PCT = 20;

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
            int tries = 0; while (tries < K && !has_more(s)) { s = (s + 1) % K; ++tries; }
            if (tries == K) break;
        }
        int burst = 1 + (int)(rnd() % BURST_MAX);
        while (burst-- > 0 && out.size() < N && has_more(s)) {
            out.push_back(runs[s][cur[s]++]);
        }
        int toss = (int)(rnd() % 100);
        if (toss < STAY_PCT) {}
        else if (toss < STAY_PCT + RIGHT_PCT) s = (s + 1) % K;
        else s = (s + K - 1) % K;
    }
    return out;
}

//=========================== Benchmark harness ===============================
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
    { std::vector<int> v = base; v.push_back(-1); fn(v); consume(v); }

    std::vector<double> times; times.reserve(rounds);
    for (int r = 0; r < rounds; ++r) {
        std::vector<int> v = base;
        v.push_back(-1);
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
    res.avg_seconds = mean; res.std_seconds = std::sqrt(acc / times.size());
    return res;
}

static void print_result(const Result& r) {
    std::cout << std::left << std::setw(14) << r.name
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

    const vector<size_t> sizes = {1'000'000ULL, 10'000'000ULL};
    const int rounds = 5;

    for (auto n : sizes) {
        vector<int> base = generate_dataset(n, 0xC0FFEEULL);
        cout << "\n=== LadderSort benchmark, n=" << n << ", rounds=" << rounds << " ===\n\n";

        auto r_ladder = bench_algo_postinsert("LadderSort", base, rounds, [](std::vector<int>& v){
            ladder_sort_into(v, g_ladder_out_ws);
            v.swap(g_ladder_out_ws);
        });

        print_result(r_ladder);
    }

    if (g_sink64 == 0xdeadbeefULL) cerr << "";
    return 0;
}
