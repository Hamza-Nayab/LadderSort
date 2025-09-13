//1
// Fixed-stride round-robin of W sorted runs.
// Models: many tiny, interleaved sorted sources (ideal for LadderSort’s hinted index).
static std::vector<int> generate_dataset(size_t N, uint64_t /*seed*/) {
    const int W = 16;
    std::vector<int> out; out.reserve(N);
    std::vector<size_t> start(W), stop(W);
    size_t q = N / W, r = N % W, acc = 0;
    for (int k = 0; k < W; ++k) {
        size_t len = q + (k < (int)r ? 1 : 0);
        start[k] = acc + 1;
        stop[k]  = acc + len;
        acc     += len;
    }
    for (size_t i = 0; ; ++i) {
        bool any = false;
        for (int k = 0; k < W; ++k) {
            size_t idx = start[k] + i;
            if (idx <= stop[k]) { out.push_back((int)idx); any = true; }
        }
        if (!any) break;
    }
    return out;
}
//2
// Block-cyclic interleave: K sorted runs emitted in tiny blocks of size B.
// Models: bursty arrivals from a few sorted producers, preserving per-run order.
static std::vector<int> generate_dataset(size_t N, uint64_t /*seed*/) {
    const int    K = 8;
    const size_t B = 4;
    std::vector<int> out; out.reserve(N);
    std::vector<size_t> cur(K), stop(K);
    size_t q = N / K, r = N % K, acc = 0;
    for (int k = 0; k < K; ++k) {
        size_t len = q + (k < (int)r ? 1 : 0);
        cur[k]  = acc + 1;
        stop[k] = acc + len;
        acc    += len;
    }
    size_t produced = 0;
    int k = 0;
    while (produced < N) {
        if (cur[k] <= stop[k]) {
            size_t take = std::min(B, stop[k] - cur[k] + 1);
            for (size_t t = 0; t < take; ++t) out.push_back((int)cur[k]++);
            produced += take;
        }
        k = (k + 1) % K;
    }
    return out;
}


//3
// Band-limited permutation: max displacement ≤ W.
// Models: nearly-sorted data with bounded local jitter.
static std::vector<int> generate_dataset(size_t N, uint64_t seed) {
    const size_t W = 32;
    const size_t BLOCK = W + 1;
    std::vector<int> out(N);
    for (size_t i = 0; i < N; ++i) out[i] = (int)(i + 1);
    uint64_t x = seed ? seed : 0x9E3779B97F4A7C15ULL;
    auto rnd = [&]() -> uint64_t { x ^= x << 7;  x ^= x >> 9;  x *= 0x2545F4914F6CDD1DULL; return x; };
    size_t phase = (BLOCK > 1) ? (size_t)(rnd() % BLOCK) : 0;
    auto shuffle_block = [&](size_t b, size_t e) {
        for (size_t i = e; i > b + 1; ) {
            --i;
            size_t j = b + (size_t)(rnd() % (i - b + 1));
            std::swap(out[i], out[j]);
        }
    };
    if (phase && phase < N) shuffle_block(0, phase);
    for (size_t b = phase; b < N; b += BLOCK) {
        size_t e = std::min(N, b + BLOCK);
        shuffle_block(b, e);
    }
    return out;
}

//4
// Uniform random integers (duplicates allowed; no structure).
// Models: fully unstructured data; worst-case for “structure-exploiting” sorts.
static std::vector<int> generate_dataset(size_t N, uint64_t seed) {
    std::vector<int> out; out.reserve(N);
    uint64_t x = seed ? seed : 0x9E3779B97F4A7C15ULL;
    auto rnd = [&]() -> uint64_t { x ^= x << 7;  x ^= x >> 9;  x *= 0x2545F4914F6CDD1DULL; return x; };
    for (size_t i = 0; i < N; ++i) out.push_back((int)(rnd() & 0x7fffffffULL));
    return out;
}


//5
// Sorted ascending baseline.
static std::vector<int> generate_dataset(size_t N, uint64_t /*seed*/) {
    std::vector<int> out(N);
    for (size_t i = 0; i < N; ++i) out[i] = (int)i;
    return out;
}
//6
// Sorted descending baseline.
static std::vector<int> generate_dataset(size_t N, uint64_t /*seed*/) {
    std::vector<int> out(N);
    for (size_t i = 0; i < N; ++i) out[i] = (int)(N - 1 - i);
    return out;
}

//7
// Two-run riffle: coin-flip interleave of two sorted halves.
// Models: merging two time-sorted sources with random interleaving.
static std::vector<int> generate_dataset(size_t N, uint64_t seed) {
    size_t L = N / 2, R = N - L;
    std::vector<int> out; out.reserve(N);
    uint64_t x = seed ? seed : 0x9E3779B97F4A7C15ULL;
    auto u01 = [&]() { x ^= x << 7; x ^= x >> 9; x *= 0x2545F4914F6CDD1ULL;
                       return (double)(((x >> 11) & ((1ull<<53)-1)) * (1.0 / (1ull<<53))); };
    size_t i = 0, j = 0;
    while (i < L || j < R) {
        if (i == L) { out.push_back((int)(L + j)); ++j; }
        else if (j == R) { out.push_back((int)i); ++i; }
        else if (u01() < 0.5) { out.push_back((int)i++); }
        else { out.push_back((int)(L + j++)); }
    }
    return out;
}


//8
// Shuffled concatenation of long sorted blocks.
// Models: external sort runs / row-groups appended in random block order.
static std::vector<int> generate_dataset(size_t N, uint64_t seed) {
    const int K = 32;
    std::vector<int> out; out.reserve(N);
    std::vector<size_t> base(K), len(K);
    size_t q = N / K, r = N % K, acc = 0;
    for (int k = 0; k < K; ++k) {
        len[k]  = q + (k < (int)r ? 1 : 0);
        base[k] = acc + 1;
        acc    += len[k];
    }
    uint64_t x = seed ? seed : 0x9E3779B97F4A7C15ULL;
    std::vector<int> order(K); std::iota(order.begin(), order.end(), 0);
    for (int i = K - 1; i > 0; --i) {
        x ^= x << 7; x ^= x >> 9; x *= 0x2545F4914F6CDD1ULL;
        std::swap(order[i], order[(int)(x % (i + 1))]);
    }
    for (int idx : order)
        for (size_t t = 0; t < len[idx]; ++t)
            out.push_back((int)(base[idx] + t));
    return out;
}
//9
// High-duplication uniform over 0..M-1.
// Models: low cardinality keys; large equality plateaus (stability matters).
static std::vector<int> generate_dataset(size_t N, uint64_t seed) {
    const int M = 1000;
    std::vector<int> out; out.reserve(N);
    uint64_t x = seed ? seed : 0x9E3779B97F4A7C15ULL;
    auto rnd = [&]() -> uint64_t { x ^= x << 7; x ^= x >> 9; x *= 0x2545F4914F6CDD1ULL; return x; };
    for (size_t i = 0; i < N; ++i) out.push_back((int)(rnd() % M));
    return out;
}

//10
// Zipf-like high-dup over 0..M-1.
// Models: skewed popularity; many ties on head keys, long tail on rare keys.
static std::vector<int> generate_dataset(size_t N, uint64_t seed) {
    const int M = 1000;
    const double alpha = 1.2;
    std::vector<double> cdf(M);
    double H = 0.0; for (int k = 1; k <= M; ++k) H += 1.0 / std::pow((double)k, alpha);
    double acc = 0.0; for (int k = 1; k <= M; ++k) { acc += (1.0 / std::pow((double)k, alpha)) / H; cdf[k-1] = acc; }
    std::vector<int> out; out.reserve(N);
    uint64_t x = seed ? seed : 0x9E3779B97F4A7C15ULL;
    auto u01 = [&]() { x ^= x << 7; x ^= x >> 9; x *= 0x2545F4914F6CDD1ULL;
                       return (double)(((x >> 11) & ((1ull<<53)-1)) * (1.0 / (1ull<<53))); };
    for (size_t i = 0; i < N; ++i) {
        double u = u01();
        int k = (int)(std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
        out.push_back(k);
    }
    return out;
}


//11
// Nearly-sorted with sparse errors (~0.1% random swaps).
// Models: mostly sorted data with rare out-of-order items.
static std::vector<int> generate_dataset(size_t N, uint64_t seed) {
    std::vector<int> out(N);
    for (size_t i = 0; i < N; ++i) out[i] = (int)i;
    uint64_t x = seed ? seed : 0x9E3779B97F4A7C15ULL;
    auto rnd = [&]() -> uint64_t { x ^= x << 7; x ^= x >> 9; x *= 0x2545F4914F6CDD1ULL; return x; };
    size_t S = std::max<size_t>(1, N / 1000);
    for (size_t s = 0; s < S; ++s) {
        size_t i = (size_t)(rnd() % N), j = (size_t)(rnd() % N);
        if (i != j) std::swap(out[i], out[j]);
    }
    return out;
}

//12
// Kafka/Kinesis-style multi-partition fan-in (K disjoint increasing runs; sticky bursts).
// Models: per-partition ordered streams arriving in short bursts with neighbor hops.
static std::vector<int> generate_dataset(size_t N, uint64_t seed) {
    const int K = 8;
    const int BURST_MAX = 16;
    std::vector<int> out; out.reserve(N);
    std::vector<size_t> next(K), end(K);
    size_t q = N / K, r = N % K, acc = 0;
    for (int k = 0; k < K; ++k) {
        size_t len = q + (k < (int)r ? 1 : 0);
        next[k] = acc + 1;
        end[k]  = acc + len;
        acc    += len;
    }
    uint64_t x = seed ? seed : 0x9E3779B97F4A7C15ULL;
    auto rnd = [&]() -> uint64_t { x ^= x << 7;  x ^= x >> 9;  x *= 0x2545F4914F6CDD1DULL;  return x; };
    auto has_more = [&](int k) -> bool { return next[k] <= end[k]; };
    int p = (int)(rnd() % K);
    while (out.size() < N) {
        if (!has_more(p)) {
            int tries = 0;
            while (tries < K && !has_more(p)) { p = (p + 1) % K; ++tries; }
            if (tries == K) break;
        }
        int burst = 1 + (int)(rnd() % BURST_MAX);
        while (burst-- > 0 && out.size() < N && has_more(p)) {
            out.push_back((int)next[p]++);
        }
        uint64_t r8 = rnd() & 0xFF;
        if (r8 < 153) { /* stay */ }
        else if (r8 < 204) p = (p + 1) % K;
        else               p = (p + K - 1) % K;
    }
    return out;
}



//13
// Consolidated market data (multi-venue feeds): K time-sorted runs with heavy ties.
// Models: venue fan-in with coarse buckets, jittery bursts; stability on ties matters.

static std::vector<int> generate_dataset(size_t N, uint64_t seed) {
    const int K = 16;
    const int BURST_MAX = 16;
    const int STEP_MAX  = 2;
    const int P_SAME    = 192;
    std::vector<int> out; out.reserve(N);
    std::vector<size_t> need(K);
    size_t q = N / K, r = N % K;
    for (int k = 0; k < K; ++k) need[k] = q + (k < (int)r ? 1 : 0);
    std::vector<size_t> emitted(K, 0);
    std::vector<int>    ts(K, 0);
    uint64_t x = seed ? seed : 0x9E3779B97F4A7C15ULL;
    auto rnd = [&]() -> uint64_t { x ^= x << 7;  x ^= x >> 9;  x *= 0x2545F4914F6CDD1DULL; return x; };
    auto venue_has_more = [&](int k) -> bool { return emitted[k] < need[k]; };
    int v = (int)(rnd() % K);
    while (out.size() < N) {
        if (!venue_has_more(v)) {
            int tries = 0;
            while (tries < K && !venue_has_more(v)) { v = (v + 1) % K; ++tries; }
            if (tries == K) break;
        }
        int burst = 1 + (int)(rnd() % BURST_MAX);
        while (burst-- > 0 && out.size() < N && venue_has_more(v)) {
            if (emitted[v] > 0) {
                uint64_t r8 = rnd() & 0xFF;
                if (r8 >= (uint64_t)P_SAME) ts[v] += 1 + (int)(rnd() % STEP_MAX);
            }
            out.push_back(ts[v]);
            ++emitted[v];
        }
        uint64_t s8 = rnd() & 0xFF;
        if      (s8 < 153) { /* stay */ }
        else if (s8 < 204)  v = (v + 1) % K;
        else                v = (v + K - 1) % K;
    }
    return out;
}

//14
// Social feed assembly (per-user fan-in): 10–40 followees, bursty & tie-heavy.
// Models: small-K k-way merge with frequent equal timestamps and sticky bursts.
static std::vector<int> generate_dataset(size_t N, uint64_t seed) {
    const int K = 24;
    const int BURST_MAX = 24;
    const int STEP_MAX  = 2;
    const int P_SAME    = 192;
    std::vector<int> out; out.reserve(N);
    std::vector<size_t> need(K);
    size_t q = N / K, r = N % K;
    for (int k = 0; k < K; ++k) need[k] = q + (k < (int)r ? 1 : 0);
    std::vector<int>    t(K, 0);
    std::vector<size_t> em(K, 0);
    uint64_t x = seed ? seed : 0x9E3779B97F4A7C15ULL;
    auto rnd = [&]() -> uint64_t { x ^= x << 7; x ^= x >> 9; x *= 0x2545F4914F6CDD1DULL; return x; };
    auto has_more = [&](int k){ return em[k] < need[k]; };
    int f = (int)(rnd() % K);
    while (out.size() < N) {
        if (!has_more(f)) {
            int tries = 0;
            while (tries < K && !has_more(f)) { f = (f + 1) % K; ++tries; }
            if (tries == K) break;
        }
        int burst = 1 + (int)(rnd() % BURST_MAX);
        while (burst-- > 0 && out.size() < N && has_more(f)) {
            if (em[f] > 0) {
                if ((rnd() & 0xFF) >= (uint64_t)P_SAME) t[f] += 1 + (int)(rnd() % STEP_MAX);
            }
            out.push_back(t[f]);
            ++em[f];
        }
        uint64_t u = rnd() & 0xFF;
        if      (u < 153) { /* stay */ }
        else if (u < 204)  f = (f + 1) % K;
        else               f = (f + K - 1) % K;
    }
    return out;
}

//15
// IoT/telemetry fan-in: dozens of sensors with bounded skew.
// Models: per-sensor monotone timestamps, small jitter, small-K merge.
static std::vector<int> generate_dataset(size_t N, uint64_t seed) {
    const int K = 32;
    const int W = 3;
    const int MIN_PER_ROUND = std::max(1, K / 4);
    const int MAX_PER_ROUND = std::max(MIN_PER_ROUND, K / 2);
    std::vector<int> out; out.reserve(N);
    std::vector<size_t> need(K);
    size_t q = N / K, r = N % K;
    for (int k = 0; k < K; ++k) need[k] = q + (k < (int)r ? 1 : 0);
    std::vector<int>    last(K, -1);
    std::vector<size_t> em(K, 0);
    uint64_t x = seed ? seed : 0x9E3779B97F4A7C15ULL;
    auto rnd = [&]() -> uint64_t { x ^= x << 7; x ^= x >> 9; x *= 0x2545F4914F6CDD1DULL; return x; };
    size_t produced = 0;
    int base_time = 0;
    int start_sensor = (int)(rnd() % K);
    while (produced < N) {
        ++base_time;
        int m = MIN_PER_ROUND + (int)(rnd() % (MAX_PER_ROUND - MIN_PER_ROUND + 1));
        int s = start_sensor;
        for (int picked = 0; picked < K && m > 0 && produced < N; ++picked) {
            int k = (s + picked) % K;
            if (em[k] >= need[k]) continue;
            int jitter = (int)(rnd() % (W + 1));
            int inc    = (int)(rnd() & 1);
            int t_prop = base_time + jitter;
            int t_emit = std::max(last[k] + inc, t_prop);
            out.push_back(t_emit);
            last[k] = t_emit;
            ++em[k];
            ++produced;
            --m;
        }
        start_sensor = (start_sensor + 1 + (int)(rnd() % 3)) % K;
    }
    return out;
}

//16
// Search-engine partial index merges: few segments, overlapping docID ranges.
// Models: merging docID-sorted postings from hot segments; many equal keys across runs.
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



//cases todo 16,15,14,13,7،3,2,1,4,5,6
//cases preffered { 3, 4, 5, 6, 2, 7, 14, 15, 16 }


//cases ignore for now 12,11,10