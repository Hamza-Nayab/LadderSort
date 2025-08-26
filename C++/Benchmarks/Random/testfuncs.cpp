
static std::vector<int> generate_dataset(size_t N, uint64_t /*seed*/) {
    const int W = 16; // tweak: 8..32 are good values for LadderSort
    std::vector<int> out; out.reserve(N);

    // Partition 1..N into W disjoint increasing runs.
    std::vector<size_t> start(W), stop(W); // stop is inclusive
    size_t q = N / W, r = N % W, acc = 0;
    for (int k = 0; k < W; ++k) {
        size_t len = q + (k < (int)r ? 1 : 0);
        start[k] = acc + 1;
        stop[k]  = acc + len;  // inclusive
        acc     += len;
    }

    // Round-robin by single elements: R0[0], R1[0], ..., RW-1[0], then R0[1], ...
    for (size_t i = 0; ; ++i) {
        bool any = false;
        for (int k = 0; k < W; ++k) {
            size_t idx = start[k] + i;
            if (idx <= stop[k]) { out.push_back((int)idx); any = true; }
        }
        if (!any) break; // all runs exhausted
    }
    return out;
}


static std::vector<int> generate_dataset(size_t N, uint64_t /*seed*/) {
    const int    K = 8;   // small number of runs (e.g., >8)
    const size_t B = 4;    // tiny block size (e.g., >4)

    std::vector<int> out; out.reserve(N);

    // Partition 1..N into K disjoint increasing runs (nearly equal lengths).
    std::vector<size_t> cur(K), stop(K); // stop is inclusive
    size_t q = N / K, r = N % K, acc = 0;
    for (int k = 0; k < K; ++k) {
        size_t len = q + (k < (int)r ? 1 : 0);
        cur[k]  = acc + 1;
        stop[k] = acc + len;   // inclusive
        acc    += len;
    }

    // Emit blocks of size B from each run in cyclic order.
    size_t produced = 0;
    int k = 0;
    while (produced < N) {
        if (cur[k] <= stop[k]) {
            size_t take = std::min(B, stop[k] - cur[k] + 1);
            for (size_t t = 0; t < take; ++t) out.push_back((int)cur[k]++);
            produced += take;
        }
        k = (k + 1) % K; // next run
    }
    return out;
}

static std::vector<int> generate_dataset(size_t N, uint64_t seed) {
    // Max displacement W (tune as you like: e.g., 16, 32, 64)
    const size_t W = 32;
    const size_t BLOCK = W + 1;

    // Start with 1..N
    std::vector<int> out(N);
    for (size_t i = 0; i < N; ++i) out[i] = (int)(i + 1);

    // Tiny xorshift64* PRNG (no <random> dependency)
    uint64_t x = seed ? seed : 0x9E3779B97F4A7C15ULL;
    auto rnd = [&]() -> uint64_t {
        x ^= x << 7;  x ^= x >> 9;  x *= 0x2545F4914F6CDD1DULL;
        return x;
    };

    // Optional: random phase so block edges don't always align at 0
    size_t phase = (BLOCK > 1) ? (size_t)(rnd() % BLOCK) : 0;

    // Shuffle each block independently; block size ≤ W+1 ⇒ displacement ≤ W
    auto shuffle_block = [&](size_t b, size_t e) {
        // In-place Fisher–Yates on [b, e)
        for (size_t i = e; i > b + 1; ) {
            --i;
            size_t j = b + (size_t)(rnd() % (i - b + 1));
            std::swap(out[i], out[j]);
        }
    };

    // First (possibly short) block to apply the random phase
    if (phase && phase < N) shuffle_block(0, phase);

    // Full blocks of size BLOCK
    for (size_t b = phase; b < N; b += BLOCK) {
        size_t e = std::min(N, b + BLOCK);
        shuffle_block(b, e);
    }

    return out;
}

static std::vector<int> generate_dataset(size_t N, uint64_t /*seed*/) {
    // K-way round-robin interleaving of strictly increasing runs (K = 64)
    const int K = 32;
    std::vector<int> out;
    out.reserve(N);

    const size_t chunk = N / K;                 // length of the first K-1 runs
    const size_t last_size = N - chunk * (K-1); // length of the last run (>= chunk)

    auto run_size = [&](int k) -> size_t {
        return (k == K - 1) ? last_size : chunk;
    };
    auto run_base = [&](int k) -> size_t {
        // values are 1..N spread across runs; run k starts at 1 + k*chunk
        return 1 + static_cast<size_t>(k) * chunk;
    };

    // Interleave by rows: S[0][0], S[1][0], ..., S[K-1][0], then S[0][1], ...
    // Note: exactly mirrors your earlier loop: for (i < chunk + 1) ...
    for (size_t i = 0; i < chunk + 1; ++i) {
        for (int k = 0; k < K; ++k) {
            if (i < run_size(k)) {
                out.push_back(static_cast<int>(run_base(k) + i));
            }
        }
    }

    // If N is a multiple of 64 (your benchmarks: 1e6, 1e7, 1e8), out.size() == N.
    // For other N, this matches your exact construction (which may leave out some tail
    // of the last run if N % 64 > 1).

    return out;
}


static std::vector<int> generate_dataset(size_t N, uint64_t seed) {
    std::vector<int> out; out.reserve(N);

    // Tiny xorshift64* PRNG (no <random> dependency)
    uint64_t x = seed ? seed : 0x9E3779B97F4A7C15ULL;
    auto rnd = [&]() -> uint64_t {
        x ^= x << 7;  x ^= x >> 9;  x *= 0x2545F4914F6CDD1DULL;
        return x;
    };

    for (size_t i = 0; i < N; ++i) {
        // 0 .. 2^31-1 (non-negative so -1 is still smallest when you append it)
        out.push_back((int)(rnd() & 0x7fffffffULL));
    }
    return out;
}



static std::vector<int> generate_dataset(size_t N, uint64_t /*seed*/) {
    std::vector<int> out(N);
    for (size_t i = 0; i < N; ++i) out[i] = static_cast<int>(i);  // 0,1,2,...,N-1
    return out;
}


static std::vector<int> generate_dataset(size_t N, uint64_t /*seed*/) {
    std::vector<int> out(N);
    for (size_t i = 0; i < N; ++i) {
        out[i] = static_cast<int>(N - 1 - i);   // N-1, N-2, ..., 0
    }
    return out;
}


// CASE: Two-run riffle (coin-flip interleave of two sorted halves)
static std::vector<int> generate_dataset(size_t N, uint64_t seed) {
    size_t L = N / 2, R = N - L;
    std::vector<int> out; out.reserve(N);
    uint64_t x = seed ? seed : 0x9E3779B97F4A7C15ULL;
    auto u01 = [&]() { return (double)((((x ^= x << 7), (x ^= x >> 9), x *= 0x2545F4914F6CDD1ULL) >> 11) * (1.0 / (1ull << 53))); };

    size_t i = 0, j = 0;
    while (i < L || j < R) {
        if (i == L) { out.push_back((int)(L + j)); ++j; }
        else if (j == R) { out.push_back((int)i); ++i; }
        else if (u01() < 0.5) { out.push_back((int)i++); }
        else { out.push_back((int)(L + j++)); }
    }
    // shift up to be >= 0 (already >=0); optional +1 if you prefer 1..N
    return out;
}


// CASE: Shuffled concatenation of long sorted blocks
static std::vector<int> generate_dataset(size_t N, uint64_t seed) {
    const int K = 32; // #blocks
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

// CASE: High-duplication (uniform over 0..M-1)
static std::vector<int> generate_dataset(size_t N, uint64_t seed) {
    const int M = 1000; // cardinality
    std::vector<int> out; out.reserve(N);
    uint64_t x = seed ? seed : 0x9E3779B97F4A7C15ULL;
    auto rnd = [&]() -> uint64_t { x ^= x << 7; x ^= x >> 9; x *= 0x2545F4914F6CDD1ULL; return x; };
    for (size_t i = 0; i < N; ++i) out.push_back((int)(rnd() % M));
    return out;
}
