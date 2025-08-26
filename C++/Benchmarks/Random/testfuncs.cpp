// Fixed-stride (pure round-robin) interleave with small W.
// Choose W (e.g., 8, 16, or 32). The seed is unused (kept to match your signature).
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
