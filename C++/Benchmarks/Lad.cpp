// ladder_bench_nonunique_kway.cpp
#include <iostream>
#include <vector>
#include <queue>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>

using namespace std;

// === Config Flags ===
const bool PRINT_EACH_RUN = false;
const int NUM_ELEMENTS = 10'000'000;
const int NUM_RUNS = 10;
const int NUM_BLOCKS = 20; // change this to change block size

int binary_search_lad(const vector<int>& tails, int target) {
    int low = 0, high = (int)tails.size();
    while (low < high) {
        int mid = (low + high) / 2;
        if (tails[mid] > target)
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

struct HeapItem {
    int value, list_idx, elem_idx;
    bool operator>(const HeapItem& other) const {
        return value > other.value;
    }
};

// K-way merge using min-heap (works on vector<vector<int>> blocks)
vector<int> merge_ladders(vector<vector<int>>& lists) {
    vector<int> merged;
    merged.reserve( accumulate(lists.begin(), lists.end(), size_t(0),
                               [](size_t s, const vector<int>& v){ return s + v.size(); }) );

    priority_queue<HeapItem, vector<HeapItem>, greater<HeapItem>> heap;
    for (int i = 0; i < (int)lists.size(); ++i)
        if (!lists[i].empty())
            heap.push({lists[i][0], i, 0});
    while (!heap.empty()) {
        auto top = heap.top(); heap.pop();
        merged.push_back(top.value);
        int li = top.list_idx;
        int ei = top.elem_idx + 1;
        if (ei < (int)lists[li].size())
            heap.push({lists[li][ei], li, ei});
    }
    return merged;
}

vector<int> ladder(const vector<int>& array) {
    if (array.empty()) return {};

    vector<vector<int>> lad;
    vector<int> tails; // store last element of each ladder

    lad.push_back({array[0]});
    tails.push_back(array[0]);

    for (int i = 1; i < (int)array.size(); ++i) {
        int a = array[i];
        int idx = binary_search_lad(tails, a);
        if (idx == (int)lad.size()) {
            lad.push_back({a});
            tails.push_back(a);
        } else {
            lad[idx].push_back(a);
            tails[idx] = a; // update tail
        }
    }
    return merge_ladders(lad);
}

void benchmark_case(const string& case_name, const vector<int>& input) {
    double total = 0.0;
    cout << "\n==== Benchmark: " << case_name << " ====\n";

    for (int run = 1; run <= NUM_RUNS; ++run) {
        vector<int> data = input;
        auto start = chrono::high_resolution_clock::now();
        vector<int> sorted = ladder(data);
        auto end = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration<double>(end - start).count();
        total += elapsed;

        if (PRINT_EACH_RUN) cout << "Run #" << run << ": " << elapsed << " seconds\n";

        if (run == 1) {
            vector<int> expected = input;
            sort(expected.begin(), expected.end());
            cout << "  Correct: " << boolalpha << (sorted == expected) << "\n";
        }
    }
    cout << "Avg Time: " << (total / NUM_RUNS) << " seconds\n";
}

int main() {
    mt19937 rng(45);

    // Base generator: non-unique
    vector<int> base(NUM_ELEMENTS);
    uniform_int_distribution<int> dist(-5'000'000, 5'000'000);
    for (int &x : base) x = dist(rng);

    vector<vector<int>> datasets;
    vector<string> names;

    // === Case 1: Block Sorted (build blocks) ===
    {
        vector<vector<int>> blocks;
        blocks.reserve(NUM_BLOCKS);
        int block_size = NUM_ELEMENTS / NUM_BLOCKS;
        int used = 0;
        int next_val = -NUM_ELEMENTS / 2;

        for (int b = 0; b < NUM_BLOCKS; ++b) {
            int this_block_size = (b == NUM_BLOCKS - 1) ? (NUM_ELEMENTS - used) : block_size;
            vector<int> block;
            block.reserve(this_block_size);
            for (int i = 0; i < this_block_size; ++i) {
                block.push_back(next_val++);
            }
            // shuffle within the block so block is locally sorted after sort()
            sort(block.begin(), block.end()); // already increasing, but keep for clarity
            // slight internal shuffle to introduce within-block randomness, then re-sort:
            // (this is optional; keep blocks internally sorted as required)
            //shuffle(block.begin(), block.end(), rng);
            //sort(block.begin(), block.end());
            blocks.push_back(move(block));
            used += this_block_size;
        }

        // Create two datasets:
        // A) concatenated blocks (locally sorted, globally unsorted)
        vector<int> concat;
        concat.reserve(NUM_ELEMENTS);
        for (auto &blk : blocks) {
            concat.insert(concat.end(), blk.begin(), blk.end());
        }
        datasets.push_back(move(concat));
        names.push_back("Block Sorted - Concatenated (locally-sorted)");

        // B) k-way merged using min-heap (globally sorted)
        vector<int> kmerged = merge_ladders(blocks); // uses min-heap
        datasets.push_back(move(kmerged));
        names.push_back("Block Sorted - K-way Merged (min-heap)");
    }

    // === Case 2: Few Unique + Stable Groups ===
    {
        vector<int> arr;
        arr.reserve(NUM_ELEMENTS);
        for (int v = 1; v <= 10; ++v) {
            for (int count = 0; count < NUM_ELEMENTS / 10; ++count)
                arr.push_back(v);
        }
        for (int v = 0; v < 10; ++v) {
            shuffle(arr.begin() + v * (NUM_ELEMENTS / 10),
                    arr.begin() + (v + 1) * (NUM_ELEMENTS / 10), rng);
        }
        datasets.push_back(arr);
        names.push_back("Few Unique + Stable Groups");
    }

    // === Case 3: Random Input ===
    datasets.push_back(base);
    names.push_back("Random Input");

    // === Case 4: Sorted Input ===
    {
        auto arr = base;
        sort(arr.begin(), arr.end());
        datasets.push_back(arr);
        names.push_back("Sorted Input");
    }

    // === Case 5: Reverse Sorted ===
    {
        auto arr = datasets.back();
        reverse(arr.begin(), arr.end());
        datasets.push_back(arr);
        names.push_back("Reverse Sorted");
    }

    // === Case 6: Nearly Sorted (95%) ===
    {
        auto arr = datasets[names.size() - 2]; // sorted version
        int swap_count = NUM_ELEMENTS / 20;
        for (int i = 0; i < swap_count; ++i)
            swap(arr[rng() % NUM_ELEMENTS], arr[rng() % NUM_ELEMENTS]);
        datasets.push_back(arr);
        names.push_back("Nearly Sorted (95%)");
    }

    // === Case 7: Few Unique Elements at End ===
    {
        auto arr = datasets[names.size() - 3]; // sorted version
        uniform_int_distribution<int> few_dist(1, 10);
        for (int i = NUM_ELEMENTS - 1000; i < NUM_ELEMENTS; ++i)
            arr[i] = few_dist(rng);
        datasets.push_back(arr);
        names.push_back("Few Unique Elements at End");
    }

    // === Case 8: Post-Insertion Sorted ===
    {
        auto arr = datasets[names.size() - 4]; // sorted version
        arr.push_back(500);
        datasets.push_back(arr);
        names.push_back("Post-Insertion Sorted");
    }

    // Run all benchmarks
    for (size_t i = 0; i < datasets.size(); ++i)
        benchmark_case("[General] " + names[i], datasets[i]);

    return 0;
}
