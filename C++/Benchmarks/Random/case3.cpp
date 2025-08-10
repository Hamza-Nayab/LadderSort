#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <chrono>
#include <random>

using namespace std;

constexpr int NUM_ELEMENTS = 10'000'000;
constexpr int NUM_BLOCKS = 100;

// Heap structure
struct HeapItem {
    int value, list_idx, elem_idx;
    bool operator>(const HeapItem& other) const {
        return value > other.value;
    }
};

// Merge sorted blocks using min-heap
vector<int> merge_ladders(const vector<vector<int>>& lists) {
    vector<int> merged;
    merged.reserve(NUM_ELEMENTS);

    priority_queue<HeapItem, vector<HeapItem>, greater<HeapItem>> heap;
    for (int i = 0; i < (int)lists.size(); ++i) {
        if (!lists[i].empty())
            heap.push({lists[i][0], i, 0});
    }

    while (!heap.empty()) {
        auto top = heap.top(); heap.pop();
        merged.push_back(top.value);
        const auto& curr = lists[top.list_idx];
        int next_idx = top.elem_idx + 1;
        if (next_idx < (int)curr.size())
            heap.push({curr[next_idx], top.list_idx, next_idx});
    }

    return merged;
}

// Ladder Sort
vector<int> ladder(const vector<int>& array) {
    if (array.empty()) return {};

    vector<vector<int>> lad;
    lad.reserve(64);
    lad.push_back({array[0]});

    for (int i = 1, n = (int)array.size(); i < n; ++i) {
        int a = array[i];
        int low = 0, high = (int)lad.size();
        while (low < high) {
            int mid = (low + high) / 2;
            if (lad[mid].back() > a)
                low = mid + 1;
            else
                high = mid;
        }
        if (low == (int)lad.size())
            lad.emplace_back().emplace_back(a);
        else
            lad[low].emplace_back(a);
    }

    return merge_ladders(lad);
}

int main() {
    mt19937 rng(45);

    // === Build Block Sorted - Concatenated (locally-sorted) ===
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
        sort(block.begin(), block.end()); // keep block sorted
        blocks.push_back(move(block));
        used += this_block_size;
    }

    // Concatenate blocks into one array
    vector<int> concat;
    concat.reserve(NUM_ELEMENTS);
    for (auto &blk : blocks) {
        concat.insert(concat.end(), blk.begin(), blk.end());
    }

    // === Benchmark Ladder Sort on this case ===
    double total_time = 0.0;
    for (int run = 1; run <= 10; ++run) {
        vector<int> test_arr = concat;

        auto start = chrono::steady_clock::now();
        vector<int> result = ladder(test_arr);
        auto end = chrono::steady_clock::now();

        total_time += chrono::duration<double>(end - start).count();
    }

    cout << "Average time (Block Sorted - Concatenated): "
         << fixed << total_time / 10.0 << " seconds\n";

    return 0;
}
