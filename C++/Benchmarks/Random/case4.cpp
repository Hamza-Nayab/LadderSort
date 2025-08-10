#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>

using namespace std;

inline int binary_search_lad(const vector<vector<int>>& lad, int target) {
    int low = 0, high = lad.size();
    while (low < high) {
        int mid = (low + high) / 2;
        if (lad[mid].back() > target)
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

vector<int> merge_ladders(const vector<vector<int>>& lists) {
    vector<int> merged;
    merged.reserve(10'000'001);

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

vector<int> ladder(const vector<int>& array) {
    if (array.empty()) return {};

    vector<vector<int>> lad;
    lad.reserve(64);
    lad.push_back({array[0]});

    for (int i = 1, n = (int)array.size(); i < n; ++i) {
        int a = array[i];
        int idx = binary_search_lad(lad, a);
        if (idx == (int)lad.size()) {
            lad.emplace_back().emplace_back(a);
        } else {
            lad[idx].emplace_back(a);
        }
    }

    return merge_ladders(lad);
}

int main() {
    constexpr int NUM_ELEMENTS = 10'000'000;
    mt19937 rng(42); // fixed seed
    double total_time = 0.0;

    // Generate "Few Unique + Stable Groups" dataset
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

    for (int run = 1; run <= 10; ++run) {
        vector<int> test_arr = arr;
        test_arr.push_back(-1); // force reordering

        auto start = chrono::steady_clock::now();
        vector<int> result = ladder(test_arr);
        auto end = chrono::steady_clock::now();

        total_time += chrono::duration<double>(end - start).count();
    }

    cout << "Average time (Few Unique + Stable Groups + Post-insert): "
         << fixed << total_time / 10.0 << " seconds\n";
}
