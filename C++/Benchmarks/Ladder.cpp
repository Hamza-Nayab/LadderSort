#include <bits/stdc++.h>
using namespace std;

// === Your exact helper functions ===
int binary_search_lad(const vector<vector<int>>& lad, int target) {
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
    bool operator>(const HeapItem& other) const { return value > other.value; }
};

vector<int> merge_ladders(vector<vector<int>>& lists) {
    vector<int> merged;
    priority_queue<HeapItem, vector<HeapItem>, greater<HeapItem>> heap;
    for (int i = 0; i < (int)lists.size(); ++i)
        if (!lists[i].empty())
            heap.push({lists[i][0], i, 0});
    while (!heap.empty()) {
        auto top = heap.top(); heap.pop();
        merged.push_back(top.value);
        if (top.elem_idx + 1 < (int)lists[top.list_idx].size())
            heap.push({lists[top.list_idx][top.elem_idx + 1], top.list_idx, top.elem_idx + 1});
    }
    return merged;
}

vector<int> ladder(const vector<int>& array) {
    if (array.empty()) return {};
    vector<vector<int>> lad = { {array[0]} };
    for (int i = 1; i < (int)array.size(); ++i) {
        int a = array[i];
        int idx = binary_search_lad(lad, a);
        if (idx == (int)lad.size()) lad.push_back({a});
        else lad[idx].push_back(a);
    }
    return merge_ladders(lad);
}

// === Benchmark: Post-insertion sorted ===
int main() {
    const int NUM_ELEMENTS = 10'000'000;
    mt19937 rng(45);

    // Base sorted array
    vector<int> arr(NUM_ELEMENTS);
    iota(arr.begin(), arr.end(), 0);

    // Append one element that breaks sorted order
    arr.push_back(-1);

    auto start = chrono::high_resolution_clock::now();
    vector<int> sorted = ladder(arr);
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed = end - start;

    cout << "Post-Insertion Sorted: " << elapsed.count() << " seconds\n";
    cout << "Correct: " << boolalpha << is_sorted(sorted.begin(), sorted.end()) << "\n";
}
