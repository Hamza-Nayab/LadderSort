#include <iostream>
#include <vector>
#include <queue>
#include <random>
#include <chrono>
#include <algorithm>

using namespace std;

// Binary search to find the pile index
int findPile(vector<vector<int>>& piles, int val) {
    int low = 0, high = piles.size();
    while (low < high) {
        int mid = (low + high) / 2;
        if (piles[mid].back() >= val)
            high = mid;
        else
            low = mid + 1;
    }
    return low;
}

// Min-heap item for merging
struct HeapItem {
    int value, pileIndex, elementIndex;
    bool operator>(const HeapItem& other) const {
        return value > other.value;
    }
};

vector<int> patience_sort(const vector<int>& input) {
    vector<vector<int>> piles;

    // Build piles
    for (int x : input) {
        int i = findPile(piles, x);
        if (i == piles.size())
            piles.push_back({x});
        else
            piles[i].push_back(x);
    }

    // Merge piles
    priority_queue<HeapItem, vector<HeapItem>, greater<HeapItem>> heap;
    for (int i = 0; i < piles.size(); ++i)
        heap.push({piles[i][0], i, 0});

    vector<int> result;
    while (!heap.empty()) {
        auto [val, pileIdx, elemIdx] = heap.top(); heap.pop();
        result.push_back(val);
        if (++elemIdx < piles[pileIdx].size())
            heap.push({piles[pileIdx][elemIdx], pileIdx, elemIdx});
    }

    return result;
}

int main() {
    mt19937 rng(42);
    uniform_int_distribution<int> dist(1, 10'000'000);

    vector<int> data(10'000'000);
    for (int i = 0; i < data.size(); ++i)
        data[i] = dist(rng);

    // Patience sort run
    auto start = chrono::high_resolution_clock::now();
    vector<int> sorted = patience_sort(data);
    auto end = chrono::high_resolution_clock::now();
    cout << "Patience Sort Time: "
         << chrono::duration<double>(end - start).count() << " seconds\n";

    // Add number and rerun
    sorted.push_back(500);
    start = chrono::high_resolution_clock::now();
    vector<int> re_sorted = patience_sort(sorted);
    end = chrono::high_resolution_clock::now();
    cout << "Post-insert Time: "
         << chrono::duration<double>(end - start).count() << " seconds\n";

    return 0;
}
