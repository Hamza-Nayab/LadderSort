#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

// Binary search on last values using index
int binary_search_lad(const vector<vector<int>>& lad, const vector<int>& array, int value) {
    int low = 0, high = lad.size();
    while (low < high) {
        int mid = (low + high) / 2;
        if (array[lad[mid].back()] > value)
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

// Structure for heap item using index
struct HeapItem {
    int value;      // actual value (for comparison)
    int list_idx;   // index of sublist in ladder
    int elem_idx;   // index within sublist

    bool operator>(const HeapItem& other) const {
        return value > other.value;
    }
};

// Merge ladders of indexes into final sorted array
vector<int> merge_ladders(const vector<vector<int>>& ladders, const vector<int>& array) {
    vector<int> result;
    priority_queue<HeapItem, vector<HeapItem>, greater<HeapItem>> heap;

    for (int i = 0; i < ladders.size(); ++i) {
        if (!ladders[i].empty()) {
            int idx = ladders[i][0];
            heap.push({array[idx], i, 0});
        }
    }

    while (!heap.empty()) {
        HeapItem top = heap.top();
        heap.pop();

        int idx = ladders[top.list_idx][top.elem_idx];
        result.push_back(array[idx]);

        int next_elem_idx = top.elem_idx + 1;
        if (next_elem_idx < ladders[top.list_idx].size()) {
            int next_idx = ladders[top.list_idx][next_elem_idx];
            heap.push({array[next_idx], top.list_idx, next_elem_idx});
        }
    }

    return result;
}


vector<int> ladder(const vector<int>& array) {
    if (array.empty()) return {};

    vector<vector<int>> lad;
    lad.push_back({0});  

    for (int i = 1; i < array.size(); ++i) {
        int val = array[i];
        int idx = binary_search_lad(lad, array, val);
        if (idx == lad.size()) {
            lad.push_back({i});
        } else {
            lad[idx].push_back(i);
        }
    }

    return merge_ladders(lad, array);
}

int main() {
    // Generate random data
    mt19937 rng(5);  // Fixed seed for reproducibility
    uniform_int_distribution<int> dist(1, 1'000'000);
    vector<int> data(1'000'000);
    for (int i = 0; i < data.size(); ++i)
        data[i] = dist(rng);

    // First run
    auto start = chrono::high_resolution_clock::now();
    vector<int> sorted = ladder(data);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "Initial ladder sort time: " << duration.count() << " seconds\n";

    // Correctness check
    vector<int> expected = data;
    sort(expected.begin(), expected.end());
    cout << "Correct: " << boolalpha << (sorted == expected) << endl;

    // Add new element and rerun
    sorted.push_back(500);
    start = chrono::high_resolution_clock::now();
    vector<int> sorted2 = ladder(sorted);
    end = chrono::high_resolution_clock::now();
    duration = end - start;
    cout << "Post-insert sort time: " << duration.count() << " seconds\n";

    return 0;
}
