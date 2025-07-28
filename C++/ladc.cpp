#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

// Binary search for ladder index
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

// MinHeap item structure
struct HeapItem {
    int value;
    int list_idx;
    int elem_idx;

    bool operator>(const HeapItem& other) const {
        return value > other.value;
    }
};

// Merge ladders using custom MinHeap
vector<int> merge_ladders(vector<vector<int>>& lists) {
    vector<int> merged;
    // Min-heap using greater comparator
    priority_queue<HeapItem, vector<HeapItem>, greater<HeapItem>> heap;

    for (int i = 0; i < lists.size(); ++i) {
        if (!lists[i].empty()) {
            heap.push({lists[i][0], i, 0});
        }
    }

    while (!heap.empty()) {
        HeapItem top = heap.top();
        heap.pop();

        merged.push_back(top.value);

        if (top.elem_idx + 1 < lists[top.list_idx].size()) {
            int next_val = lists[top.list_idx][top.elem_idx + 1];
            heap.push({next_val, top.list_idx, top.elem_idx + 1});
        }
    }

    return merged;
}

// Ladder function
vector<int> ladder(const vector<int>& array) {
    if (array.empty()) return {};

    vector<vector<int>> lad;
    lad.push_back({array[0]});

    for (int i = 1; i < array.size(); ++i) {
        int a = array[i];
        int idx = binary_search_lad(lad, a);
        if (idx == lad.size()) {
            lad.push_back({a});
        } else {
            lad[idx].push_back(a);
        }
    }

    return merge_ladders(lad);
}


int main() {
    mt19937 rng(5);  // Fixed seed
    uniform_int_distribution<int> dist(1, 10'000'000);

    double total_initial = 0.0;
    double total_post_insert = 0.0;

    for (int run = 1; run <= 10; ++run) {
        cout << "Run #" << run << ":\n";

        vector<int> original(1'000'000);
        for (int i = 0; i < original.size(); ++i)
            original[i] = dist(rng);

        // Initial sort
        auto start = chrono::high_resolution_clock::now();
        vector<int> a = ladder(original);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        cout << "  Ladder sort time: " << duration.count() << " seconds\n";
        total_initial += duration.count();

        if (run == 1) {
            vector<int> sorted_copy = original;
            sort(sorted_copy.begin(), sorted_copy.end());
            cout << "  Correct: " << boolalpha << (a == sorted_copy) << "\n";
        }

        // Insert element and sort again
        a.push_back(500);
        start = chrono::high_resolution_clock::now();
        vector<int> c = ladder(a);
        end = chrono::high_resolution_clock::now();
        duration = end - start;
        cout << "  Post-insert Ladder sort time: " << duration.count() << " seconds\n";
        total_post_insert += duration.count();
    }

    cout << "\n==== Summary for Ladder Sort ====\n";
    cout << "Total time (Initial): " << total_initial << " seconds\n";
    cout << "Average time (Initial): " << total_initial / 10.0 << " seconds\n";
    cout << "Total time (Post-insert): " << total_post_insert << " seconds\n";
    cout << "Average time (Post-insert): " << total_post_insert / 10.0 << " seconds\n";

    return 0;
}


