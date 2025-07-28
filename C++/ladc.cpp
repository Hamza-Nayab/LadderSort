#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

// Ladder index binary search
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

// Heap structure
struct HeapItem {
    int value, list_idx, elem_idx;
    bool operator>(const HeapItem& other) const {
        return value > other.value;
    }
};

// Merging ladders
vector<int> merge_ladders(const vector<vector<int>>& lists) {
    vector<int> merged;
    merged.reserve(10'000'000); // expected output size

    priority_queue<HeapItem, vector<HeapItem>, greater<HeapItem>> heap;
    for (int i = 0; i < lists.size(); ++i) {
        if (!lists[i].empty())
            heap.push({lists[i][0], i, 0});
    }

    while (!heap.empty()) {
        auto top = heap.top(); heap.pop();
        merged.push_back(top.value);

        const auto& curr = lists[top.list_idx];
        int next_idx = top.elem_idx + 1;
        if (next_idx < curr.size())
            heap.push({curr[next_idx], top.list_idx, next_idx});
    }

    return merged;
}

// Ladder Sort
vector<int> ladder(const vector<int>& array) {
    if (array.empty()) return {};

    vector<vector<int>> lad;
    lad.reserve(64); // reserve for fewer reallocs
    lad.push_back({array[0]});

    for (int i = 1, n = array.size(); i < n; ++i) {
        int a = array[i];
        int idx = binary_search_lad(lad, a);
        if (idx == lad.size()) {
            lad.emplace_back().emplace_back(a);
        } else {
            lad[idx].emplace_back(a);
        }
    }

    return merge_ladders(lad);
}

int main() {
    constexpr int n = 10'000'000;
    mt19937 rng(5);
    uniform_int_distribution<int> dist(1, 100'000'000);

    double total_initial = 0, total_post_insert = 0;

    for (int run = 1; run <= 10; ++run) {
        cout << "Run #" << run << ":\n";

        vector<int> b(n);
        for (int i = 0; i < n; ++i)
            b[i] = dist(rng);

        auto start = chrono::steady_clock::now();
        vector<int> a = ladder(b);
        auto end = chrono::steady_clock::now();
        double duration = chrono::duration<double>(end - start).count();
        cout << "  Initial time: " << duration << " seconds\n";
        total_initial += duration;

        if (run == 1) {
            vector<int> sorted_b = b;
            sort(sorted_b.begin(), sorted_b.end());
            cout << "  Correct: " << boolalpha << (a == sorted_b) << "\n";
        }

        a.push_back(500);

        start = chrono::steady_clock::now();
        vector<int> c = ladder(a);
        end = chrono::steady_clock::now();
        duration = chrono::duration<double>(end - start).count();
        cout << "  Post-insert time: " << duration << " seconds\n";
        total_post_insert += duration;
    }

    cout << "\n==== Summary for Ladder Sort ====\n";
    cout << "Total time (Initial): " << total_initial << " seconds\n";
    cout << "Average time (Initial): " << total_initial / 10.0 << " seconds\n";
    cout << "Total time (Post-insert): " << total_post_insert << " seconds\n";
    cout << "Average time (Post-insert): " << total_post_insert / 10.0 << " seconds\n";

    return 0;
}
