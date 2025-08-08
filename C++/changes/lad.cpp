#include <iostream>
#include <vector>
#include <queue>
#include <random>
#include <chrono>
#include <algorithm>

using namespace std;

// ===================== CONFIG =====================
const bool PRINT_DEBUG = false;
const int NUM_ELEMENTS = 10'000'000;
const int NUM_RUNS = 5;

// ===================== BINARY SEARCH (using ladder tails) =====================
inline int binary_search_lad(const vector<int>& tails, int target) {
    int low = 0, high = tails.size();
    while (low < high) {
        int mid = (low + high) / 2;
        if (tails[mid] < target) high = mid;
        else low = mid + 1;
    }
    return low;
}

// ===================== LADDER SORT =====================
vector<int> ladder_sort(const vector<int>& arr) {
    if (arr.empty()) return {};

    vector<vector<int>> ladders;
    vector<int> tails; // last elements of each ladder

    ladders.reserve(arr.size() / 4);
    tails.reserve(arr.size() / 4);

    // First element initializes first ladder
    ladders.push_back({arr[0]});
    tails.push_back(arr[0]);

    // Phase 1: Building ladders
    for (size_t i = 1; i < arr.size(); i++) {
        int pos = binary_search_lad(tails, arr[i]);

        if (pos == (int)ladders.size()) {
            ladders.push_back({arr[i]});
            tails.push_back(arr[i]);
        } else {
            ladders[pos].push_back(arr[i]);
            tails[pos] = arr[i];
        }
    }

    // Phase 2: Merge ladders
    vector<int> sorted;
    sorted.reserve(arr.size());

    if (ladders.size() < 8) {
        // Simple merge for fewer subsequences
        for (auto& seq : ladders)
            sorted.insert(sorted.end(), seq.begin(), seq.end());
        sort(sorted.begin(), sorted.end());
    } else {
        // k-way merge using min-heap
        struct Node { int val, ladIdx, pos; };
        auto cmp = [](const Node& a, const Node& b) { return a.val > b.val; };
        priority_queue<Node, vector<Node>, decltype(cmp)> minHeap(cmp);

        for (int i = 0; i < ladders.size(); i++)
            if (!ladders[i].empty())
                minHeap.push({ladders[i][0], i, 0});

        while (!minHeap.empty()) {
            auto cur = minHeap.top();
            minHeap.pop();
            sorted.push_back(cur.val);

            if (cur.pos + 1 < (int)ladders[cur.ladIdx].size()) {
                minHeap.push({ladders[cur.ladIdx][cur.pos + 1], cur.ladIdx, cur.pos + 1});
            }
        }
    }

    return sorted;
}

// ===================== MAIN TEST =====================
int main() {
    vector<int> arr(NUM_ELEMENTS);
    mt19937 rng(42);
    uniform_int_distribution<int> dist(1, 10'000'000);

    for (auto& x : arr) x = dist(rng);

    // Ladder sort benchmark
    auto start = chrono::high_resolution_clock::now();
    auto sorted = ladder_sort(arr);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "Ladder sort time: " << elapsed.count() << "s\n";

    // Append + std::sort benchmark
    sorted.push_back(500000); // Example new element
    start = chrono::high_resolution_clock::now();
    ladder_sort(sorted);
    end = chrono::high_resolution_clock::now();
    elapsed = end - start;
    cout << "Append + std::sort time: " << elapsed.count() << "s\n";

}
