#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <queue>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>

using namespace std;

// ===== Ladder Sort with K-way merge =====
void ladderSort(vector<int>& arr) {
    if (arr.empty()) return;

    vector<vector<int>> ladders;
    ladders.push_back({arr[0]});

    // Build ladders
    for (size_t i = 1; i < arr.size(); ++i) {
        int x = arr[i];
        int lo = 0, hi = ladders.size();
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (ladders[mid].back() > x) lo = mid + 1;
            else hi = mid;
        }
        if (lo == (int)ladders.size()) ladders.push_back({});
        ladders[lo].push_back(x);
    }

    // K-way merge using min-heap
    struct Node {
        int value;
        size_t lad_idx;
        size_t elem_idx;
        bool operator>(const Node& other) const { return value > other.value; }
    };

    priority_queue<Node, vector<Node>, greater<Node>> pq;

    for (size_t i = 0; i < ladders.size(); ++i) {
        if (!ladders[i].empty()) {
            pq.push({ladders[i][0], i, 0});
        }
    }

    vector<int> result;
    result.reserve(arr.size());

    while (!pq.empty()) {
        auto [val, lad_idx, elem_idx] = pq.top();
        pq.pop();
        result.push_back(val);

        if (elem_idx + 1 < ladders[lad_idx].size()) {
            pq.push({ladders[lad_idx][elem_idx + 1], lad_idx, elem_idx + 1});
        }
    }

    arr = move(result);
}

// ===== Benchmark =====
void benchmark_case_post_insertion_sorted() {
    size_t N = 10'000'000;

    // Prepare sorted data
    vector<int> arr(N);
    iota(arr.begin(), arr.end(), 0);

    // Append one element at the end
    arr.push_back(-1);

    // Time measurement
    auto start = chrono::high_resolution_clock::now();
    ladderSort(arr);
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed = end - start;

    cout << left << setw(25) << "Post-Insertion Sorted"
         << setw(12) << fixed << setprecision(6) << elapsed.count()
         << (is_sorted(arr.begin(), arr.end()) ? "  OK" : "  FAIL")
         << "\n";
}

int main() {
    cout << left << setw(25) << "Dataset"
         << setw(12) << "Time(s)"
         << "Result\n";
    cout << string(45, '-') << "\n";

    benchmark_case_post_insertion_sorted();

    return 0;
}
