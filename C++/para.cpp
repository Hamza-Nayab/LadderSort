#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <omp.h>
#include <mutex>

using namespace std;

int binary_search_lad(const vector<vector<int>>& lad, int target) {
    int low = 0, high = lad.size();
    while (low < high) {
        int mid = (low + high) / 2;
        if (lad[mid].back() >= target)
            high = mid;
        else
            low = mid + 1;
    }
    return low;
}

int main() {
    int n = 100000;
    vector<int> data(n);

    mt19937 rng(5);  // Fixed seed for reproducibility
    uniform_int_distribution<int> dist(1, 1000000);
    for (int i = 0; i < n; ++i)
        data[i] = dist(rng);

    vector<vector<int>> lad;
    vector<mutex> lad_locks;
    int num_threads = omp_get_max_threads();
    cout << "Threads used: " << num_threads << endl;

    auto start = omp_get_wtime();

    #pragma omp parallel
    {
        vector<vector<int>> local_lad;
        vector<mutex> local_locks;
        #pragma omp for nowait
        for (int i = 0; i < n; ++i) {
            int val = data[i];
            int pos = binary_search_lad(local_lad, val);
            if (pos == local_lad.size()) {
                local_lad.emplace_back();
                local_locks.emplace_back();
            }
            local_lad[pos].push_back(val);
        }

        #pragma omp critical
        {
            for (int i = 0; i < local_lad.size(); ++i) {
                if (lad.size() <= i) {
                    lad.emplace_back();
                    lad_locks.emplace_back();
                }
                lock_guard<mutex> lock(lad_locks[i]);
                lad[i].insert(lad[i].end(), local_lad[i].begin(), local_lad[i].end());
            }
        }
    }

    vector<int> sorted;
    for (auto& bucket : lad) {
        sort(bucket.begin(), bucket.end());
        sorted.insert(sorted.end(), bucket.begin(), bucket.end());
    }

    auto end = omp_get_wtime();
    cout << "Total time taken: " << (end - start) << " seconds\n";

    return 0;
}
