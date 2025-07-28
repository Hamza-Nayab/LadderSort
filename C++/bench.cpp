#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <random>
#include <chrono>
#include <numeric>
using namespace std;
using namespace chrono;

const int RUN = 32;
const int N = 10'000'000; // Data size

// ---------- LADDER ALGORITHM ----------

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

struct HeapItem {
    int value;
    int list_idx;
    int elem_idx;
    bool operator>(const HeapItem& other) const {
        return value > other.value;
    }
};

vector<int> merge_ladders(vector<vector<int>>& lists) {
    vector<int> merged;
    priority_queue<HeapItem, vector<HeapItem>, greater<HeapItem>> heap;

    for (int i = 0; i < lists.size(); ++i)
        if (!lists[i].empty())
            heap.push({lists[i][0], i, 0});

    while (!heap.empty()) {
        HeapItem top = heap.top(); heap.pop();
        merged.push_back(top.value);
        if (top.elem_idx + 1 < lists[top.list_idx].size())
            heap.push({lists[top.list_idx][top.elem_idx + 1], top.list_idx, top.elem_idx + 1});
    }
    return merged;
}

vector<int> ladder(const vector<int>& array) {
    if (array.empty()) return {};
    vector<vector<int>> lad = { {array[0]} };

    for (int i = 1; i < array.size(); ++i) {
        int idx = binary_search_lad(lad, array[i]);
        if (idx == lad.size())
            lad.push_back({array[i]});
        else
            lad[idx].push_back(array[i]);
    }

    return merge_ladders(lad);
}

// ---------- TIMSORT ALGORITHM ----------

void insertionSort(vector<int>& arr, int left, int right) {
    for (int i = left + 1; i <= right; i++) {
        int temp = arr[i], j = i - 1;
        while (j >= left && arr[j] > temp) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = temp;
    }
}

void merge(vector<int>& arr, int l, int m, int r) {
    int len1 = m - l + 1, len2 = r - m;
    vector<int> left(len1), right(len2);
    for (int i = 0; i < len1; i++) left[i] = arr[l + i];
    for (int i = 0; i < len2; i++) right[i] = arr[m + 1 + i];

    int i = 0, j = 0, k = l;
    while (i < len1 && j < len2)
        arr[k++] = (left[i] <= right[j]) ? left[i++] : right[j++];
    while (i < len1) arr[k++] = left[i++];
    while (j < len2) arr[k++] = right[j++];
}

void timSort(vector<int>& arr, int n) {
    for (int i = 0; i < n; i += RUN)
        insertionSort(arr, i, min((i + RUN - 1), (n - 1)));
    for (int size = RUN; size < n; size = 2 * size)
        for (int left = 0; left < n; left += 2 * size) {
            int mid = min(left + size - 1, n - 1);
            int right = min(left + 2 * size - 1, n - 1);
            if (mid < right) merge(arr, left, mid, right);
        }
}

// ---------- BENCHMARK ----------

double benchmark_ladder(const vector<int>& input) {
    auto start = high_resolution_clock::now();
    vector<int> sorted = ladder(input);
    auto end = high_resolution_clock::now();
    return duration<double>(end - start).count();
}

double benchmark_timsort(const vector<int>& input) {
    vector<int> copy = input;
    auto start = high_resolution_clock::now();
    timSort(copy, copy.size());
    auto end = high_resolution_clock::now();
    return duration<double>(end - start).count();
}

int main() {
    mt19937 rng(42);
    uniform_int_distribution<int> dist(1, 10'000'000);

    vector<int> base(N);
    for (int i = 0; i < N; ++i)
        base[i] = dist(rng);

    vector<double> ladder_initial_times, ladder_post_times;
    vector<double> timsort_initial_times, timsort_post_times;

    // Ladder sort benchmarks
    for (int i = 0; i < 10; ++i) {
        ladder_initial_times.push_back(benchmark_ladder(base));
    }

    vector<int> post = base;
    post.push_back(500);
    for (int i = 0; i < 10; ++i) {
        ladder_post_times.push_back(benchmark_ladder(post));
    }

    // Timsort benchmarks
    for (int i = 0; i < 10; ++i) {
        timsort_initial_times.push_back(benchmark_timsort(base));
    }

    vector<int> post2 = base;
    post2.push_back(500);
    for (int i = 0; i < 10; ++i) {
        timsort_post_times.push_back(benchmark_timsort(post2));
    }

    auto avg = [](const vector<double>& v) {
        return accumulate(v.begin(), v.end(), 0.0) / v.size();
    };

    auto sum = [](const vector<double>& v) {
        return accumulate(v.begin(), v.end(), 0.0);
    };

    cout << fixed;
    cout << "=== Ladder Sort ===\n";
    cout << "Initial after 10 runs = " << sum(ladder_initial_times) << " seconds\n";
    cout << "After insertion and 10 runs = " << sum(ladder_post_times) << " seconds\n";
    cout << "Average of initial: " << avg(ladder_initial_times) << " seconds\n";
    cout << "Average of post-insert: " << avg(ladder_post_times) << " seconds\n\n";

    cout << "=== Custom Timsort ===\n";
    cout << "Initial after 10 runs = " << sum(timsort_initial_times) << " seconds\n";
    cout << "After insertion and 10 runs = " << sum(timsort_post_times) << " seconds\n";
    cout << "Average of initial: " << avg(timsort_initial_times) << " seconds\n";
    cout << "Average of post-insert: " << avg(timsort_post_times) << " seconds\n";

    return 0;
}
