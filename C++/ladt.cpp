#include <iostream>
#include <vector>
#include <queue>
#include <random>
#include <chrono>
#include <algorithm>

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

// Ladder sort
vector<int> ladder(const vector<int>& array) {
    if (array.empty()) return {};
    vector<vector<int>> lad = { {array[0]} };
    for (int i = 1; i < array.size(); ++i) {
        int a = array[i];
        int idx = binary_search_lad(lad, a);
        if (idx == lad.size()) lad.push_back({a});
        else lad[idx].push_back(a);
    }
    return merge_ladders(lad);
}

// Run benchmark on given input vector
void benchmark_case(const string& case_name, const vector<int>& input) {
    double total = 0.0;

    cout << "\n==== Benchmark: " << case_name << " ====\n";

    for (int run = 1; run <= 10; ++run) {
        vector<int> data = input;
        auto start = chrono::high_resolution_clock::now();
        vector<int> sorted = ladder(data);
        auto end = chrono::high_resolution_clock::now();

        chrono::duration<double> duration = end - start;
        double elapsed = duration.count();
        total += elapsed;
        cout << "Run #" << run << ": " << elapsed << " seconds\n";

        if (run == 1) {
            vector<int> expected = input;
            sort(expected.begin(), expected.end());
            cout << "  Correct: " << boolalpha << (sorted == expected) << "\n";
        }
    }

    cout << "Avg Time: " << (total / 10.0) << " seconds\n";
}

// Separate benchmark for post-insert case
void benchmark_insert_case(const vector<int>& input) {
    cout << "\n==== Post-insert Benchmark ====\n";

    vector<int> data = ladder(input);
    data.push_back(500); // Insert 1 new value

    auto start = chrono::high_resolution_clock::now();
    vector<int> sorted = ladder(data);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    cout << "Post-insert sort time: " << duration.count() << " seconds\n";
}

int main() {
    const int N = 10'000'000;
    mt19937 rng(5);
    uniform_int_distribution<int> dist(1, 10'000'000);

    // Case 1: Random
    vector<int> random_input(N);
    for (int& x : random_input) x = dist(rng);
    benchmark_case("Random Input", random_input);

    // Case 2: Sorted
    vector<int> sorted_input = random_input;
    sort(sorted_input.begin(), sorted_input.end());
    benchmark_case("Sorted Input", sorted_input);

    // Case 3: Reverse Sorted
    vector<int> reverse_input = sorted_input;
    reverse(reverse_input.begin(), reverse_input.end());
    benchmark_case("Reverse Sorted", reverse_input);

    // Case 4: Nearly Sorted (95%)
    vector<int> nearly_sorted = sorted_input;
    for (int i = 0; i < N / 20; ++i)
        swap(nearly_sorted[rng() % N], nearly_sorted[rng() % N]);
    benchmark_case("Nearly Sorted", nearly_sorted);

    // Case 5: Few Unique
    uniform_int_distribution<int> few_dist(1, 10);
    vector<int> few_unique(N);
    for (int& x : few_unique) x = few_dist(rng);
    benchmark_case("Few Unique Elements", few_unique);

    // Insert-1-at-end benchmark
    benchmark_insert_case(sorted_input);

    return 0;
}
