#include <iostream>
#include <vector>
#include <queue>
#include <random>
#include <chrono>
#include <algorithm>

using namespace std;

// === Config Flags ===
const bool PRINT_EACH_RUN = false;
const int NUM_ELEMENTS = 10'000'000;
const int NUM_RUNS = 10;


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

// === MinHeap item ===
struct HeapItem {
    int value, list_idx, elem_idx;
    bool operator>(const HeapItem& other) const {
        return value > other.value;
    }
};

// === Merge ladders ===
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

// === Ladder Sort ===
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

// === Benchmark for normal case ===
void benchmark_case(const string& case_name, const vector<int>& input) {
    double total = 0.0;
    cout << "\n==== Benchmark: " << case_name << " ====\n";

    for (int run = 1; run <= NUM_RUNS; ++run) {
        vector<int> data = input;
        auto start = chrono::high_resolution_clock::now();
        vector<int> sorted = ladder(data);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        double elapsed = duration.count();
        total += elapsed;

        if (PRINT_EACH_RUN) cout << "Run #" << run << ": " << elapsed << " seconds\n";

        if (run == 1) {
            vector<int> expected = input;
            sort(expected.begin(), expected.end());
            cout << "  Correct: " << boolalpha << (sorted == expected) << "\n";
        }
    }
    cout << "Avg Time: " << (total / NUM_RUNS) << " seconds\n";
}

// === Benchmark for repeated insertions ===
void benchmark_insert_case(const vector<int>& input) {
    cout << "\n==== Post-insert Benchmark (10 runs) ====\n";
    double total = 0.0;

    for (int run = 1; run <= NUM_RUNS; ++run) {
        vector<int> data = ladder(input);
        data.push_back(500);  // Insert 1 new value
        auto start = chrono::high_resolution_clock::now();
        vector<int> sorted = ladder(data);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        double elapsed = duration.count();
        total += elapsed;

        if (PRINT_EACH_RUN) cout << "Run #" << run << ": " << elapsed << " seconds\n";

        if (run == 1) {
            vector<int> expected = data;
            sort(expected.begin(), expected.end());
            cout << "  Correct: " << boolalpha << (sorted == expected) << "\n";
        }
    }

    cout << "Avg Insert-1-at-end Time: " << (total / NUM_RUNS) << " seconds\n";
}

// === Main function ===
int main() {
    mt19937 rng(45);

    auto generate_tests = [&](bool unique_only) {
        vector<vector<int>> datasets;
        vector<string> names;

        // === Base generator ===
        vector<int> base(NUM_ELEMENTS);
        if (unique_only) {
            iota(base.begin(), base.end(), -NUM_ELEMENTS/2); // unique integers
            shuffle(base.begin(), base.end(), rng);
        } else {
            uniform_int_distribution<int> dist(-5'000'000, 5'000'000);
            for (int &x : base) x = dist(rng);
        }

        // === Case 1: Block Sorted (20 blocks) ===
        {
            vector<int> arr(NUM_ELEMENTS);
            int block_size = NUM_ELEMENTS / 20;
            int val = -NUM_ELEMENTS / 2;
            for (int b = 0; b < 20; ++b) {
                for (int i = 0; i < block_size; ++i)
                    arr[b * block_size + i] = val++;
                // Shuffle each block internally
                shuffle(arr.begin() + b * block_size,
                        arr.begin() + (b + 1) * block_size, rng);
            }
            datasets.push_back(arr);
            names.push_back("Block Sorted (20 blocks)");
        }

        // === Case 2: Few Unique + Stable Groups ===
        {
            vector<int> arr;
            arr.reserve(NUM_ELEMENTS);
            for (int v = 1; v <= 10; ++v) {
                for (int count = 0; count < NUM_ELEMENTS / 10; ++count)
                    arr.push_back(v);
            }
            // Shuffle only within each group
            for (int v = 0; v < 10; ++v) {
                shuffle(arr.begin() + v * (NUM_ELEMENTS / 10),
                        arr.begin() + (v + 1) * (NUM_ELEMENTS / 10), rng);
            }
            datasets.push_back(arr);
            names.push_back("Few Unique + Stable Groups");
        }

        // === Case 3: Random Input ===
        {
            datasets.push_back(base);
            names.push_back("Random Input");
        }

        // === Case 4: Sorted Input ===
        {
            vector<int> arr = base;
            sort(arr.begin(), arr.end());
            datasets.push_back(arr);
            names.push_back("Sorted Input");
        }

        // === Case 5: Reverse Sorted ===
        {
            vector<int> arr = datasets.back();
            reverse(arr.begin(), arr.end());
            datasets.push_back(arr);
            names.push_back("Reverse Sorted");
        }

        // === Case 6: Nearly Sorted (95%) ===
        {
            vector<int> arr = datasets[names.size() - 2]; // sorted version
            int swap_count = NUM_ELEMENTS / 20; // 5%
            for (int i = 0; i < swap_count; ++i)
                swap(arr[rng() % NUM_ELEMENTS], arr[rng() % NUM_ELEMENTS]);
            datasets.push_back(arr);
            names.push_back("Nearly Sorted (95%)");
        }

        // === Case 7: Few Unique Elements at End ===
        {
            vector<int> arr = datasets[names.size() - 3]; // sorted version
            uniform_int_distribution<int> few_dist(1, 10);
            for (int i = NUM_ELEMENTS - 1000; i < NUM_ELEMENTS; ++i)
                arr[i] = few_dist(rng);
            datasets.push_back(arr);
            names.push_back("Few Unique Elements at End");
        }

        // === Case 8: Post-Insertion Sorted ===
        {
            vector<int> arr = datasets[names.size() - 4]; // sorted version
            arr.push_back(unique_only ? NUM_ELEMENTS + 1 : 500);
            datasets.push_back(arr);
            names.push_back("Post-Insertion Sorted");
        }

        // === Run all benchmarks ===
        for (size_t i = 0; i < datasets.size(); ++i)
            benchmark_case((unique_only ? "[Unique] " : "[General] ") + names[i], datasets[i]);
    };

    // === Run for both general and unique datasets ===
    generate_tests(false); // general dataset (duplicates possible)
    generate_tests(true);  // unique dataset

    return 0;
}
