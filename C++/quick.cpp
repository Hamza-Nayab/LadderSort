#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

// QuickSort function
void quicksort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int mid = (low + high) / 2;
        swap(arr[mid], arr[high]); // move pivot to end
        int pivot = arr[high];
        int i = low - 1;

        for (int j = low; j < high; ++j) {
            if (arr[j] < pivot) {
                ++i;
                swap(arr[i], arr[j]);
            }
        }
        swap(arr[i + 1], arr[high]);
        int pi = i + 1;

        quicksort(arr, low, pi - 1);
        quicksort(arr, pi + 1, high);
    }
}

int main() {
    const int N = 10'000'000;
    mt19937 rng(45);  // Fixed seed for reproducibility
    uniform_int_distribution<int> dist(1, 10'000'000);

    double total_initial = 0.0;
    double total_post_insert = 0.0;

    for (int run = 1; run <= 10; ++run) {
        cout << "Run #" << run << ":\n";

        // Generate fresh random array
        vector<int> arr(N);
        for (int i = 0; i < N; ++i)
            arr[i] = dist(rng);

        // Initial Quicksort
        auto start = chrono::high_resolution_clock::now();
        quicksort(arr, 0, arr.size() - 1);
        auto end = chrono::high_resolution_clock::now();

        chrono::duration<double> elapsed = end - start;
        cout << "  Initial quicksort time: " << elapsed.count() << " seconds\n";
        total_initial += elapsed.count();

        if (run == 1) {
            cout << "  Correct: " << boolalpha << is_sorted(arr.begin(), arr.end()) << "\n";
        }

        // Insert one number and sort again
        arr.push_back(50);
        start = chrono::high_resolution_clock::now();
        quicksort(arr, 0, arr.size() - 1);
        end = chrono::high_resolution_clock::now();

        elapsed = end - start;
        cout << "  Post-insert quicksort time: " << elapsed.count() << " seconds\n";
        total_post_insert += elapsed.count();
    }

    // Summary
    cout << "\n==== Summary for Quicksort ====\n";
    cout << "Total time (Initial): " << total_initial << " seconds\n";
    cout << "Average time (Initial): " << total_initial / 10.0 << " seconds\n";
    cout << "Total time (Post-insert): " << total_post_insert << " seconds\n";
    cout << "Average time (Post-insert): " << total_post_insert / 10.0 << " seconds\n";

    return 0;
}
