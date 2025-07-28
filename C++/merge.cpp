#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

// Merge Sort implementation
void merge(vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // Create temp arrays
    vector<int> L(n1), R(n2);

    // Copy data to temp arrays
    for (int i = 0; i < n1; ++i)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; ++j)
        R[j] = arr[mid + 1 + j];

    // Merge the temp arrays back into arr[left..right]
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }

    // Copy any remaining elements of L[]
    while (i < n1)
        arr[k++] = L[i++];

    // Copy any remaining elements of R[]
    while (j < n2)
        arr[k++] = R[j++];
}

void merge_sort(vector<int>& arr, int left, int right) {
    if (left >= right) return;

    int mid = left + (right - left) / 2;
    merge_sort(arr, left, mid);
    merge_sort(arr, mid + 1, right);
    merge(arr, left, mid, right);
}

// Benchmark function
int main() {
    mt19937 rng(5);  // fixed seed for reproducibility
    uniform_int_distribution<int> dist(1, 10'000'000);

    double total_initial = 0.0;
    double total_post_insert = 0.0;

    for (int run = 1; run <= 10; ++run) {
        cout << "Run #" << run << ":\n";

        // Generate data
        vector<int> original(100'000'000);
        for (int i = 0; i < original.size(); ++i)
            original[i] = dist(rng);

        // Initial sort
        vector<int> data = original;  // copy
        auto start = chrono::high_resolution_clock::now();
        merge_sort(data, 0, data.size() - 1);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        cout << "  Merge sort time: " << duration.count() << " seconds\n";
        total_initial += duration.count();

        if (run == 1) {
            bool correct = is_sorted(data.begin(), data.end());
            cout << "  Correct: " << boolalpha << correct << "\n";
        }

        // Post-insert sort
        data.push_back(500);
        start = chrono::high_resolution_clock::now();
        merge_sort(data, 0, data.size() - 1);
        end = chrono::high_resolution_clock::now();
        duration = end - start;
        cout << "  Post-insert merge sort time: " << duration.count() << " seconds\n";
        total_post_insert += duration.count();
    }

    cout << "\n==== Summary for Merge Sort ====\n";
    cout << "Total time (Initial): " << total_initial << " seconds\n";
    cout << "Average time (Initial): " << total_initial / 10.0 << " seconds\n";
    cout << "Total time (Post-insert): " << total_post_insert << " seconds\n";
    cout << "Average time (Post-insert): " << total_post_insert / 10.0 << " seconds\n";

    return 0;
}

