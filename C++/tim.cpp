#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
using namespace std;

const int RUN = 32;

// Insertion sort for small segments
void insertionSort(vector<int>& arr, int left, int right) {
    for (int i = left + 1; i <= right; i++) {
        int temp = arr[i];
        int j = i - 1;
        while (j >= left && arr[j] > temp) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = temp;
    }
}

// Merge function used in Timsort
void merge(vector<int>& arr, int l, int m, int r) {
    int len1 = m - l + 1, len2 = r - m;
    vector<int> left(len1), right(len2);
    for (int i = 0; i < len1; i++)
        left[i] = arr[l + i];
    for (int i = 0; i < len2; i++)
        right[i] = arr[m + 1 + i];

    int i = 0, j = 0, k = l;

    while (i < len1 && j < len2) {
        if (left[i] <= right[j])
            arr[k++] = left[i++];
        else
            arr[k++] = right[j++];
    }

    while (i < len1)
        arr[k++] = left[i++];
    while (j < len2)
        arr[k++] = right[j++];
}

// Timsort main algorithm
void timSort(vector<int>& arr, int n) {
    for (int i = 0; i < n; i += RUN)
        insertionSort(arr, i, min((i + RUN - 1), (n - 1)));

    for (int size = RUN; size < n; size = 2 * size) {
        for (int left = 0; left < n; left += 2 * size) {
            int mid = min((left + size - 1), (n - 1));
            int right = min((left + 2 * size - 1), (n - 1));
            if (mid < right)
                merge(arr, left, mid, right);
        }
    }
}
int main() {
    mt19937 rng(5);  // fixed seed
    uniform_int_distribution<int> dist(1, 10'000'000);

    double total_initial = 0.0;
    double total_post_insert = 0.0;

    for (int run = 1; run <= 10; ++run) {
        cout << "Run #" << run << ":\n";

        vector<int> original(100'000'000);
        for (int i = 0; i < original.size(); ++i)
            original[i] = dist(rng);

        // Initial sort
        vector<int> data = original;  // copy
        auto start = chrono::high_resolution_clock::now();
        timSort(data, data.size());
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        cout << "  Timsort time: " << duration.count() << " seconds\n";
        total_initial += duration.count();

        if (run == 1) {
            bool correct = is_sorted(data.begin(), data.end());
            cout << "  Correct: " << boolalpha << correct << "\n";
        }

        // Insert element and sort again
        data.push_back(500);
        start = chrono::high_resolution_clock::now();
        timSort(data, data.size());
        end = chrono::high_resolution_clock::now();
        duration = end - start;
        cout << "  Post-insert Timsort time: " << duration.count() << " seconds\n";
        total_post_insert += duration.count();
    }

    cout << "\n==== Summary for Timsort ====\n";
    cout << "Total time (Initial): " << total_initial << " seconds\n";
    cout << "Average time (Initial): " << total_initial / 10.0 << " seconds\n";
    cout << "Total time (Post-insert): " << total_post_insert << " seconds\n";
    cout << "Average time (Post-insert): " << total_post_insert / 10.0 << " seconds\n";

    return 0;
}

