#include <iostream>
#include <vector>
#include <random>
#include <chrono>
using namespace std;

// Custom max function
int getMax(const vector<int>& arr) {
    int maxVal = arr[0];
    for (int i = 1; i < arr.size(); i++) {
        if (arr[i] > maxVal)
            maxVal = arr[i];
    }
    return maxVal;
}

// Counting sort for digit place
void countingSort(vector<int>& arr, int exp) {
    int n = arr.size();
    vector<int> output(n);
    int count[10] = {0};

    for (int i = 0; i < n; i++)
        count[(arr[i] / exp) % 10]++;

    for (int i = 1; i < 10; i++)
        count[i] += count[i - 1];

    for (int i = n - 1; i >= 0; i--) {
        int digit = (arr[i] / exp) % 10;
        output[count[digit] - 1] = arr[i];
        count[digit]--;
    }

    for (int i = 0; i < n; i++)
        arr[i] = output[i];
}

// Main radix sort
void radixSort(vector<int>& arr) {
    int maxVal = getMax(arr);
    for (int exp = 1; maxVal / exp > 0; exp *= 10)
        countingSort(arr, exp);
}

// Custom is_sorted check
bool isSorted(const vector<int>& arr) {
    for (int i = 1; i < arr.size(); i++) {
        if (arr[i] < arr[i - 1])
            return false;
    }
    return true;
}
int main() {
    mt19937 rng(5);  // fixed seed for reproducibility
    uniform_int_distribution<int> dist(1, 10'000'000);

    double total_initial = 0.0;
    double total_post_insert = 0.0;

    for (int run = 1; run <= 10; ++run) {
        cout << "Run #" << run << ":\n";

        vector<int> original(100'000'000);
        for (int i = 0; i < original.size(); ++i)
            original[i] = dist(rng);

        // Initial sort
        vector<int> data = original;
        auto start = chrono::high_resolution_clock::now();
        radixSort(data);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        cout << "  Radix sort time: " << duration.count() << " seconds\n";
        total_initial += duration.count();

        if (run == 1) {
            bool correct = isSorted(data);
            cout << "  Correct: " << boolalpha << correct << "\n";
        }

        // Post-insert sort
        data.push_back(500);
        start = chrono::high_resolution_clock::now();
        radixSort(data);
        end = chrono::high_resolution_clock::now();
        duration = end - start;
        cout << "  Post-insert Radix sort time: " << duration.count() << " seconds\n";
        total_post_insert += duration.count();
    }

    cout << "\n==== Summary for Radix Sort (No STL) ====\n";
    cout << "Total time (Initial): " << total_initial << " seconds\n";
    cout << "Average time (Initial): " << total_initial / 10.0 << " seconds\n";
    cout << "Total time (Post-insert): " << total_post_insert << " seconds\n";
    cout << "Average time (Post-insert): " << total_post_insert / 10.0 << " seconds\n";

    return 0;
}
