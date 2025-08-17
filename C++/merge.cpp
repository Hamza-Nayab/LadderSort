#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

/// Stable mergesort with single reusable buffer and small-run cutoff.
static void mergesort(std::vector<int>& a, std::vector<int>& buf, int l, int r) {
    const int INSERTION_CUTOFF = 32;
    if (r - l <= INSERTION_CUTOFF) {
        for (int i = l + 1; i <= r; ++i) {
            int x = a[i], j = i - 1;
            while (j >= l && a[j] > x) { a[j + 1] = a[j]; --j; }
            a[j + 1] = x;
        }
        return;
    }
    int m = l + (r - l) / 2;
    mergesort(a, buf, l, m);
    mergesort(a, buf, m + 1, r);

    // If already ordered, skip merge (helps nearly-sorted cases)
    if (a[m] <= a[m + 1]) return;

    int i = l, j = m + 1, k = l;
    while (i <= m && j <= r) buf[k++] = (a[i] <= a[j]) ? a[i++] : a[j++];
    while (i <= m) buf[k++] = a[i++];
    while (j <= r) buf[k++] = a[j++];

    for (i = l; i <= r; ++i) a[i] = buf[i];
}

static void mergesort_with_buf(std::vector<int>& a) {
    if (a.empty()) return;
    std::vector<int> buf(a.size());       // one allocation
    mergesort(a, buf, 0, (int)a.size() - 1);
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
        vector<int> original(10'000'000);
        for (int i = 0; i < original.size(); ++i)
            original[i] = dist(rng);

        // Initial sort
        vector<int> data = original;  // copy
        auto start = chrono::high_resolution_clock::now();
        mergesort_with_buf(data);
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
        mergesort_with_buf(data);
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