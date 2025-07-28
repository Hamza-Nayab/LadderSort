#include <iostream>
#include <vector>
#include <algorithm>  // for std::sort, std::is_sorted
#include <random>
#include <chrono>

using namespace std;

int main() {
    mt19937 rng(5);  // fixed seed for fair comparison
    uniform_int_distribution<int> dist(1, 10'000'000);

    double total_initial = 0.0;
    double total_post_insert = 0.0;

    for (int run = 1; run <= 10; ++run) {
        cout << "Run #" << run << ":\n";

        // Generate random input
        vector<int> original(1'000'000);
        for (int i = 0; i < original.size(); ++i)
            original[i] = dist(rng);

        // Initial std::sort
        vector<int> data = original;
        auto start = chrono::high_resolution_clock::now();
        sort(data.begin(), data.end());
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        cout << "  std::sort time: " << duration.count() << " seconds\n";
        total_initial += duration.count();

        if (run == 1) {
            bool correct = is_sorted(data.begin(), data.end());
            cout << "  Correct: " << boolalpha << correct << "\n";
        }

        // Post-insert sort
        data.push_back(500);
        start = chrono::high_resolution_clock::now();
        sort(data.begin(), data.end());
        end = chrono::high_resolution_clock::now();
        duration = end - start;
        cout << "  Post-insert std::sort time: " << duration.count() << " seconds\n";
        total_post_insert += duration.count();
    }

    cout << "\n==== Summary for std::sort ====\n";
    cout << "Total time (Initial): " << total_initial << " seconds\n";
    cout << "Average time (Initial): " << total_initial / 10.0 << " seconds\n";
    cout << "Total time (Post-insert): " << total_post_insert << " seconds\n";
    cout << "Average time (Post-insert): " << total_post_insert / 10.0 << " seconds\n";

    return 0;
}
