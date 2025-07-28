#include <iostream>
#include <vector>
#include <queue>
#include <random>
#include <chrono>


// Declare kWayMerge before it's used
std::vector<int> kWayMerge(const std::vector<int>& arr, const std::vector<int>& sortedEnds);

std::vector<int> getSortedIndices(const std::vector<int>& arr) {
    std::vector<int> result;
    for (int i = 1; i < arr.size(); ++i) {
        if (arr[i] < arr[i - 1]) {
            result.push_back(i - 1);
        }
    }
    result.push_back(arr.size() - 1); // end of last sorted segment
    return kWayMerge(arr, result);    // merge on-the-fly
}

std::vector<int> kWayMerge(const std::vector<int>& arr, const std::vector<int>& sortedEnds) {
    struct HeapNode {
        int value;
        int fromList;
        int indexInList;
        bool operator>(const HeapNode& other) const {
            return value > other.value;
        }
    };

    std::vector<std::pair<int, int>> ranges;
    int start = 0;
    for (int end : sortedEnds) {
        ranges.emplace_back(start, end);
        start = end + 1;
    }

    std::priority_queue<HeapNode, std::vector<HeapNode>, std::greater<HeapNode>> minHeap;

    for (int i = 0; i < ranges.size(); ++i) {
        int startIdx = ranges[i].first;
        if (startIdx <= ranges[i].second) {
            minHeap.push({arr[startIdx], i, startIdx});
        }
    }

    std::vector<int> result;
    while (!minHeap.empty()) {
        auto node = minHeap.top();
        minHeap.pop();
        result.push_back(node.value);

        int listIdx = node.fromList;
        int nextIdx = node.indexInList + 1;
        if (nextIdx <= ranges[listIdx].second) {
            minHeap.push({arr[nextIdx], listIdx, nextIdx});
        }
    }

    return result;
}

bool isSorted(const std::vector<int>& arr) {
    for (size_t i = 1; i < arr.size(); ++i) {
        if (arr[i] < arr[i - 1]) return false;
    }
    return true;
}

int main() {
    const int N = 10'000'000;
    std::vector<int> arr(N);

    std::mt19937 rng(42); // fixed seed
    std::uniform_int_distribution<int> noise_dist(1, 1'000'000);
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);

    int current = 0;
    for (int i = 0; i < N; ++i) {
        if (prob(rng) < 0.8f) {
            // 80% of the time: add sorted element
            current += rng() % 10;  // small increasing step
            arr[i] = current;
        } else {
            // 20% of the time: inject noise (breaks sort)
            arr[i] = noise_dist(rng);
        }
    }

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<int> sorted = getSortedIndices(arr); // includes kWayMerge

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Validate sorting
    if (isSorted(sorted)) {
        std::cout << "✅ Array is sorted correctly.\n";
    } else {
        std::cout << "❌ Array is NOT sorted correctly.\n";
    }

    // Show sample
    std::cout << "First 10 sorted elements:\n";
    for (int i = 0; i < 10 && i < sorted.size(); ++i) {
        std::cout << sorted[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Time taken: " << duration.count() << " seconds\n";
    return 0;
}
