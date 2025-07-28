import random
import time

def binary_search_lad(lad, target):
    low, high = 0, len(lad)
    while low < high:
        mid = (low + high) // 2
        if lad[mid][-1] > target:
            low = mid + 1
        else:
            high = mid
    return low

def ladder(array):
    if not array:
        return []

    lad = [[array[0]]]

    for a in array[1:]:
        i = binary_search_lad(lad, a)
        if i == len(lad):
            lad.append([a])
        else:
            lad[i].append(a)

    return merge_ladders(lad)

class MinHeap:
    def __init__(self):
        self.heap = []

    def push(self, val):
        self.heap.append(val)
        self._sift_up(len(self.heap) - 1)

    def pop(self):
        if not self.heap:
            return None
        self._swap(0, len(self.heap) - 1)
        min_val = self.heap.pop()
        self._sift_down(0)
        return min_val

    def _sift_up(self, idx):
        parent = (idx - 1) // 2
        while idx > 0 and self.heap[idx][0] < self.heap[parent][0]:
            self._swap(idx, parent)
            idx = parent
            parent = (idx - 1) // 2

    def _sift_down(self, idx):
        size = len(self.heap)
        while True:
            smallest = idx
            left = 2 * idx + 1
            right = 2 * idx + 2
            if left < size and self.heap[left][0] < self.heap[smallest][0]:
                smallest = left
            if right < size and self.heap[right][0] < self.heap[smallest][0]:
                smallest = right
            if smallest == idx:
                break
            self._swap(idx, smallest)
            idx = smallest

    def _swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def __len__(self):
        return len(self.heap)

def merge_ladders(lists):
    merged = []
    heap = MinHeap()
    
    for i in range(len(lists)):
        if lists[i]:
            heap.push((lists[i][0], i, 0))

    while len(heap) > 0:
        val, list_idx, elem_idx = heap.pop()
        merged.append(val)
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heap.push((next_val, list_idx, elem_idx + 1))

    return merged

# Benchmarking
random.seed(5)
b = [random.randint(1, 1000000) for _ in range(1000000)]

start = time.time()
a = ladder(b)
end = time.time()
print("Initial time:", round(end - start, 4), "seconds")
print("Correct:", a == sorted(b))
# Incremental insert test
a.append(500)
start = time.time()
c = ladder(a)
end = time.time()
print("Post-insert time:", round(end - start, 4), "seconds")