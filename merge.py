import random
import time

def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr)//2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result



random.seed(5)
b = [random.randint(1, 1000000) for _ in range(1000000)]

print("Running merge sort...")
start = time.time()
merge_res = merge_sort(b)
end = time.time()
print("Merge sort time:", end - start)
print("Correct:", merge_res == sorted(b))
merge_res.append(1001)
start = time.time()
merge_res = merge_sort(merge_res)
end = time.time()
print("Post-insert time:", round(end - start, 4), "seconds")