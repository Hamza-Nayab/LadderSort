import random
import time
def quicksort(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr)//2]
    left = []
    middle = []
    right = []

    for x in arr:
        if x < pivot:
            left.append(x)
        elif x == pivot:
            middle.append(x)
        else:
            right.append(x)

    return quicksort(left) + middle + quicksort(right)

b = [random.randint(1, 1000000) for _ in range(1000000)]
#b = list(range(1,100000))  # Reverse sorted list for testing
print("Running quicksort...")
start = time.time()
a = quicksort(b)
end = time.time()
print("Quicksort time:", end - start)
print("Correct:", a == sorted(b))

a.append(1001)
start = time.time()
quick_res = quicksort(a)
end = time.time()
print("Post-insert time:", round(end - start, 4), "seconds")