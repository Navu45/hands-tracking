import numpy as np
import heapq

n = np.random.randint(1, 5)
m = np.random.randint(1, 5)

a = list(np.arange(n))
b = list(np.random.randint(0, m, size=n))

heaps = [[] for _ in range(m)]
for i in a:
    heapq.heappush(heaps[b[i] - 1], (2.0 ** (-i), i))

print(a)
print(b)
print(heaps)
best_list = []
last_b = -1
relevance = 0
for _ in range(n):
    next_item = None
    for heap in heaps:
        print(heap, next_item)
        if len(heap) > 0 and next_item is None or heap[0] > next_item and b[heap[0][1]] != last_b:
            next_item = heap
    if next_item is None:
        break
    print(next_item)
    # last_b = b[next_item[1]]
    # relevance += 2 ** (-len(best_list)) * next_item[0]
