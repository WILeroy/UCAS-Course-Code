import numpy as np

A = np.array([1, 0, 0, 0, 0, -2,  0,  1, -1, -1, -1, -2,
              0, 1, 0, 0, 0,  0, -2, -2, -2,  2, -1, -5,
              0, 0, 1, 0, 0, -1,  2,  0,  0,  0, -1, 2,
              0, 0, 0, 1, 0,  1,  2,  0,  0,  0, -1, 6,
              0, 0, 0, 0, 1,  1, -2,  0,  0,  0, -1, 2]).reshape(5, 12).astype(np.float)

b_idx = np.array([0, 1, 2, 3, 4])

in_idx = 10
out_idx = np.argmax(-A[:, 11])
print(out_idx)

for i in range(5):
    if i == out_idx:
        A[i, :] /= A[i, in_idx]
    else:
        A[i, :] -= A[out_idx, :] * A[i, in_idx] / A[out_idx, in_idx]

print(A)

t = in_idx
in_idx = (b_idx[out_idx] + 5) % 10
b_idx[out_idx] = t
print(b_idx)

minz = 1525123
out_idx = -1
for i in range(5):
    if A[i, in_idx] > 0:
        z = A[i, 11] / A[i, in_idx]
        if z < minz:
            minz = z
            out_idx = i

print(out_idx, in_idx)

for i in range(5):
    if i == out_idx:
        A[i, :] /= A[i, in_idx]
    else:
        A[i, :] -= A[out_idx, :] * A[i, in_idx] / A[out_idx, in_idx]

print(A)

t = in_idx
in_idx = (b_idx[out_idx] + 5) % 10
b_idx[out_idx] = t
print(b_idx)

minz = 1525123
out_idx = -1
for i in range(5):
    if A[i, in_idx] > 0:
        z = A[i, 11] / A[i, in_idx]
        if z < minz:
            minz = z
            out_idx = i

print(out_idx, in_idx)

for i in range(5):
    if i == out_idx:
        A[i, :] /= A[i, in_idx]
    else:
        A[i, :] -= A[out_idx, :] * A[i, in_idx] / A[out_idx, in_idx]

print(A)

t = in_idx
in_idx = (b_idx[out_idx] + 5) % 10
b_idx[out_idx] = t
print(b_idx)

minz = 1525123
out_idx = -1
for i in range(5):
    if A[i, in_idx] > 0:
        z = A[i, 11] / A[i, in_idx]
        if z < minz:
            minz = z
            out_idx = i

print(out_idx, in_idx)

for i in range(5):
    if i == out_idx:
        A[i, :] /= A[i, in_idx]
    else:
        A[i, :] -= A[out_idx, :] * A[i, in_idx] / A[out_idx, in_idx]

print(A)

t = in_idx
in_idx = (b_idx[out_idx] + 5) % 10
b_idx[out_idx] = t
print(b_idx)

minz = 1525123
out_idx = -1
for i in range(5):
    if A[i, in_idx] > 0:
        z = A[i, 11] / A[i, in_idx]
        if z < minz:
            minz = z
            out_idx = i

print(out_idx, in_idx)

print(A[:, 11])