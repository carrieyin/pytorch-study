import numpy as np

a = np.arange(24).reshape(2, 3, 4)
print(a)

print("sum a axis 0")
print(a.sum(axis=0))
print(a[0, :, :] + a[1, :, :])

print("sum a axis 1")
print(a.sum(axis=1))
print("axis 1-----:")
print(a[:, 0, :])
print(a[:, 1, :])
print(a[:, 2, :])


print("axis 2-----")
print(a[:, :, 0])
print(a[:, :, 1])
print(a[:, :, 2])
print(a[:, :, 3])
#print(a[1:2:1,0, 1:2:1])

print("axis num 2")
b = np.arange(6).reshape(2,3)
print(b)

print(b.sum(axis=0))
print(b[0, :] + b[1 ,:])

print(b.sum(axis=1))
print(b[:, 0] + b[:,1] + b[:, 2])

c = np.array([1,2,3]).shape
print(c)