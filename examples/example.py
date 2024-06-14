import sys

import numpy as np

from PQTorcHA import Accel

in_channels = int(sys.argv[1])
out_channels = int(sys.argv[2])
kernel_size = [int(sys.argv[3]), int(sys.argv[4])]
padding = int(sys.argv[5])
stride = int(sys.argv[6])

H = int(sys.argv[7])
W = int(sys.argv[8])

Part = sys.argv[9]
R = int(sys.argv[10])
C = int(sys.argv[11])
N = int(sys.argv[12])
Es = int(sys.argv[13])
QSize = int(sys.argv[14])
Depth = int(sys.argv[15])
num_threads = int(sys.argv[16])

remote = True if sys.argv[17] == 'remote' else False
debug = True if sys.argv[18] == 'debug' else False

if not remote:
    default_accel = Accel(Part, R, C, N, Es, QSize, Depth, num_threads)
else:
    default_remoteaccel = RemoteAccel('127.0.0.1', 8080, -1, Part, R, C, N, Es, QSize, Depth, num_threads)

Hout = (H + 2 * padding - kernel_size[0]) // stride + 1
Wout = (W + 2 * padding - kernel_size[1]) // stride + 1

# default_rng = np.random.default_rng()

# I = default_rng.integers(0, 255, [in_channels, H, W])
I = np.zeros([in_channels, H, W], dtype = np.uint8) + 72

X = np.pad(I, ((0, 0), (padding, padding), (padding, padding)))
# A = default_rng.integers(0, 9, [out_channels, in_channels, kernel_size[0], kernel_size[1]])
A = np.zeros([out_channels, in_channels, kernel_size[0], kernel_size[1]], dtype = np.uint8) + 72

C = np.zeros([Wout * Hout, in_channels * kernel_size[0] * kernel_size[1]], dtype = np.uint8)

for i in range(Hout):
    for j in range(Wout):
        C[i * Wout + j, :] = X[:, i * stride:i * stride + kernel_size[1], j * stride:j * stride + kernel_size[0]].flatten()

if remote:
    Y = default_remoteaccel.GEMM(C, A.reshape(out_channels, -1).T)
else:
    Y = default_accel.GEMM(C, A.reshape(out_channels, -1).T)

Y = Y.reshape(Hout, Wout, out_channels).transpose(2, 0, 1)

if debug:
    print("X: {} {}".format(X, X.shape, type(X.item(0))))
    print("A: {} {}".format(A, A.shape, type(A.item(0))))
    print("Y: {} {}".format(Y, Y.shape, type(Y.item(0))))
