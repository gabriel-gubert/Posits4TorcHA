# Posits4TorcHA
Companion Python Package for [Posits4Torch](https://github.com/gabriel-gubert/Posits4Torch) to be installed on an Adaptive Computing Platform (ACP) such as a Kria KV260 Vision Starter Kit / K26 System-on-Module (SOM). Posits4TorcHA enables Hardware Acceleration (HA) of General Matrix Multiplication (GEMM) with Posits through an AXI Direct Memory Access (DMA)-compliant 2D Posit Multiply-Accumulate (MAC) Unit Array.

## Instructions
On an ACP such as a Kria KV260 Vision Starter Kit / K26 SOM:

1. Install [Python Productiviy for ZYNQ (PYNQ)](https://www.pynq.io/) (see [Getting Started](https://pynq.readthedocs.io/en/latest/getting_started.html) with PYNQ or [Kria-PYNQ](https://github.com/Xilinx/Kria-PYNQ));
2. Install Posits4TorcHA.
```
git clone https://github.com/gabriel-gubert/Posits4TorcHA.git
cd Posits4TorcHA
pip install -r requirements.txt
pip install .
```
## Usage
### Standalone
```
import numpy as np
from Posits4TorcHA import Accel

Part = 'KV260'
R = 8
C = 8
N = 8
Es = 2
QSize = 128
Depth = 512
num_threads = 4

accel = Accel(Part, R, C, N, Es, QSize, Depth, num_threads)

A = np.zeros([8, 8], dtype = np.uint8) + 72
B = np.zeros([8, 8], dtype = np.uint8) + 72

Y = accel.GEMM(A, B.T)
```
### Client-Server Architecture w/ Posits4Torch as the Client and Posits4TorcHA as the Server
```
python3 HTTPServer.py [--host <host>] [--port <port>] [--verbose] [--report_timings]
```
