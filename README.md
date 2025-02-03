# Posits4TorcHA
Companion Python Package for [Posits4Torch](https://github.com/gabriel-gubert/Posits4Torch) to be installed on an Adaptive Computing Platform (ACP) such as a Kria KV260 Vision Starter Kit / K26 System-on-Module (SOM). Posits4TorcHA enables Hardware Acceleration (HA) of General Matrix Multiplication (GEMM) with Posits through a PYNQ Overlay for a AXI Direct Memory Access (DMA)-compliant 2D Posit Multiply-Accumulate (MAC) Unit Array.

## Installation
On an ACP such as a Kria KV260 Vision Starter Kit / K26 SOM:

1. Install [Python Productiviy for ZYNQ (PYNQ)](https://www.pynq.io/) (see [Getting Started](https://pynq.readthedocs.io/en/latest/getting_started.html) with PYNQ or [Kria-PYNQ](https://github.com/Xilinx/Kria-PYNQ));
2. Install Cython;
```
pip install Cython
```
3. Install Posits4TorcHA;
```
git clone https://github.com/gabriel-gubert/Posits4TorcHA.git
cd Posits4TorcHA
pip install .
```
4. (Optional) Install Posits4Torch.
```
git clone --recursive https://github.com/gabriel-gubert/Posits4Torch.git
cd Posits4Torch
source install.sh
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

A = np.zeros([2, 2], dtype = np.uint8) + 72 # A = [[2.0, 2.0], [2.0, 2.0]]
B = np.zeros([2, 2], dtype = np.uint8) + 72 # B = [[2.0, 2.0], [2.0, 2.0]]
 
Y = accel.GEMM(A, B.T) # Y = [[8.0, 8.0], [8.0, 8.0]]
```
### Standalone w/ Posits4Torch
```
import numpy as np
from Posits4TorcHA import Accel
from Posits4Torch.Utilities import gettype, astype, frombin, tobin

Part = 'KV260'
R = 8
C = 8
N = 8
Es = 2
QSize = 128
Depth = 512
num_threads = 4

accel = Accel(Part, R, C, N, Es, QSize, Depth, num_threads)

A = tobin(astype(np.zeros([2, 2]) + 2.0, gettype(N, Es))) # A = [[2.0, 2.0], [2.0, 2.0]]
B = tobin(astype(np.zeros([2, 2]) + 2.0, gettype(N, Es))) # B = [[2.0, 2.0], [2.0, 2.0]]

Y = astype(frombin(accel.GEMM(A, B.T), gettype(N, Es)), np.double) # Y = [[8.0, 8.0], [8.0, 8.0]]
```
### HTTP-Server
```
python3 HTTPServer.py [--host <host>] [--port <port>] [--verbose] [--report_timings]
```
## Supported Configurations
Posits4TorcHA offers off-the-shelf PYNQ Overlay implementations specifically synthesized for the AMD Kria KV260 Vision Starter Kit. Currently supported configurations for the hardware accelerator are described in [Table 1](#table-1-supported-hardware-accelerator-configurations). PYNQ Overlays for hardware platforms other than the AMD Kria KV260 Vision Starter Kit can be synthesized using the Vivado Design Suite from AMD along with the project files, which include VHDL and/or Verilog descriptions for both the AXI4-Stream-compliant Posit MAC Unit wrapper and the AXI DMA-compliant 2D Posit MAC Unit Array, in conjunction with their respectives parameterized IP packages and the top-level block diagram for the overlay.
### Table 1: Supported Hardware Accelerator Configurations.
| Part Name (Part)  | Rows (R) | Columns (C) | Posit Precision (N) | Exponent Bit-length (Es) | Quire Size (QSize) | FIFO Depth (Depth) |
|-------|---|---|---|----|-------|-------|
| KV260 | 4 | 4 | 8 | 2  | 128   | 8     |
| KV260 | 4 | 4 | 8 | 2  | 128   | 16    |
| KV260 | 4 | 4 | 8 | 2  | 128   | 32    |
| KV260 | 4 | 4 | 8 | 2  | 128   | 64    |
| KV260 | 4 | 4 | 8 | 2  | 128   | 128   |
| KV260 | 4 | 4 | 8 | 2  | 128   | 256   |
| KV260 | 8 | 8 | 6 | 2  | 128   | 512   |
| KV260 | 8 | 8 | 7 | 2  | 128   | 512   |
| KV260 | 8 | 8 | 8 | 2  | 128   | 8     |
| KV260 | 8 | 8 | 8 | 2  | 128   | 16    |
| KV260 | 8 | 8 | 8 | 2  | 128   | 32    |
| KV260 | 8 | 8 | 8 | 2  | 128   | 64    |
| KV260 | 8 | 8 | 8 | 2  | 128   | 128   |
| KV260 | 8 | 8 | 8 | 2  | 128   | 256   |
| KV260 | 8 | 8 | 8 | 2  | 128   | 512   |
