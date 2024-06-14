import sys

import cython
from cython.parallel import prange

import pynq

import numpy as np

import time

uint_ = cython.fused_type(cython.uchar, cython.ushort, cython.uint)

@cython.nogil
@cython.cfunc
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def to_sendchannel(
        sendchannel: uint_ [::1], 
        a: uint_ [::1], 
        b: uint_ [::1], 
        sendchannel_height: cython.int, sendchannel_width: cython.int,
        a_height: cython.int, a_width: cython.int, 
        b_height: cython.int, b_width: cython.int, 
        accel_rows: cython.int, accel_columns: cython.int,
        a_width_offset: cython.int, b_width_offset: cython.int,
        num_threads: cython.int
    ) -> cython.void:
    sendchannel_row_offset: cython.int
    a_row_offset: cython.int
    b_row_offset: cython.int
    a_column_offset: cython.int
    b_column_offset: cython.int
    a_offset: cython.int
    b_offset: cython.int

    i: cython.int
    j: cython.int

    for i in prange(sendchannel_height, num_threads = num_threads):
        # Guard for 1 * sendchannel_width (1 Clock Cycle)
        sendchannel_row_offset = (i + 1) * sendchannel_width
        a_row_offset = (i % a_height) * a_width
        b_row_offset = (i % b_height) * b_width
        a_column_offset = (a_width_offset + ((b_width_offset + (i // b_height) * accel_columns) // b_width) * accel_rows) % a_width
        b_column_offset = (b_width_offset + (i // b_height) * accel_columns) % b_width

        a_offset = a_row_offset + a_column_offset
        b_offset = b_row_offset + b_column_offset

        for j in range(accel_rows):
            sendchannel[sendchannel_row_offset + j] = a[a_offset + j]

        for j in range(accel_columns):
            sendchannel[sendchannel_row_offset + j + accel_rows] = b[b_offset + j]

@cython.nogil
@cython.cfunc
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def from_recvchannel(
        recvchannel: uint_ [::1],
        a: uint_ [:, ::1],
        recvchannel_height: cython.int, recvchannel_width: cython.int,
        a_height: cython.int, a_width: cython.int,
        accel_rows: cython.int, accel_columns: cython.int,
        a_height_offset: cython.int, a_width_offset: cython.int) -> cython.void:
    i: cython.int
    j: cython.int
    k: cython.int
    recvchannel_row_offset: cython.int
    recvchannel_column_offset: cython.int
    a_row: cython.int
    a_column: cython.int

    # Guard for 1 * recvchannel_width (1 Clock Cycle)
    for k in range(1, recvchannel_height):
        recvchannel_row_offset = k * recvchannel_width

        for i in range(accel_rows):
            for j in range(accel_columns):
                recvchannel_column_offset = i * accel_columns + j

                a_column = j + a_width_offset + (k - 1) * accel_columns
                a_row = i + a_height_offset + (a_column // a_width) * accel_rows
                a_column %= a_width
                a_row %= a_height

                a[a_row, a_column] = recvchannel[recvchannel_row_offset + recvchannel_column_offset]

class Accel:
    def __init__(self, Part = 'KV260', R = 8, C = 8, N = 8, Es = 2, QSize = 128, Depth = 8, num_threads = 1, verbose = True, report_timings = True):
        super().__init__()

        self.verbose = verbose
        self.report_timings = report_timings

        self._overlay = None
        self._axi_dma_0 = None
        self._axi_gpio_0 = None

        self.Part = None
        self.R = None
        self.C = None
        self.N = None
        self.Es = None
        self.QSize = None
        self.Depth = None

        self.num_threads = None

        self.load(Part, R, C, N, Es, QSize, Depth, num_threads)


    def load(self, Part, R, C, N, Es, QSize, Depth, num_threads, force = False):
        params = [self.Part, self.R, self.C, self.N, self.Es, self.QSize, self.Depth, self.num_threads]
        args = [Part, R, C, N, Es, QSize, Depth, num_threads]

        if not force:
            if (all([args[i] == params[i] for i in range(len(args))])):
                return

        if self.verbose:
            print("Downloading Bitstream...")

        pynq.PL.reset()

        self._overlay = pynq.Overlay("data/{}_{}_{}_{}_{}_{}_{}.bit".format(Part, R, C, N, Es, QSize, Depth))

        self._axi_dma_0 = self._overlay.axi_dma_0 

        self._axi_gpio_0 = self._overlay.axi_gpio_0

        if self.verbose:
            print("Downloading Bitstream... Completed.")

        self.Part = Part
        self.R = R
        self.C = C
        self.N = N
        self.Es = Es
        self.QSize = QSize
        self.Depth = Depth

        self.num_threads = num_threads

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    def GEMM(self, A: uint_[:, :], B: uint_[:, :]):
        if A.shape[1] != B.shape[0]:
            raise RuntimeError("Matrix Inner-Dimensions Mismatch ({} and {}).".format(A.shape, B.shape)) 

        N2P2 = int(2**np.ceil(np.log2(self.N)))

        if 0 < N2P2 and N2P2 <= 8:
            dtype = np.uint8
        elif 8 < N2P2 and N2P2 <= 16:
            dtype = np.uint16
        elif 16 < N2P2 and N2P2 <= 32:
            dtype = np.uint32
        else:
            raise RuntimeError("Non-Supported Precision ({}-bits).".format(self.N))

        if uint_ is cython.uchar:
            assert dtype is np.uint8
        elif uint_ is cython.ushort:
            assert dtype is np.uint16
        else:
            assert dtype is np.uint32

        r = int(np.ceil(A.shape[0] / self.R))
        c = int(np.ceil(B.shape[1] / self.C))
        r = int(np.ceil(r * c / self.Depth)) * self.Depth // c

        Yr = r * self.R
        Yc = c * self.C
        _Yr: cython.int = Yr
        _Yc: cython.int = Yc

        _A = np.pad(A, ((0, Yr - A.shape[0]), (0, 0))).T
        _B = np.pad(B, ((0, 0), (0, Yc - B.shape[1])))

        _Ar: cython.int = _A.shape[0]
        _Ac: cython.int = _A.shape[1]
        _Br: cython.int = _B.shape[0]
        _Bc: cython.int = _B.shape[1]

        _A = _A.reshape(_Ar * _Ac)
        _B = _B.reshape(_Br * _Bc)
        _A = np.ndarray.copy(_A)
        _B = np.ndarray.copy(_B)

        _A_view: uint_[::1] = _A
        _B_view: uint_[::1] = _B

        _Y = np.zeros([Yr, Yc], dtype = dtype)
        _Y_view: uint_[:, ::1] = _Y

        Depth: cython.Py_ssize_t = self.Depth

        sendchannel_height: cython.int = self.Depth * _Ar + 1
        sendchannel_width: cython.int = 2 ** int(np.ceil(np.log2(self.R + self.C)))
        recvchannel_height: cython.int = self.Depth + 1
        recvchannel_width: cython.int = 2 ** int(np.ceil(np.log2(self.R * self.C)))

        to_HA = pynq.allocate(shape = (sendchannel_height * sendchannel_width), dtype = dtype)
        from_HA = pynq.allocate(shape = (recvchannel_height * recvchannel_width), dtype = dtype) 

        to_HA_view: uint_[::1] = to_HA
        from_HA_view: uint_[::1] = from_HA

        self._axi_gpio_0.channel1.write(_Ar, 0xFFFFFFFF)

        i: cython.Py_ssize_t
        j: cython.Py_ssize_t

        ii: cython.Py_ssize_t = r
        jj: cython.Py_ssize_t = c

        I: cython.Py_ssize_t = (ii * jj) // Depth
        J: cython.Py_ssize_t

        a_column_offset: cython.Py_ssize_t
        b_column_offset: cython.Py_ssize_t
        a_column_offset_int: cython.int
        b_column_offset_int: cython.int

        R: cython.int = self.R
        C: cython.int = self.C
        M: cython.Py_ssize_t = self.R
        N: cython.Py_ssize_t = self.C

        num_threads: cython.int = self.num_threads

        t = 0.0
        t_to_HA = 0.0
        t_Y = 0.0
        dt = 0.0
        dt_to_HA = 0.0
        dt_Y = 0.0

        now = time.time()

        for i in range(I):
            a_column_offset = ((i * Depth) // jj) * M
            b_column_offset = ((i * Depth) % jj) * N
            a_column_offset_int = a_column_offset
            b_column_offset_int = b_column_offset

            t_to_HA = time.time()

            to_sendchannel(to_HA_view, _A_view, _B_view, sendchannel_height, sendchannel_width, _Ar, _Ac, _Br, _Bc, R, C, a_column_offset_int, b_column_offset_int, num_threads)

            dt_to_HA += time.time() - t_to_HA

            t = time.time() 

            self._axi_dma_0.sendchannel.transfer(to_HA)
            self._axi_dma_0.sendchannel.wait()

            self._axi_dma_0.recvchannel.transfer(from_HA)
            self._axi_dma_0.recvchannel.wait()

            dt += time.time() - t

            t_Y = time.time()

            from_recvchannel(from_HA_view, _Y_view, recvchannel_height, recvchannel_width, _Yr, _Yc, R, C, a_column_offset_int, b_column_offset_int)

            dt_Y += time.time() - t_Y

        if self.report_timings:
            print("Total to_HA (s): {}".format(dt_to_HA))
            print("Total Y (s): {}".format(dt_Y))
            print("Total Field-Programmable Gate Array (FPGA) Time (s): {}".format(dt))
            print("Average Field-Programmable Gate Array (FPGA) Time (s): {}".format(dt / ii / jj))
            print("Total Time (s): {}".format(time.time() - now)) 

        Y = _Y[0:A.shape[0], 0:B.shape[1]]

        return Y
