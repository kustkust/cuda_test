import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from numpy import int8, int32, float32
from pycuda.autoinit import context

from pycuda.compiler import SourceModule


def fful(shape, val=0):
    return np.full(shape, val, dtype=float32)


def farr(arr):
    return np.array(arr, dtype=float32)


def get_size(m):
    return [int32(m.shape[0]), int32(m.shape[1])]


def make_arr(shape):
    h_a = fful(shape)
    d_a = drv.mem_alloc(h_a.nbytes)  # type: ignore
    drv.memcpy_htod(d_a, h_a)  # type: ignore
    return h_a, d_a


def copy_arr(h_a):
    h_a = farr(h_a)
    d_a = drv.mem_alloc(h_a.nbytes)  # type: ignore
    drv.memcpy_htod(d_a, h_a)  # type: ignore
    return d_a


def trans_and_cop(h: np.ndarray):
    shape = h.T.shape
    return h.T.flatten().reshape(shape)


def sync(): context.synchronize()  # type: ignore


class QR_CGS:
    def __init__(self):
        with open("GPU_FUNC.cu") as GPU_FUNC:
            self.mod = SourceModule(GPU_FUNC.read())
        self.fn = {}
        self.add_func("printVectorKernel")
        self.add_func("printMatrixKernel")
        self.add_func("printMatTKernel")
        self.add_func("printMatFlatKernel")
        self.add_func("getVectorKernel")
        self.add_func("matrixTransposeKernel")

        self.add_func("matrixMultGPU")

        self.add_func("calculateProjectionGPU")
        self.add_func("innerProductGPU")
        self.add_func("sumProjectionsGPU")
        self.add_func("vectorSubGPU")
        self.add_func("vectorNormsGPU")
        self.add_func("normMultGPU")

        self.speck = {"block": (1, 1, 1), "grid": (1,)}

    def add_func(self, name):
        self.fn[name] = self.mod.get_function(name)

    def printVectorKernel(self, v, sync_b=True):
        self.fn["printVectorKernel"](v, *self.shape,  **self.speck)
        if sync_b:
            sync()

    def printMatrixKernel(self, v, sync_b=True):
        self.fn["printMatrixKernel"](
            v, *self.shape,  block=(1, 1, 1), grid=(1,))
        if sync_b:
            sync()

    def printMatTKernel(self, v, sync_b=True):
        self.fn["printMatTKernel"](v, *self.shape,  block=(1, 1, 1), grid=(1,))
        if sync_b:
            sync()

    def printMatFlatKernel(self, v, sync_b=True):
        self.fn["printMatFlatKernel"](
            v, *self.shape,  block=(1, 1, 1), grid=(1,))
        if sync_b:
            sync()

    def getVectorKernel(self, v, V_t, rowNum, reverse):
        self.fn["getVectorKernel"](v, V_t, int32(
            rowNum), int8(reverse), *self.shape, **self.speck)

    def matrixTransposeKernel(self, V, V_t, reverse):
        self.fn["matrixTransposeKernel"](
            V, V_t, int8(reverse), *self.shape, **self.speck)

    def matrixMultGPU(self, Q_t, A, R):
        self.fn["matrixMultGPU"](Q_t, A, R, *self.shape, **self.speck)

    def calculateProjectionGPU(self, u, upper, lower, p):
        self.fn["calculateProjectionGPU"](
            u, upper, lower, p, *self.shape, **self.speck, shared=int(self.shape[0]*4))

    def innerProductGPU(self, a, b, c):
        self.fn["innerProductGPU"](
            a, b, c, *self.shape, **self.speck, shared=int(self.shape[0]*4))

    def sumProjectionsGPU(self, P_t, projSum):
        self.fn["sumProjectionsGPU"](P_t, projSum, *self.shape, **self.speck)

    def vectorSubGPU(self, v, projSum, u):
        self.fn["vectorSubGPU"](v, projSum, u, *self.shape, **self.speck)

    def vectorNormsGPU(self, U_t, norms):
        self.fn["vectorNormsGPU"](U_t, norms, *self.shape, **self.speck)

    def normMultGPU(self, U, norms, E):
        self.fn["normMultGPU"](U, norms, E, *self.shape, **self.speck)

    def runCGS(self, h_V: np.ndarray) -> np.ndarray:
        # h_V = h_V.T
        self.shape = get_size(h_V)
        shape = self.shape
        M, N = shape

        self.speck["block"] = (32, 1, 1)
        self.speck["grid"] = (1,)

        h_v, d_v = make_arr(shape)
        d_V = copy_arr(h_V)
        d_Vt = copy_arr(h_Vt := trans_and_cop(h_V))
        h_u, d_u = make_arr(M)
        h_Ut = fful([shape[1], shape[0]])
        h_Ut[0][:] = h_Vt[0][:]
        d_Ut = copy_arr(h_Ut)
        d_U = copy_arr(h_U := trans_and_cop(h_Ut))

        h_Upper, d_Upper = make_arr(1)
        h_Lower, d_Lower = make_arr(1)

        h_p, d_p = make_arr(M)
        h_Pt, d_Pt = make_arr(shape)
        h_PS, d_PS = make_arr(M)

        h_N, d_N = make_arr(N)
        h_E, d_E = make_arr(shape)

        # print(h_V)
        # self.printMatrixKernel(d_V)
        # print(h_Vt)
        # self.printMatTKernel(d_Vt)
        # print(h_U)
        # self.printMatrixKernel(d_U)
        # print(h_Ut)
        # self.printMatTKernel(d_Ut)
        # self.printMatFlatKernel(d_Ut)
        # return

        for i in range(1, N):
            self.getVectorKernel(d_v, d_Vt, i, False)
            sync()
            for j in range(0, N):
                self.getVectorKernel(d_u, d_Ut, j, False)
                sync()
                self.innerProductGPU(d_u, d_v, d_Upper)
                self.innerProductGPU(d_u, d_u, d_Lower)
                sync()
                self.calculateProjectionGPU(d_u, d_Upper, d_Lower, d_p)
                sync()
                self.getVectorKernel(d_p, d_Pt, j, True)
                sync()
            self.sumProjectionsGPU(d_Pt, d_PS)
            sync()
            self.vectorSubGPU(d_v, d_PS, d_u)
            sync()
            self.getVectorKernel(d_u, d_Ut, i, True)
            sync()
            self.matrixTransposeKernel(d_U, d_Ut, True)
            sync()
        self.vectorNormsGPU(d_Ut, d_N)
        sync()
        self.normMultGPU(d_U, d_N, d_E)
        sync()

        drv.memcpy_dtoh(h_E, d_E)  # type: ignore
        sync()

        d_v.free()
        d_V.free()
        d_Vt.free()
        d_u.free()
        d_U.free()
        d_Ut.free()
        d_Upper.free()
        d_Lower.free()
        d_Pt.free()
        d_PS.free()
        d_N.free()
        d_E.free()

        print(h_E)
        return h_E

    def runQR(self, h_A, h_V):
        self.shape = get_size(h_V)
        shape = self.shape
        M, N = shape
        h_Q = self.runCGS(h_V)
        d_Qt = copy_arr(h_Qt := trans_and_cop(h_Q))
        d_A = copy_arr(h_A)
        h_R, d_R = make_arr([N, N])
        self.speck["block"] = (32, 32, 1)
        self.speck["grid"] = (1,)

        self.matrixMultGPU(d_Qt, d_A, d_R)

        drv.memcpy_dtoh(h_R, d_R)  # type: ignore
        print(h_R)


qr_cgs = QR_CGS()

# qr_cgs.runCGS(farr([[1, -1, 4], [1, 4, -2], [1, 4, 2], [1, -1, 0]]))
qr_cgs.runQR(farr([[1, -1, 4], [1, 4, -2], [1, 4, 2], [1, -1, 0]]),
             farr([[1, -1, 4], [1, 4, -2], [1, 4, 2], [1, -1, 0]]))
