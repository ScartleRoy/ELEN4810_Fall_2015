__author__ = 'Roy'

# The implementation of FFT needed by YIN.
# Different from the FFT in scipy.fftpack

import numpy as np


class FFT:
    def __init__(self):
        self.gFFTBitTable = []
        self.MaxFastBits = 16

    @staticmethod
    def IsPowerOfTwo(x):
        if x < 2:
            return False

        if x & (x - 1):
            return False

        return True

    @staticmethod
    def NumberOfBitsNeeded(PowerOfTwo):
        if PowerOfTwo < 2:
            print "Error: FFT called with size %d\n" %PowerOfTwo
            return -1
        i = 0
        while i >= 0:
            if PowerOfTwo & (1 << i):
                return i
            i += 1

    @staticmethod
    def ReverseBits(index, NumBits):
        rev = 0
        for i in range(0, NumBits):
            rev = (rev << 1) | (index & 1)
            index >>= 1
        return rev

    def InitFFT(self):
        self.gFFTBitTable = [0] * self.MaxFastBits
        len = 2
        for b in range(1, self.MaxFastBits+1):
            self.gFFTBitTable[b-1] = [0] * len
            for i in range(0, len):
                self.gFFTBitTable[b-1][i] = self.ReverseBits(i, b)
            len <<= 1


    def DeinitFFT(self):
        if self.gFFTBitTable:
            for b in range(1, self.MaxFastBits+1):
                del self.gFFTBitTable[b-1]
            del self.gFFTBitTable


    def FastReverseBits(self,i, NumBits):
        if NumBits <= self.MaxFastBits:
            return self.gFFTBitTable[NumBits - 1][i]
        else:
            return self.ReverseBits(i, NumBits)

    def FFT(self, NumSamples, Inverse, RealIn, ImagIn, RealOut, ImagOut):
        NumBits = 0  # Number of bits needed to store indices
        BlockSize = 0
        BlockEnd = 0

        angle_numerator = 2.0 * np.pi
        tr = 0.  # temp real
        ti = 0.  # temp imaginary

        if not self.IsPowerOfTwo(NumSamples):
            print "%d is not a power of two\n" %NumSamples
            return -1

        if not self.gFFTBitTable:
            self.InitFFT()

        if not Inverse:
            angle_numerator = -angle_numerator

        NumBits = self.NumberOfBitsNeeded(NumSamples)

        # Do simultaneous data copy and bit-reversal ordering into outputs...

        for i in range(0, NumSamples):
            j = self.FastReverseBits(i, NumBits)
            RealOut[j] = RealIn[i]
            ImagOut[j] = 0.0 if not ImagIn else ImagIn[i]

        # Do the FFT itself...

        BlockEnd = 1
        BlockSize = 2
        while BlockSize <= NumSamples:
            delta_angle = angle_numerator / (BlockSize * 1.0)
            sm2 = np.sin(-2 * delta_angle)
            sm1 = np.sin(-delta_angle)
            cm2 = np.cos(-2 * delta_angle)
            cm1 = np.cos(-delta_angle)
            w = 2 * cm1

            for i in range(0, NumSamples, BlockSize):
                ar2 = cm2
                ar1 = cm1
                ai2 = sm2
                ai1 = sm1

                j = i
                for n in range(0, BlockEnd):
                    ar0 = w * ar1 - ar2
                    ar2 = ar1
                    ar1 = ar0
                    ai0 = w * ai1 - ai2
                    ai2 = ai1
                    ai1 = ai0

                    k = j + BlockEnd
                    tr = ar0 * RealOut[k] - ai0 * ImagOut[k]
                    ti = ar0 * ImagOut[k] + ai0 * RealOut[k]

                    RealOut[k] = RealOut[j] - tr
                    ImagOut[k] = ImagOut[j] - ti

                    RealOut[j] += tr
                    ImagOut[j] += ti
                    j += 1

            BlockEnd = BlockSize
            BlockSize <<= 1

        # Need to normalize if inverse transform...

        if Inverse:
            denom = NumSamples * 1.0

            for i in range(0, NumSamples):
                RealOut[i] /= denom
                ImagOut[i] /= denom
