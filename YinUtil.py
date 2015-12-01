__author__ = 'Roy'

# The utilities needed for YIN.

from FFT import *

class YinUtil:

    @staticmethod
    def fastDifference(Input, yinBuffer, yinBufferSize):
        """
        The autocorrelation method and difference function in YIN
        input: input signal
        yinBuffer: buffer for saving the output of yin
        yinBufferSize: size of the buffer
        """
        frameSize = 2 * yinBufferSize

        for j in range(0, yinBufferSize):
            yinBuffer[j] = 0.

        audioTransformedReal = [0.]*frameSize
        audioTransformedImag = [0.]*frameSize
        nullImag = [0.]*frameSize
        kernel = [0.]*frameSize
        kernelTransformedReal = [0.]*frameSize
        kernelTransformedImag = [0.]*frameSize
        yinStyleACFReal = [0.]*frameSize
        yinStyleACFImag = [0.]*frameSize
        powerTerms = [0.]*yinBufferSize


        # POWER TERM CALCULATION
        # ... for the power terms in equation (7) in the Yin paper
        for j in range(0, yinBufferSize):
            powerTerms[0] += Input[j] * Input[j]


        # now iteratively calculate all others (saves a few multiplications)
        for tau in range(1, yinBufferSize):
            powerTerms[tau] = powerTerms[tau-1] - Input[tau-1] * Input[tau-1] + Input[tau + yinBufferSize] * Input[tau + yinBufferSize]



        # YIN-STYLE AUTOCORRELATION via FFT
        # 1. data
        y1 = FFT()
        y1.FFT(frameSize, False, Input, nullImag, audioTransformedReal, audioTransformedImag)

        # 2. half of the data, disguised as a convolution kernel

        for j in range(0, yinBufferSize):
            kernel[j] = Input[yinBufferSize-1-j]
            kernel[j+yinBufferSize] = 0.
        y2 = FFT()
        y2.FFT(frameSize, False, kernel, nullImag, kernelTransformedReal, kernelTransformedImag)

        # 3. convolution via complex multiplication -- written into

        for j in range(0, frameSize):
            yinStyleACFReal[j] = audioTransformedReal[j] * kernelTransformedReal[j] - audioTransformedImag[j] * kernelTransformedImag[j]  # real
            yinStyleACFImag[j] = audioTransformedReal[j] * kernelTransformedReal[j] + audioTransformedImag[j] * kernelTransformedImag[j]  # imaginary
        y3 = FFT()
        y3.FFT(frameSize, True, yinStyleACFReal, yinStyleACFImag, audioTransformedReal, audioTransformedImag)

        # CALCULATION OF difference function
        # ... according to (7) in the Yin paper.

        for j in range(0, yinBufferSize):
            yinBuffer[j] = powerTerms[0] + powerTerms[j] - 2 * audioTransformedReal[j + yinBufferSize - 1]

        del audioTransformedReal
        del audioTransformedImag
        del nullImag
        del kernel
        del kernelTransformedReal
        del kernelTransformedImag
        del yinStyleACFReal
        del yinStyleACFImag
        del powerTerms

    @staticmethod
    def cumulativeDifference(yinBuffer, yinBufferSize):
        yinBuffer[0] = 1.
        runningSum = 0.

        for tau in range(1, yinBufferSize):
            runningSum += yinBuffer[tau]
            if runningSum == 0:
                yinBuffer[tau] = 1.
            else:
                yinBuffer[tau] *= tau * 1.0 / runningSum

    @staticmethod
    def absoluteThreshold(yinBuffer, yinBufferSize, thresh):
        minTau = 0
        minVal = 1000.

        tau = 2
        while tau < yinBufferSize:
            if yinBuffer[tau] < thresh:
                while tau + 1 < yinBufferSize and yinBuffer[tau+1] < yinBuffer[tau]:
                    tau += 1
                return tau
            else:
                if yinBuffer[tau] < minVal:
                    minVal = yinBuffer[tau]
                    minTau = tau
            tau += 1
        if minTau > 0:
            return - minTau
        return 0

    @staticmethod
    def yinProb(yinBuffer, prior, yinBufferSize):
        minWeight = 0.01
        thresholds = []
        distribution = []
        peakProb = [0.]*yinBufferSize

        # TODO: make the distributions below part of a class, so they don't have to
        # be allocated every time

        uniformDist = [0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000, 0.0100000]
        betaDist1 = [0.028911, 0.048656, 0.061306, 0.068539, 0.071703, 0.071877, 0.069915, 0.066489, 0.062117, 0.057199, 0.052034, 0.046844, 0.041786, 0.036971, 0.032470, 0.028323, 0.024549, 0.021153, 0.018124, 0.015446, 0.013096, 0.011048, 0.009275, 0.007750, 0.006445, 0.005336, 0.004397, 0.003606, 0.002945, 0.002394, 0.001937, 0.001560, 0.001250, 0.000998, 0.000792, 0.000626, 0.000492, 0.000385, 0.000300, 0.000232, 0.000179, 0.000137, 0.000104, 0.000079, 0.000060, 0.000045, 0.000033, 0.000024, 0.000018, 0.000013, 0.000009, 0.000007, 0.000005, 0.000003, 0.000002, 0.000002, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
        betaDist2 = [0.012614, 0.022715, 0.030646, 0.036712, 0.041184, 0.044301, 0.046277, 0.047298, 0.047528, 0.047110, 0.046171, 0.044817, 0.043144, 0.041231, 0.039147, 0.036950, 0.034690, 0.032406, 0.030133, 0.027898, 0.025722, 0.023624, 0.021614, 0.019704, 0.017900, 0.016205, 0.014621, 0.013148, 0.011785, 0.010530, 0.009377, 0.008324, 0.007366, 0.006497, 0.005712, 0.005005, 0.004372, 0.003806, 0.003302, 0.002855, 0.002460, 0.002112, 0.001806, 0.001539, 0.001307, 0.001105, 0.000931, 0.000781, 0.000652, 0.000542, 0.000449, 0.000370, 0.000303, 0.000247, 0.000201, 0.000162, 0.000130, 0.000104, 0.000082, 0.000065, 0.000051, 0.000039, 0.000030, 0.000023, 0.000018, 0.000013, 0.000010, 0.000007, 0.000005, 0.000004, 0.000003, 0.000002, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
        betaDist3 = [0.006715, 0.012509, 0.017463, 0.021655, 0.025155, 0.028031, 0.030344, 0.032151, 0.033506, 0.034458, 0.035052, 0.035331, 0.035332, 0.035092, 0.034643, 0.034015, 0.033234, 0.032327, 0.031314, 0.030217, 0.029054, 0.027841, 0.026592, 0.025322, 0.024042, 0.022761, 0.021489, 0.020234, 0.019002, 0.017799, 0.016630, 0.015499, 0.014409, 0.013362, 0.012361, 0.011407, 0.010500, 0.009641, 0.008830, 0.008067, 0.007351, 0.006681, 0.006056, 0.005475, 0.004936, 0.004437, 0.003978, 0.003555, 0.003168, 0.002814, 0.002492, 0.002199, 0.001934, 0.001695, 0.001481, 0.001288, 0.001116, 0.000963, 0.000828, 0.000708, 0.000603, 0.000511, 0.000431, 0.000361, 0.000301, 0.000250, 0.000206, 0.000168, 0.000137, 0.000110, 0.000088, 0.000070, 0.000055, 0.000043, 0.000033, 0.000025, 0.000019, 0.000014, 0.000010, 0.000007, 0.000005, 0.000004, 0.000002, 0.000002, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
        betaDist4 = [0.003996, 0.007596, 0.010824, 0.013703, 0.016255, 0.018501, 0.020460, 0.022153, 0.023597, 0.024809, 0.025807, 0.026607, 0.027223, 0.027671, 0.027963, 0.028114, 0.028135, 0.028038, 0.027834, 0.027535, 0.027149, 0.026687, 0.026157, 0.025567, 0.024926, 0.024240, 0.023517, 0.022763, 0.021983, 0.021184, 0.020371, 0.019548, 0.018719, 0.017890, 0.017062, 0.016241, 0.015428, 0.014627, 0.013839, 0.013068, 0.012315, 0.011582, 0.010870, 0.010181, 0.009515, 0.008874, 0.008258, 0.007668, 0.007103, 0.006565, 0.006053, 0.005567, 0.005107, 0.004673, 0.004264, 0.003880, 0.003521, 0.003185, 0.002872, 0.002581, 0.002312, 0.002064, 0.001835, 0.001626, 0.001434, 0.001260, 0.001102, 0.000959, 0.000830, 0.000715, 0.000612, 0.000521, 0.000440, 0.000369, 0.000308, 0.000254, 0.000208, 0.000169, 0.000136, 0.000108, 0.000084, 0.000065, 0.000050, 0.000037, 0.000027, 0.000019, 0.000014, 0.000009, 0.000006, 0.000004, 0.000002, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
        single10 = [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,1.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        single15 = [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,1.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
        single20 = [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,1.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]

        nThreshold = 100
        nThresholdInt = nThreshold

        for i in range(0, nThresholdInt):
            if prior == 0:
                distribution.append(uniformDist[i])
            elif prior == 1:
                distribution.append(betaDist1[i])
            elif prior == 2:
                distribution.append(betaDist2[i])
            elif prior == 3:
                distribution.append(betaDist3[i])
            elif prior == 4:
                distribution.append(betaDist4[i])
            elif prior == 5:
                distribution.append(single10[i])
            elif prior == 6:
                distribution.append(single15[i])
            elif prior == 7:
                distribution.append(single20[i])
            else:
                distribution.append(uniformDist[i])
            thresholds.append(0.01 + i * 0.01)

        currThreshInd = nThreshold - 1
        tau = 2

        minInd = 0
        minVal = 42.
        while (not currThreshInd == -1) and tau < yinBufferSize:
            if yinBuffer[tau] < thresholds[currThreshInd]:
                while tau + 1 < yinBufferSize and yinBuffer[tau+1] < yinBuffer[tau]:
                    tau += 1
                # tau is now local minimum
                if yinBuffer[tau] < minVal and tau > 2:
                    minVal = yinBuffer[tau]
                    minInd = tau
                peakProb[tau] += distribution[currThreshInd]
                currThreshInd -= 1
            else:
                tau += 1

        nonPeakProb = 1.
        for i in range(0, yinBufferSize):
            nonPeakProb -= peakProb[i]

        if minInd > 0:
            peakProb[minInd] += nonPeakProb * minWeight

        return peakProb

    @staticmethod
    def parabolicInterpolation(yinBuffer, tau, yinBufferSize):
        if tau == yinBufferSize:  # not valid anyway
            return tau
        x0 = 0
        x2 = 0
        betterTau = 0.

        if tau < 1:
            x0 = tau
        else:
            x0 = tau - 1

        if tau + 1 < yinBufferSize:
            x2 = tau + 1
        else:
            x2 = tau

        if x0 == tau:
            if yinBuffer[tau] <= yinBuffer[x2]:
                betterTau = tau * 1.0
            else:
                betterTau = x2 * 1.0
        elif x2 == tau:
            if yinBuffer[tau] <= yinBuffer[x0]:
                betterTau = tau * 1.0
            else:
                betterTau = x0 * 1.0
        else:
            s0 = yinBuffer[x0] * 1.
            s1 = yinBuffer[tau] * 1.
            s2 = yinBuffer[x2] * 1.
            betterTau = tau * 1.0 + (s2 - s0) / (2 * ( 2 * s1 - s2 - s0))

        return betterTau

    @staticmethod
    def sumSquare(Input, start, end):
        out = 0.
        for i in range(start, end):
            out += Input[i] * Input[i]
        return out