__author__ = 'Roy'

# The YIN algorithm robust to vibrato
# Based on YIN by Cheveigne et al.

from YinUtil import *
import numpy as np

class Yin:
    class YinOutput:
        def __init__(self, _f=None, _p=None, _r=None, _salience=None):
            if not _f:
                self.f0 = 0.
            else:
                self.f0 = _f * 1.0
            if not _p:
                self.periodicity = 0.
            else:
                self.periodicity = _p * 1.0
            if not _r:
                self.rms = 0.
            else:
                self.rms = _r * 1.0
            if not _salience:
                self.salience = []
            else:
                self.salience = _salience
            self.freqProb = []

    def __init__(self, frameSize, inputSampleRate, thresh = 0.2):
        self.m_frameSize = frameSize
        self.m_inputSampleRate = inputSampleRate
        self.m_thresh = thresh
        self.m_threshDistr = 2
        self.m_yinBufferSize = frameSize / 2
        if frameSize & (frameSize-1):
            print "N must be a power of two"

    def process(self, Input):
        yinBuffer = [0.] * self.m_yinBufferSize

        # calculate aperiodicity function for all periods

        YinUtil.fastDifference(Input, yinBuffer, self.m_yinBufferSize)
        YinUtil.cumulativeDifference(yinBuffer, self.m_yinBufferSize)

        tau = YinUtil.absoluteThreshold(yinBuffer, self.m_yinBufferSize, self.m_thresh)

        interpolatedTau = 0.
        aperiodicity = 0.
        f0 = 0.

        if tau:
            interpolatedTau = YinUtil.parabolicInterpolation(yinBuffer, abs(tau), self.m_yinBufferSize)
            f0 = self.m_inputSampleRate * (1.0 / interpolatedTau)
        else:
            interpolatedTau = 0.
            f0 = 0.

        rms = np.sqrt(YinUtil.sumSquare(Input, 0, self.m_yinBufferSize) / self.m_yinBufferSize)
        aperiodicity = yinBuffer[abs(tau)]

        if tau < 0:
            f0 = -f0

        yo = Yin.YinOutput(f0, 1 - aperiodicity, rms)

        for iBuf in range(0, self.m_yinBufferSize):
            yo.salience.append(1-yinBuffer[iBuf] if yinBuffer[iBuf] < 1 else 0)

        del yinBuffer
        return yo

    def processProbabilisticYin(self, Input):
        yinBuffer = [0.] * self.m_yinBufferSize

        # calculate aperiodicity function for all periods
        YinUtil.fastDifference(Input, yinBuffer, self.m_yinBufferSize)
        YinUtil.cumulativeDifference(yinBuffer, self.m_yinBufferSize)

        peakProbability = YinUtil.yinProb(yinBuffer, self.m_threshDistr, self.m_yinBufferSize)

        # calculate overall "probability" from peak probability

        probSum = 0.
        for iBin in range(0, self.m_yinBufferSize):
            probSum += peakProbability[iBin]

        yo = Yin.YinOutput(0,0,0)

        for iBuf in range(0, self.m_yinBufferSize):
            yo.salience.append(peakProbability[iBuf])
            if peakProbability[iBuf] > 0:
                currentF0 = self.m_inputSampleRate * (1.0 / YinUtil.parabolicInterpolation(yinBuffer, iBuf, self.m_yinBufferSize))
                yo.freqProb.append([currentF0, peakProbability[iBuf]])

        # print 'freProb', yo.freqProb
        del yinBuffer
        return yo

    def setThreshold(self, parameter):
        self.m_thresh = parameter
        return 0

    def setThresholdDistr(self, parameter):
        self.m_threshDistr = parameter
        return 0

    def setFrameSize(self, parameter):
        self.m_frameSize = parameter
        self.m_yinBufferSize = self.m_frameSize / 2
        return 0