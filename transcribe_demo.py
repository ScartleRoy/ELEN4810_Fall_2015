# -*- coding: utf-8 -*-
__author__ = 'Roy'

from singing_transcription import *
import os

def evaluate_path(eval_file_path='wav'):

    os.system("find " + eval_file_path + " -name '*.wav' > wav_files.txt")

    eval_file = open("wav_files.txt")

    file_names = []
    for line in eval_file:
        file_names.append(eval_file_path.strip("'")+"/"+line[:-1])

    file_num = len(file_names)
    for i in xrange(file_num):
        singing_transcription(file_names[i])
        print "Finished analyzing file " + file_names[i].split('/')[-1] + ". " + str(i+1) + " of " + str(file_num) +" finished."

    os.system("rm "+"wav_files.txt")


def evaluate_single(filename='child1.wav'):
    singing_transcription(filename)
    print "Finished analyzing file " + filename + ". "
