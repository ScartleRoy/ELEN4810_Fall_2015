import numpy as np
import string
import os
import copy
import warnings
warnings.filterwarnings('ignore')
import librosa
from necessity import *


def phrase_segment(MIDI):
    # decide all the phrase onset and offsets
    # if clean chroma is completely different, an onset
    first_frame = np.where(MIDI > 10.)[0][0]
    last_frame = np.where(MIDI > 10.)[0][-1]
    onset = [first_frame]
    offset = []

    cnt = first_frame+1
    while cnt < last_frame:
        if MIDI[cnt] < 10.:
            # when find a pitch that is too small
            offset.append(cnt-1)
            cnt += 1
            # continue searching until pass all the zero pitches
            while MIDI[cnt] < 10. and cnt < last_frame:
                cnt += 1
            # add an onset
            onset.append(cnt)
            cnt += 1
        else:
            cnt += 1


    if len(onset) > len(offset):
        offset.append(last_frame)

    # adjust the position of phrase endpoints

    for i in xrange(len(onset)):
        # search through onset, until the consecutive pitch change is less than a semitone
        correct_onset = onset[i]
        while abs(MIDI[correct_onset] - MIDI[correct_onset+1]) > 1.:
            correct_onset += 1
        onset[i] = correct_onset
        # same for offset
        correct_offset = offset[i]
        while abs(MIDI[correct_offset] - MIDI[correct_offset-1]) > 2.:
            correct_offset -= 1
        offset[i] = correct_offset

    # detect phrases inside a phrase
    new_onset = []
    new_offset = []
    for i in xrange(len(onset)):
        cnt = onset[i]+3
        while cnt < offset[i]-3:
            # if there is a gap
            # find a local minimum or maximum with pitch difference larger than 2 semitones
            if MIDI[cnt-2] <=  MIDI[cnt] and MIDI[cnt] >=  MIDI[cnt+2] and \
            max(MIDI[cnt]-MIDI[cnt-2], MIDI[cnt]-MIDI[cnt+2]) > 2.:
                # add an offset and an onset
                new_offset.append(cnt-2)
                new_onset.append(cnt+2)
                cnt += 4
            elif MIDI[cnt-2] >= MIDI[cnt] and MIDI[cnt] <= MIDI[cnt+2] and \
            max(MIDI[cnt-2]-MIDI[cnt], MIDI[cnt+2]-MIDI[cnt]) > 2.:
                # add an offset and an onset
                new_offset.append(cnt-2)
                new_onset.append(cnt+2)
                cnt += 4
            else:
                cnt += 1

    if len(new_onset) > 0:
        # add the new phrase back
        for i in xrange(len(new_onset)):
            pos = np.where(np.array(onset) < new_onset[i])[0][-1]
            onset.insert(pos+1, new_onset[i])
            pos = np.where(np.array(offset) > new_offset[i])[0][0]
            offset.insert(pos, new_offset[i])

    return onset, offset


def onset_detection(phrase_onset, phrase_offset, chroma):
    onset = []
    offset = []
    for i in xrange(len(phrase_onset)):
        onset.append([])
        offset.append([])
        current_onset = phrase_onset[i]
        current_offset = phrase_offset[i]
        onset[-1].append(current_onset)

        # check chroma
        cnt = current_onset + 1
        while cnt < current_offset-1:
            if not np.array_equal(chroma[:, cnt], chroma[:, cnt-1]):
                # an offset
                offset[-1].append(cnt)
                # add a new onset
                onset[-1].append(cnt+1)
                cnt += 1
            else:
                cnt += 1
        offset[-1].append(current_offset)

    return onset, offset


def pitch_based_combine(phrase_onset, phrase_offset, onset, offset, note_pitch, cent, full_pitch):
    for i in xrange(len(phrase_onset)):
        current_onset = phrase_onset[i]
        current_offset = phrase_offset[i]
        current_pitch = note_pitch[i]
        # calculate the pitch and compare
        # if there is only one note in a phrase, do nothing
        # if there are only two notes in a phrase
        if len(onset[i]) == 2:
            # check the note pitch
            if np.abs(current_pitch[0] - current_pitch[1]) < 45.:
                # combine
                offset[i][0] = offset[i][1]
                note_pitch[i][0] = freq2cent(histogram_mean(onset[i][0], offset[i][1], full_pitch))
                del onset[i][1]
                del offset[i][1]
                del note_pitch[i][1]
        # if there are more than 2 notes in a phrase
        elif len(onset[i]) >= 3:
            # for a note in the middle, compare with left and right
            idx = 1
            while idx < len(onset[i])-1:
                # compare the pitch
                # if left is closer
                if np.abs(current_pitch[idx-1] - current_pitch[idx]) < 45. and \
                np.abs(current_pitch[idx-1] - current_pitch[idx]) < \
                np.abs(current_pitch[idx+1] - current_pitch[idx]):
                    # combine with left
                    offset[i][idx-1] = offset[i][idx]
                    note_pitch[i][idx-1] = freq2cent(histogram_mean(onset[i][idx-1], offset[i][idx], full_pitch))
                    del onset[i][idx]
                    del offset[i][idx]
                    del note_pitch[i][idx]

                # else if right is closer
                elif np.abs(current_pitch[idx+1] - current_pitch[idx]) < 45. and \
                np.abs(current_pitch[idx+1] - current_pitch[idx]) < \
                np.abs(current_pitch[idx-1] - current_pitch[idx]):
                    # combine with right
                    offset[i][idx] = offset[i][idx+1]
                    note_pitch[i][idx] = freq2cent(histogram_mean(onset[i][idx], offset[i][idx+1], full_pitch))
                    del onset[i][idx+1]
                    del offset[i][idx+1]
                    del note_pitch[i][idx+1]
                    idx += 1

                # else if nor of them are close, check the length of the note
                else:
                    # if length too small
                    if offset[i][idx] - onset[i][idx] < 18:
                        # delete
                        del onset[i][idx]
                        del offset[i][idx]
                        del note_pitch[i][idx]
                    else:
                        idx += 1

    return onset, offset, note_pitch


def length_based_combine(phrase_onset, phrase_offset, onset, offset, note_pitch, cent, full_pitch):
    for i in xrange(len(phrase_onset)):
        current_onset = phrase_onset[i]
        current_offset = phrase_offset[i]
        current_pitch = note_pitch[i]
        # calculate the pitch and compare
        # if there is only one note in a phrase, if it's too short, delete
        if len(onset[i]) == 1:
            if offset[i][0] - onset[i][0] < 18:
                # delete
                del onset[i][0]
                del offset[i][0]
                del note_pitch[i][0]
        # if there are only two notes in a phrase
        elif len(onset[i]) == 2:
            # if any of them has length smaller than 18
            if min(offset[i][0] - onset[i][0], offset[i][1] - onset[i][1]) < 18:
                offset[i][0] = offset[i][1]
                note_pitch[i][0] = freq2cent(histogram_mean(onset[i][0], offset[i][1], full_pitch))
                del onset[i][1]
                del offset[i][1]
                del note_pitch[i][1]
        # if there are more than 2 notes in a phrase
        elif len(onset[i]) >= 3:
            # for a note in the middle, compare with left and right
            idx = 1
            while idx < len(onset[i])-1:
                if offset[i][idx] - onset[i][idx] < 25:
                    # compare length
                    # if neighboring notes has length also less than 18
                    # combine with the one has smaller length

                    # left is smaller than 18 and left <= right
                    if offset[i][idx-1] - onset[i][idx-1] < 25 and \
                    offset[i][idx-1] - onset[i][idx-1] <= offset[i][idx+1] - onset[i][idx+1]:
                        # combine with left
                        offset[i][idx-1] = offset[i][idx]
                        note_pitch[i][idx-1] = freq2cent(histogram_mean(onset[i][idx-1], offset[i][idx], full_pitch))
                        del onset[i][idx]
                        del offset[i][idx]
                        del note_pitch[i][idx]

                    # right is smaller than 18 and left > right
                    elif offset[i][idx+1] - onset[i][idx+1] < 25 and \
                    offset[i][idx-1] - onset[i][idx-1] > offset[i][idx+1] - onset[i][idx+1]:
                        # combine with right
                        offset[i][idx] = offset[i][idx+1]
                        note_pitch[i][idx] = freq2cent(histogram_mean(onset[i][idx], offset[i][idx+1], full_pitch))
                        del onset[i][idx+1]
                        del offset[i][idx+1]
                        del note_pitch[i][idx+1]
                        idx += 1

                    # left and right are all larger than 18
                    elif offset[i][idx+1] - onset[i][idx+1] >= 25 and \
                    offset[i][idx-1] - onset[i][idx-1] >= 25:
                        # compare the pitch
                        # if left is closer
                        if np.abs(current_pitch[idx-1] - current_pitch[idx]) < \
                        np.abs(current_pitch[idx+1] - current_pitch[idx]):
                            # combine with left
                            offset[i][idx-1] = offset[i][idx]
                            note_pitch[i][idx-1] = freq2cent(histogram_mean(onset[i][idx-1], offset[i][idx], full_pitch))
                            del onset[i][idx]
                            del offset[i][idx]
                            del note_pitch[i][idx]
                        # else if right is closer
                        else:
                            # combine with right
                            offset[i][idx] = offset[i][idx+1]
                            note_pitch[i][idx] = freq2cent(histogram_mean(onset[i][idx], offset[i][idx+1], full_pitch))
                            del onset[i][idx+1]
                            del offset[i][idx+1]
                            del note_pitch[i][idx+1]
                            idx += 1


                else:
                    idx += 1

            # consider about the note at the beginning and the end of a phrase
            # because length comparation is not symmetric, while pitch comparation is symmetric

            # first note
            # if there are more than 1 notes left:
            if len(onset[i]) > 1:
                if offset[i][0] - onset[i][0] < 25:
                    # combine, no matter what the pitch difference is
                    offset[i][0] = offset[i][1]
                    note_pitch[i][0] = freq2cent(histogram_mean(onset[i][0], offset[i][1], full_pitch))
                    del onset[i][1]
                    del offset[i][1]
                    del note_pitch[i][1]

            # last note
            # if there are more than 1 notes left:
            if len(onset[i]) > 1:
                if offset[i][-1] - onset[i][-1] < 25:
                    # combine, no matter what the pitch difference is
                    offset[i][-2] = offset[i][-1]
                    note_pitch[i][-2] = freq2cent(histogram_mean(onset[i][-2], offset[i][-1], full_pitch))
                    del onset[i][-1]
                    del offset[i][-1]
                    del note_pitch[i][-1]


    return onset, offset, note_pitch

def legato(phrase_onset, phrase_offset, onset, offset, cent, note_pitch):
    for i in xrange(len(phrase_onset)):
        current_onset = phrase_onset[i]
        current_offset = phrase_offset[i]
        idx = 1
        while idx < len(onset[i])-1:
            current_pitch = cent[onset[i][idx]:offset[i][idx]+1]
            if max(current_pitch) - min(current_pitch) > 100. and \
            (strictly_increasing(current_pitch) or strictly_decreasing(current_pitch)):
                del onset[i][idx]
                del offset[i][idx]
                del note_pitch[i][idx]
            else:
                idx += 1

    return onset, offset, note_pitch


def grace_note_detect(phrase_onset, phrase_offset, onset, offset, note_pitch):
    for i in xrange(len(phrase_onset)):
        current_onset = phrase_onset[i]
        current_offset = phrase_offset[i]
        idx = 0
        while idx < len(onset[i])-1:
            # distance between offset and onset is very small
            # length of first note is no larger than 1/4 of the length of
            # the second note
            # pitch difference is 70-230 cents
            if onset[i][idx+1] - offset[i][idx] < 3 and \
            (offset[i][idx]-onset[i][idx]) < (offset[i][idx+1]-onset[i][idx-1])/4. and \
            70. < note_pitch[i][idx+1] - note_pitch[i][idx] < 230.:
                # combine
                offset[i][idx] = offset[i][idx+1]
                note_pitch[i][idx] = note_pitch[i][idx+1]
                del onset[i][idx+1]
                del offset[i][idx+1]
                del note_pitch[i][idx+1]
            else:
                idx += 1
    return onset, offset, note_pitch


def onset_offset_adjust(phrase_onset, phrase_offset, onset, offset, cent, note_pitch):
    for i in xrange(len(phrase_onset)):
        current_onset = phrase_onset[i]
        current_offset = phrase_offset[i]
        idx = 0
        while idx < len(onset[i])-1:
            if onset[i][idx+1] - offset[i][idx] < 3:
                # adjust the onset and offset
                last_match = offset[i][idx]
                first_match = onset[i][idx+1]

                # for the offset, search back
                for off in xrange(offset[i][idx], onset[i][idx], -1):
                    if abs(cent[off] - note_pitch[i][idx]) < 80.:
                        last_match = off
                        break
                # for the onset, search forward
                for on in xrange(onset[i][idx+1], offset[i][idx+1]):
                    if abs(cent[on] - note_pitch[i][idx+1]) < 20.:
                        first_match = on
                        break
                # set new onset and offset
                offset[i][idx] = last_match
                onset[i][idx+1] = first_match
                #offset[i][idx] = int((last_match + first_match) / 2)
                #onset[i][idx+1] = int((last_match + first_match) / 2)+1
            idx += 1
    return onset, offset



def singing_transcription(file_name):
    if not os.path.exists("evaluation/pitch"):
        os.makedirs("evaluation/pitch")
    if not os.path.exists("evaluation/transcription"):
        os.makedirs("evaluation/transcription")

    if not os.path.isfile('evaluation/pitch/'+file_name.split('.')[0].split('/')[-1]+'.txt'):
        os.system("$HOME/Library/Audio/Plug-Ins/Vamp/vamp-simple-host pyin:pyin:smoothedpitchtrack '"+
                  file_name+"' -o 'evaluation/pitch/"+file_name.split('.')[0].split('/')[-1]+".txt'")

    signal, fs = librosa.load(file_name, sr=44100)

    time = []
    val = []

    # read the pitch file
    test = open('evaluation/pitch/'+file_name.split('.')[0].split('/')[-1]+'.txt')
    for line in test:
        time.append(string.atof(line.split(':')[0]))
        val.append(string.atof(line.split(':')[1]))

    test.close()

    # assign pitch value 1e-3 to all unvoiced frames
    # don't assign 0

    full_pitch = []
    full_time = []
    ovlp = 256
    step_size = 1./44100*ovlp
    max_frame = len(signal) / ovlp + 1
    cnt = 0

    for i in range(max_frame):
        full_time.append(step_size * i)
        if cnt < len(val)-1 and time[cnt] - step_size * i < 0.00001:
            cnt += 1
            full_pitch.append(val[cnt])
        else:
            full_pitch.append(1e-3)

    original_pitch = full_pitch

    # Mean Filter
    # 5 points

    smooth_pitch = full_pitch[0:2]
    for i in range(2, len(full_pitch)-2):
        smooth_pitch.append(np.mean(full_pitch[i-2:i+3]))
    smooth_pitch.append(full_pitch[-2])
    smooth_pitch.append(full_pitch[-1])

    full_pitch = smooth_pitch

    signal, fs = librosa.load(file_name, sr=44100)

    S = np.abs(librosa.stft(signal, n_fft=2048, hop_length=256))
    chroma_stft = librosa.feature.chroma_stft(S=S, sr=fs)

    MIDI = 69 + 12*np.log2(np.array(full_pitch)/440.)
    MIDI[MIDI < 11] = 0.

    chroma_clear = copy.deepcopy(chroma_stft)

    for i in xrange(max_frame):
        if MIDI[i] == 0.:
            chroma_clear[:, i] = np.zeros(12)
        else:
            frame_chroma = np.zeros(12)
            frame_chroma[int(np.rint(MIDI[i])%12)] = 1.
            chroma_clear[:, i] = frame_chroma


    # a bottom-up scheme
    # first choose many possible onsets
    # then delete some of them

    MIDI = np.array(MIDI)
    full_pitch = np.array(full_pitch)
    cent = freq2cent(full_pitch)

    phrase_onset, phrase_offset = phrase_segment(MIDI)

    onset, offset = onset_detection(phrase_onset, phrase_offset, chroma_clear)
    note_pitch = []
    for i in xrange(len(onset)):
        note_pitch.append(note_pitch_calculation(onset[i], offset[i], full_pitch))

    # now there are many candidates for onset and offset
    # need to delete or combine some of them

    # first, combine notes that are too short
    # unless it's the only note in a phrase

    flat_onset = [on for sublist in onset for on in sublist]
    previous_length = 0
    while previous_length != len(flat_onset):
        previous_length = len(flat_onset)
        onset, offset, note_pitch = length_based_combine(phrase_onset, phrase_offset, onset, offset, note_pitch, cent, full_pitch)
        flat_onset = [on for sublist in onset for on in sublist]

    # next, combine the frames that have very similar pitch
    # to combine, pitch difference should be less than 50 cents

    flat_onset = [on for sublist in onset for on in sublist]
    previous_length = 0
    while previous_length != len(flat_onset):
        previous_length = len(flat_onset)
        onset, offset, note_pitch = pitch_based_combine(phrase_onset, phrase_offset, onset, offset, note_pitch, cent, full_pitch)
        flat_onset = [on for sublist in onset for on in sublist]

    # now every two consecutive note have different pitch and have length larger than 100ms
    # which is a 'steady state'

    # next we need to deal with legatos
    # notes that has monotonic increasing or decreasing frame pitch should be delete
    # legato should have pitch deviation larger than a semitone

    onset, offset, note_pitch = legato(phrase_onset, phrase_offset, onset, offset, cent, note_pitch)

    # last we detect grace note
    # if a note happen before a longer note, and pitch is 70-230 cent lower, and length is less than 100ms,
    # it is a grace note and need to combine

    onset, offset, note_pitch = grace_note_detect(phrase_onset, phrase_offset, onset, offset, note_pitch)

    # finally adjust the onset and offset
    onset, offset = onset_offset_adjust(phrase_onset, phrase_offset, onset, offset, cent, note_pitch)

    # output
    list_onset = [on for sublist in onset for on in sublist]
    list_offset = [off for sublist in offset for off in sublist]

    list_note_pitch = [pitch for sublist in note_pitch for pitch in sublist]

    MIDI_note = cent2freq(list_note_pitch)
    MIDI_note = 69 + 12*np.log2(np.array(MIDI_note)/440.)

    final_onset = np.array(full_time)[list_onset]
    final_offset = np.array(full_time)[list_offset]

    # output to file
    file_out = open('evaluation/transcription/'+file_name.split('.')[0].split('/')[-1]+".txt", 'w')

    for i in range(len(final_onset)):
        print >> file_out, "%.3f %.3f %.3f" % (final_onset[i], final_offset[i], MIDI_note[i])
    file_out.close()