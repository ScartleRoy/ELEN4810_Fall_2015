import numpy as np

def histogram_mean(onset_candidate, offset_candidate, full_pitch):
    # calculate the dominant pitch of a note
    # delete 5% at the beginning and the end
    total_length = offset_candidate - onset_candidate + 1
    # if length too small, calculate the mean
    if total_length < 10:
        return np.mean(full_pitch[onset_candidate:offset_candidate+1])
    current_pitch = full_pitch[onset_candidate + int(total_length * 0.05):offset_candidate + 1 - int(total_length * 0.05)]
    min_pitch = min(current_pitch)
    max_pitch = max(current_pitch)
    if max_pitch - min_pitch < 30:
        return np.mean(current_pitch)
    range_size = (max_pitch - min_pitch) / 5.
    # delete all zero pitch
    current_pitch = np.array(current_pitch)[np.array(current_pitch) > 10.]
    # 5 bins
    pitch_range0 = 0
    pitch_range1 = 0
    pitch_range2 = 0
    pitch_range3 = 0
    pitch_range4 = 0
    # count the number of frames
    for pitch in current_pitch:
        block = int((pitch - min_pitch) / range_size)
        if block == 0:
            pitch_range0 += 1
        elif block == 1:
            pitch_range1 += 1
        elif block == 2:
            pitch_range2 += 1
        elif block == 3:
            pitch_range3 += 1
        elif block == 4:
            pitch_range4 += 1
        elif block == 5:
            pitch_range4 += 1
    histogram = [pitch_range0, pitch_range1, pitch_range2, pitch_range3, pitch_range4]
    pos = histogram.index(max(histogram))
    weighted_pitch = np.mean([pitch for pitch in current_pitch if min_pitch + range_size * pos < pitch < min_pitch + range_size * (pos + 1)])
    return weighted_pitch

def freq2cent(freq):
    if type(freq) is not (list or np.ndarray):
        return 1200*np.log2(freq/(55.))
    else:
        cent = []
        for i in range(len(freq)):
            cent.append(1200*np.log2(freq[i]/(55.)))
        return cent


def cent2freq(cent):
    if type(cent) is not (list or np.ndarray):
        return 55. * np.power(2., cent/1200.)
    else:
        freq = []
        for i in range(len(cent)):
            freq.append(55. * np.power(2., cent[i]/1200.))
        return freq

def note_pitch_calculation(onset, offset, full_pitch):
    note_pitch = []
    for i in xrange(len(onset)):
        note_pitch.append(freq2cent(histogram_mean(onset[i], offset[i], full_pitch)))
    return note_pitch

def strictly_increasing(L):
    return all(x < y for x, y in zip(L, L[1:]))

def strictly_decreasing(L):
    return all(x > y for x, y in zip(L, L[1:]))