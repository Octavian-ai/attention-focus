from statistics import *

def get_stats(seq):
    seq = list(seq)
    return {
        'mean': mean(seq),
        'harmonic_mean':harmonic_mean(seq),
        'median':median(seq),
        'median_low': median_low(seq),
        'median_high': median_high(seq),
        'mode': mode(seq),
        'max':max(seq)
    }