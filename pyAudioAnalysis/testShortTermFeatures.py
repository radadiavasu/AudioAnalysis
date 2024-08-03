import numpy as np
import math
from scipy.fftpack import fft, dct
from scipy.signal import lfilter
from scipy.io import wavfile
from tqdm import tqdm
import sys

eps = sys.float_info.epsilon

def dc_normalize(sig_array):
    """Removes DC and normalizes to -1, 1 range"""
    sig_array_norm = sig_array.copy()
    sig_array_norm -= sig_array_norm.mean()
    sig_array_norm /= abs(sig_array_norm.max).max()+ 1e-10
    return sig_array_norm

def zero_crossing_rate(frame):
    """Computes zero crossing rate of frame"""
    count = len(frame)
    count_zero = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
    return np.float64(count_zero) / np.float64(count - 1.0)

def energy(frame):
    """Computes signal energy of frame"""
    return np.sum(frame ** 2) / np.float64(len(frame))

def energy_entropy(frame, n_short_blocks=10):
    """Computes entropy of energy"""
    frame_energy = np.sum(frame ** 2)
    frame_length = len(frame)
    sub_win_len = int(np.floor(frame_length / n_short_blocks))
    if frame_length != sub_win_len * n_short_blocks:
        frame = frame[0:sub_win_len * n_short_blocks]

    sub_wins = frame.reshape(sub_win_len, n_short_blocks, order='F').copy()
    s = np.sum(sub_wins ** 2, axis=0) / (frame_energy + eps)
    entropy = -np.sum(s * np.log2(s + eps))
    return entropy

def spectral_centroid_spread(fft_magnitude, sampling_rate):
    ind = (np.arange(1, len(fft_magnitude) + 1)) * (sampling_rate / (2.0 * len(fft_magnitude)))
    Xt = fft_magnitude.copy()
    Xt_max = Xt.max()
    Xt = Xt / Xt_max if Xt_max != 0 else Xt / eps
    NUM = np.sum(ind * Xt)
    DEN = np.sum(Xt) + eps
    centroid = NUM / DEN
    spread = np.sqrt(np.sum(((ind - centroid) ** 2) * Xt) / DEN)
    centroid = centroid / (sampling_rate / 2.0)
    spread = spread / (sampling_rate / 2.0)
    return centroid, spread

def spectral_entropy(signal, n_short_blocks=10):
    num_frames = len(signal)
    total_energy = np.sum(signal ** 2)
    sub_win_len = int(np.floor(num_frames / n_short_blocks))
    if num_frames != sub_win_len * n_short_blocks:
        signal = signal[0:sub_win_len * n_short_blocks]
    sub_wins = signal.reshape(sub_win_len, n_short_blocks, order='F').copy()
    s = np.sum(sub_wins ** 2, axis=0) / (total_energy + eps)
    entropy = -np.sum(s * np.log2(s + eps))
    return entropy

def spectral_flux(fft_magnitude, previous_fft_magnitude):
    fft_sum = np.sum(fft_magnitude + eps)
    previous_fft_sum = np.sum(previous_fft_magnitude + eps)
    sp_flux = np.sum((fft_magnitude / fft_sum - previous_fft_magnitude / previous_fft_sum) ** 2)
    return sp_flux

def spectral_rolloff(signal, c):
    energy = np.sum(signal ** 2)
    fft_length = len(signal)
    threshold = c * energy
    cumulative_sum = np.cumsum(signal ** 2) + eps
    a = np.nonzero(cumulative_sum > threshold)[0]
    sp_rolloff = np.float64(a[0]) / float(fft_length) if len(a) > 0 else 0.0
    return sp_rolloff

def harmonic(frame, sampling_rate):
    m = np.round(0.016 * sampling_rate) - 1
    r = np.correlate(frame, frame, mode='full')
    g = r[len(frame) - 1]
    r = r[len(frame):-1]
    [a, ] = np.nonzero(np.diff(np.sign(r)))
    m0 = a[0] if len(a) > 0 else len(r) - 1
    if m > len(r):
        m = len(r) - 1
    gamma = np.zeros((m), dtype=np.float64)
    cumulative_sum = np.cumsum(frame ** 2)
    gamma[m0:m] = r[m0:m] / (np.sqrt((g * cumulative_sum[m:m0:-1])) + eps)
    zcr = zero_crossing_rate(gamma)
    if zcr > 0.15:
        hr, f0 = 0.0, 0.0
    else:
        hr = np.max(gamma) if len(gamma) > 0 else 1.0
        blag = np.argmax(gamma) if len(gamma) > 0 else 0.0
        f0 = sampling_rate / (blag + eps)
        if f0 > 5000 or hr < 0.1:
            f0 = 0.0
    return hr, f0

def mfcc_filter_banks(sampling_rate, num_fft, lowfreq=133.33, linc=200 / 3, logsc=1.0711703, num_lin_filt=13, num_log_filt=27):
    if sampling_rate < 8000:
        nlogfil = 5
    num_filt_total = num_lin_filt + num_log_filt
    frequencies = np.zeros(num_filt_total + 2)
    frequencies[:num_lin_filt] = lowfreq + np.arange(num_lin_filt) * linc
    frequencies[num_lin_filt:] = frequencies[num_lin_filt - 1] * logsc ** np.arange(1, num_log_filt + 3)
    heights = 2. / (frequencies[2:] - frequencies[0:-2])
    fbank = np.zeros((num_filt_total, num_fft))
    nfreqs = np.arange(num_fft) / (1. * num_fft) * sampling_rate
    for i in range(num_filt_total):
        low_freqs = frequencies[i]
        cent_freqs = frequencies[i + 1]
        high_freqs = frequencies[i + 2]
        lid = np.arange(np.floor(low_freqs * num_fft / sampling_rate) + 1, np.floor(cent_freqs * num_fft / sampling_rate) + 1, dtype=int)
        lslope = heights[i] / (cent_freqs - low_freqs)
        rid = np.arange(np.floor(cent_freqs * num_fft / sampling_rate) + 1, np.floor(high_freqs * num_fft / sampling_rate) + 1, dtype=int)
        rslope = heights[i] / (high_freqs - cent_freqs)
        fbank[i][lid] = lslope * (nfreqs[lid] - low_freqs)
        fbank[i][rid] = rslope * (high_freqs - nfreqs[rid])
    return fbank, frequencies

def mfcc(fft_magnitude, fbank, num_mfcc_feats):
    mspec = np.log10(np.dot(fft_magnitude, fbank.T) + eps)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:num_mfcc_feats]
    return ceps

def chroma_features_init(num_fft, sampling_rate):
    freqs = np.array([((f + 1) * sampling_rate) / (2 * num_fft) for f in range(num_fft)])
    cp = 27.50
    num_chroma = np.round(12.0 * np.log2(freqs / cp)).astype(int)
    num_freqs_per_chroma = np.zeros((num_chroma.shape[0],))
    unique_chroma = np.unique(num_chroma)
    for u in unique_chroma:
        idx = np.nonzero(num_chroma == u)
        num_freqs_per_chroma[idx] = idx[0].shape
    return num_chroma, num_freqs_per_chroma

def chroma_features(signal, sampling_rate, num_fft):
    num_chroma, num_freqs_per_chroma = chroma_features_init(num_fft, sampling_rate)
    chroma_names = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    spec = signal ** 2
    if num_chroma.max() < num_chroma.shape[0]:
        C = np.zeros((num_chroma.shape[0],))
        C[num_chroma] = spec
        C /= num_freqs_per_chroma[num_chroma]
    else:
        I = np.nonzero(num_chroma > num_chroma.shape[0])[0][0]
        C = np.zeros((num_chroma.shape[0],))
        C[num_chroma[0:I - 1]] = spec
        C /= num_freqs_per_chroma
    final_matrix = np.zeros((12, 1))
    newD = int(np.ceil(C.shape[0] / 12.0) * 12)
    C2 = np.zeros((newD,))
    C2[0:C.shape[0]] = C
    C2 = C2.reshape(C2.shape[0] // 12, 12)
    final_matrix = np.sum(C2, axis=0).reshape(12, 1)
    final_matrix /= spec.sum()
    return chroma_names, final_matrix

def feature_extraction(signal, sampling_rate, window, step):
    signal = dc_normalize(signal)
    signal_length = len(signal)
    window_length = int(round(window * sampling_rate))
    step_length = int(round(step * sampling_rate))
    num_fft = int(window_length / 2)
    cur_pos = 0
    count_frames = 0
    n_frames = int(math.ceil((signal_length - window_length) / step_length)) + 1
    features = []
    while cur_pos + window_length - 1 < signal_length:
        count_frames += 1
        frame = signal[cur_pos:cur_pos + window_length]
        cur_pos += step_length
        frame = frame * np.hamming(len(frame))
        fft_magnitude = abs(fft(frame))[:num_fft]
        fft_magnitude /= len(fft_magnitude)
        if count_frames == 1:
            prev_fft_magnitude = fft_magnitude.copy()
        zcr = zero_crossing_rate(frame)
        en = energy(frame)
        entropy = energy_entropy(frame)
        centroid, spread = spectral_centroid_spread(fft_magnitude, sampling_rate)
        ent = spectral_entropy(fft_magnitude)
        sf = spectral_flux(fft_magnitude, prev_fft_magnitude)
        rolloff = spectral_rolloff(fft_magnitude, 0.90)
        hr, f0 = harmonic(frame, sampling_rate)
        if f0 > 5000:
            f0 = 0.0
        mfcc_feats = mfcc(fft_magnitude, mfcc_filter_banks(sampling_rate, num_fft)[0], 13)
        chroma_names, chroma_feat = chroma_features(fft_magnitude, sampling_rate, num_fft)
        feature_vector = np.zeros((33,))
        feature_vector[0] = zcr
        feature_vector[1] = en
        feature_vector[2] = entropy
        feature_vector[3] = centroid
        feature_vector[4] = spread
        feature_vector[5] = ent
        feature_vector[6] = sf
        feature_vector[7] = rolloff
        feature_vector[8] = hr
        feature_vector[9] = f0
        feature_vector[10:23] = mfcc_feats
        feature_vector[23:] = chroma_feat[:, 0]
        features.append(feature_vector)
        prev_fft_magnitude = fft_magnitude.copy()
    features = np.array(features)
    return features