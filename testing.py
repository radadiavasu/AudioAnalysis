# import os
# import glob
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report
# from pyAudioAnalysis import audioTrainTest as aT
# from pyAudioAnalysis import MidTermFeatures as mtf
# from pyAudioAnalysis import audioBasicIO
# import pickle
# # Set the path to ffmpeg explicitly for pydub
# from pydub.utils import which
# from pydub import AudioSegment
# AudioSegment.ffmpeg = which("ffmpeg") or "C:\\ffmpeg\\bin\\ffmpeg.exe"
# print("Using ffmpeg from:", AudioSegment.ffmpeg)


# def extract_features(audio_folder, mid_window, mid_step):
#     features = []
#     labels = []
#     class_names = []

#     for class_folder in glob.glob(os.path.join(audio_folder, '*')):
#         if os.path.isdir(class_folder):
#             class_name = os.path.basename(class_folder)
#             class_names.append(class_name)
#             for wav_file in glob.glob(os.path.join(class_folder, '*.wav')):
#                 print(f"Processing file: {wav_file}")
#                 sampling_rate, signal = audioBasicIO.read_audio_file(wav_file)
#                 feature_vector, _, _ = mtf.mid_feature_extraction(
#                     signal, sampling_rate,
#                     mid_window * sampling_rate,
#                     mid_step * sampling_rate,
#                     round(sampling_rate * 0.050),
#                     round(sampling_rate * 0.050)
#                 )
#                 if feature_vector.size == 0:
#                     print(f"No features extracted for file: {wav_file}")
#                     continue
#                 features.append(np.mean(feature_vector, axis=1))  # Average features over mid-term window
#                 labels.append(class_name)
    
#     if len(features) == 0:
#         print("No features extracted for any files. Exiting.")
#         return np.array([]), np.array([]), class_names

#     features = np.array(features)
#     labels = np.array(labels)
#     print(f"Extracted {features.shape[0]} feature vectors with {features.shape[1]} features each.")
#     return features, labels, class_names

# def train_classifier(features, labels):
#     if features.size == 0:
#         raise ValueError("No features to train on. Check the feature extraction step.")
    
#     scaler = StandardScaler()
#     features_scaled = scaler.fit_transform(features)
    
#     X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)
#     classifier = KNeighborsClassifier(n_neighbors=5)
#     classifier.fit(X_train, y_train)
    
#     y_pred = classifier.predict(X_test)
#     print(classification_report(y_test, y_pred))
    
#     return classifier, scaler

# def save_model(classifier, scaler, class_names, model_path):
#     model = (classifier, scaler, class_names)
#     with open(model_path, 'wb') as f:
#         pickle.dump(model, f)

# def load_model(model_path):
#     with open(model_path, 'rb') as f:
#         return pickle.load(f)


# if __name__ == "__main__":
#     audio_folder = r'G:\pyAudioAnalysis\pyAudioAnalysis\data\speechTesting'
#     mid_window = 1.0
#     mid_step = 0.5
    
#     features, labels, class_names = extract_features(audio_folder, mid_window, mid_step)
    
#     if features.size == 0:
#         print("No valid features extracted. Exiting.")
#     else:
#         classifier, scaler = train_classifier(features, labels)
#         model_path = r'G:\pyAudioAnalysis\pyAudioAnalysis\data\models\knnSM'
#         save_model(classifier, scaler, class_names, model_path)






# *********************************************** #
  # Tightless Validation of Feature Extraction. # 
# *********************************************** #
import numpy as np
from scipy.fftpack import fft
import sys
from pyAudioAnalysis import audioBasicIO
from scipy.fftpack import dct

eps = sys.float_info.epsilon
# Define the functions required for feature extraction
def dc_normalize(signal):
    return signal - np.mean(signal)

def zero_crossing_rate(signal):
    return np.mean(np.abs(np.diff(np.sign(signal))))

def energy(signal):
    return np.sum(signal ** 2) / np.float64(len(signal))

def energy_entropy(signal, num_short_blocks=10):
    total_energy = np.sum(signal ** 2)
    sub_win_len = int(np.floor(len(signal) / num_short_blocks))
    sub_wins = np.split(signal[:sub_win_len * num_short_blocks], num_short_blocks)
    entropy = 0.0
    for sub_win in sub_wins:
        sub_energy = np.sum(sub_win ** 2)
        entropy -= (sub_energy / total_energy) * np.log2(sub_energy / total_energy + 1e-12)
    return entropy

def spectral_centroid_spread(fft_magnitude, sampling_rate):
    ind = (np.arange(1, len(fft_magnitude) + 1)) * (sampling_rate / (2.0 * len(fft_magnitude)))
    centroid = np.sum(ind * fft_magnitude) / (np.sum(fft_magnitude) + 1e-12)
    spread = np.sqrt(np.sum(((ind - centroid) ** 2) * fft_magnitude) / (np.sum(fft_magnitude) + 1e-12))
    return centroid, spread

def spectral_entropy(fft_magnitude, num_short_blocks=10):
    fft_magnitude = fft_magnitude / np.sum(fft_magnitude)
    sub_win_len = int(np.floor(len(fft_magnitude) / num_short_blocks))
    sub_wins = np.split(fft_magnitude[:sub_win_len * num_short_blocks], num_short_blocks)
    entropy = 0.0
    for sub_win in sub_wins:
        sub_energy = np.sum(sub_win)
        entropy -= sub_energy * np.log2(sub_energy + 1e-12)
    return entropy

def spectral_flux(fft_magnitude, fft_magnitude_previous):
    return np.sum((fft_magnitude - fft_magnitude_previous) ** 2) / (len(fft_magnitude) + 1e-12)

def spectral_rolloff(fft_magnitude, c):
    energy = np.sum(fft_magnitude ** 2)
    threshold = c * energy
    cumulative_sum = np.cumsum(fft_magnitude ** 2)
    return np.where(cumulative_sum >= threshold)[0][0]

def mfcc(fft_magnitude, fbank, num_mfcc_feats):
    # Implement MFCC calculation based on the filter banks
    mspec = np.log10(np.dot(fft_magnitude, fbank.T) + eps)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:num_mfcc_feats]
    return ceps

def mfcc_filter_banks(sampling_rate, num_fft, lowfreq=133.33, linc=200 / 3,
                    logsc=1.0711703, num_lin_filt=13, num_log_filt=27):
    """
    Computes the triangular filterbank for MFCC computation 
    (used in the stFeatureExtraction function before the stMFCC function call)
    This function is taken from the scikits.talkbox library (MIT Licence):
    https://pypi.python.org/pypi/scikits.talkbox
    """

    if sampling_rate < 8000:
        nlogfil = 5

    # Total number of filters
    num_filt_total = num_lin_filt + num_log_filt

    # Compute frequency points of the triangle:
    frequencies = np.zeros(num_filt_total + 2)
    frequencies[:num_lin_filt] = lowfreq + np.arange(num_lin_filt) * linc
    frequencies[num_lin_filt:] = frequencies[num_lin_filt - 1] * logsc ** \
                                    np.arange(1, num_log_filt + 3)
    heights = 2. / (frequencies[2:] - frequencies[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((num_filt_total, num_fft))
    nfreqs = np.arange(num_fft) / (1. * num_fft) * sampling_rate

    for i in range(num_filt_total):
        low_freqs = frequencies[i]
        cent_freqs = frequencies[i + 1]
        high_freqs = frequencies[i + 2]

        lid = np.arange(np.floor(low_freqs * num_fft / sampling_rate) + 1,
                        np.floor(cent_freqs * num_fft / sampling_rate) + 1,
                        dtype=int)
        lslope = heights[i] / (cent_freqs - low_freqs)
        rid = np.arange(np.floor(cent_freqs * num_fft / sampling_rate) + 1,
                        np.floor(high_freqs * num_fft / sampling_rate) + 1,
                        dtype=int)
        rslope = heights[i] / (high_freqs - cent_freqs)
        fbank[i][lid] = lslope * (nfreqs[lid] - low_freqs)
        fbank[i][rid] = rslope * (high_freqs - nfreqs[rid])

    return fbank, frequencies

def chroma_features_init(num_fft, sampling_rate):
    """
    This function initializes the chroma matrices used in the calculation
    of the chroma features
    """
    freqs = np.array([((f + 1) * sampling_rate) /
                      (2 * num_fft) for f in range(num_fft)])
    cp = 27.50
    num_chroma = np.round(12.0 * np.log2(freqs / cp)).astype(int)

    num_freqs_per_chroma = np.zeros((num_chroma.shape[0],))

    unique_chroma = np.unique(num_chroma)
    for u in unique_chroma:
        idx = np.nonzero(num_chroma == u)
        num_freqs_per_chroma[idx] = idx[0].shape

    return num_chroma, num_freqs_per_chroma

def chroma_features(signal, sampling_rate, num_fft):
    # Implement chroma feature extraction
    num_chroma, num_freqs_per_chroma = \
        chroma_features_init(num_fft, sampling_rate)
    chroma_names = ['A', 'A#', 'B', 'C', 'C#', 'D',
                    'D#', 'E', 'F', 'F#', 'G', 'G#']
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
    C2 = C2.reshape(int(C2.shape[0] / 12), 12)
    # for i in range(12):
    #    finalC[i] = np.sum(C[i:C.shape[0]:12])
    final_matrix = np.sum(C2, axis=0).reshape(1, -1).T

    spec_sum = spec.sum()
    if spec_sum == 0:
        final_matrix /= eps
    else:
        final_matrix /= spec_sum
    return chroma_names, final_matrix
    
def feature_extraction(signal, sampling_rate, window, step, deltas=True):
    window = int(window)
    step = int(step)

    signal = np.double(signal)
    signal = signal / (2.0 ** 15)
    signal = dc_normalize(signal)

    number_of_samples = len(signal)
    current_position = 0
    count_fr = 0
    num_fft = int(window / 2)

    fbank, freqs = mfcc_filter_banks(sampling_rate, num_fft)

    n_time_spectral_feats = 8
    n_mfcc_feats = 13
    n_chroma_feats = 13
    n_total_feats = n_time_spectral_feats + n_mfcc_feats + n_chroma_feats

    feature_names = ["zcr", "energy", "energy_entropy",
                     "spectral_centroid", "spectral_spread",
                     "spectral_entropy", "spectral_flux", "spectral_rolloff"]
    feature_names += ["mfcc_{0:d}".format(mfcc_i) for mfcc_i in range(1, n_mfcc_feats + 1)]
    feature_names += ["chroma_{0:d}".format(chroma_i) for chroma_i in range(1, n_chroma_feats)]
    feature_names.append("chroma_std")

    if deltas:
        feature_names_2 = feature_names + ["delta " + f for f in feature_names]
        feature_names = feature_names_2
 
    features = []

    while current_position + window - 1 < number_of_samples:
        count_fr += 1
        x = signal[current_position:current_position + window]
        current_position = current_position + step

        fft_magnitude = abs(fft(x))[:num_fft]
        fft_magnitude = fft_magnitude / len(fft_magnitude)

        if count_fr == 1:
            fft_magnitude_previous = fft_magnitude.copy()
        feature_vector = np.zeros((n_total_feats, 1))

        feature_vector[0] = zero_crossing_rate(x)
        feature_vector[1] = energy(x)
        feature_vector[2] = energy_entropy(x)
        feature_vector[3], feature_vector[4] = spectral_centroid_spread(fft_magnitude, sampling_rate)
        feature_vector[5] = spectral_entropy(fft_magnitude)
        feature_vector[6] = spectral_flux(fft_magnitude, fft_magnitude_previous)
        feature_vector[7] = spectral_rolloff(fft_magnitude, 0.90)
        feature_vector[n_time_spectral_feats:n_time_spectral_feats + n_mfcc_feats, 0] = mfcc(fft_magnitude, fbank, n_mfcc_feats).copy()

        chroma_names, chroma_feature_matrix = chroma_features(fft_magnitude, sampling_rate, num_fft)
        chroma_features_end = n_time_spectral_feats + n_mfcc_feats + n_chroma_feats - 1
        feature_vector[n_time_spectral_feats + n_mfcc_feats:chroma_features_end] = chroma_feature_matrix
        feature_vector[chroma_features_end] = chroma_feature_matrix.std()
        
        if not deltas:
            features.append(feature_vector)
        else:
            if count_fr > 1:
                delta = feature_vector - feature_vector_previous
                feature_vector_2 = np.concatenate((feature_vector, delta))
            else:
                feature_vector_2 = np.concatenate((feature_vector, np.zeros(feature_vector.shape)))
            feature_vector_previous = feature_vector
            features.append(feature_vector_2)

        fft_magnitude_previous = fft_magnitude.copy()

    features = np.concatenate(features, 1)
    return features, feature_names

# Mid-term feature extraction function
def mid_feature_extraction(signal, sampling_rate, mid_window, mid_step, short_window, short_step):
    short_features, short_feature_names = feature_extraction(signal, sampling_rate, short_window, short_step)

    n_stats = 2
    n_feats = len(short_features)
    mid_window_ratio = round((mid_window - (short_window - short_step)) / short_step)
    mt_step_ratio = int(round(mid_step / short_step))

    mid_features, mid_feature_names = [], []
    for i in range(n_stats * n_feats):
        mid_features.append([])
        mid_feature_names.append("")

    for i in range(n_feats):
        cur_position = 0
        num_short_features = len(short_features[i])
        mid_feature_names[i] = short_feature_names[i] + "_" + "mean"
        mid_feature_names[i + n_feats] = short_feature_names[i] + "_" + "std"

        while cur_position < num_short_features:
            end = cur_position + mid_window_ratio
            if end > num_short_features:
                end = num_short_features
            cur_st_feats = short_features[i][cur_position:end]

            mid_features[i].append(np.mean(cur_st_feats))
            mid_features[i + n_feats].append(np.std(cur_st_feats))
            cur_position += mt_step_ratio
    mid_features = np.array(mid_features)
    mid_features = np.nan_to_num(mid_features)
    return mid_features, short_features, mid_feature_names

# Test feature extraction functions
def test_feature_extraction_short(file_path):
    [fs, x] = audioBasicIO.read_audio_file(file_path)
    window_size = 0.050 * fs
    step_size = 0.050 * fs
    if len(x) < window_size:
        print(f"Audio file {file_path} is too short for the given window size.")
        return
    F, f_names = feature_extraction(x, fs, window_size, step_size)
    print("Feature matrix shape:", F.shape)
    print("Feature names:", f_names)
    assert F.shape[1] > 0, "No short-term windows extracted"
    assert F.shape[0] == len(f_names), "Number of features and feature names are not the same"

def test_feature_extraction_segment(file_path):
    print("Short-term feature extraction")
    [fs, x] = audioBasicIO.read_audio_file(file_path)
    mt_window_size = 1 * fs
    mt_step_size = 1 * fs
    st_window_size = 0.05 * fs
    st_step_size = 0.05 * fs
    if len(x) < mt_window_size:
        print(f"Audio file {file_path} is too short for the given mid-term window size.")
        return
    mt, st, mt_names = mid_feature_extraction(x, fs, mt_window_size, mt_step_size, st_window_size, st_step_size)
    print("Mid-term feature matrix shape:", mt.shape)
    print("Short-term feature matrix shape:", st.shape)
    print("Mid-term feature names:", mt_names)
    assert mt.shape[1] > 0, "No mid-term windows extracted"
    assert mt.shape[0] == len(mt_names), "Number of features and feature names are not the same"

if __name__ == "__main__":
    test_feature_extraction_short("G:/pyAudioAnalysis/pytests/test_data/scottish.wav")
    test_feature_extraction_segment("G:/pyAudioAnalysis/pytests/test_data/count.wav")
    print("All tests passed.")
