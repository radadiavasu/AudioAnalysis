import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
import sklearn.cluster
import sklearn.discriminant_analysis as lda
import audioSegmentation as aF
import audioTrainTest as at
import scipy.signal
import audioBasicIO
import MidTermFeatures as mtf

# Suppress InconsistentVersionWarning
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")

# Specify the input parameters
filename = r"G:\pyAudioAnalysis\pyAudioAnalysis\data\recording1.wav"
n_speakers = 1  # Number of speakers (clusters) in the recording
mid_window = 1.0  # Mid-term window size (optional, defaults to 1.0)
mid_step = 0.1  # Mid-term window step (optional, defaults to 0.1)
short_window = 0.1  # Short-term window size (optional, defaults to 0.1)
lda_dim = 0  # LDA dimension (optional, defaults to 0)
plot_res = True  # Whether to plot the results (optional, defaults to False)

def segments_to_labels(start_times, end_times, labels, window):
    """
    Convert segment endpoints and respective segment labels to fixed-size class labels.
    Arguments:
    start_times -- List of segment start times
    end_times -- List of segment end times
    labels -- List of segment labels
    window -- Fixed-size window (in seconds)
    Returns:
    flags -- Numpy array of class indices
    class_names -- List of classnames (strings)
    """
    if not end_times:
        raise ValueError("end_times list is empty")

    flags = []
    class_names = list(set(labels))
    index = window / 2.0

    while index < end_times[-1]:
        for i in range(len(start_times)):
            if start_times[i] < index <= end_times[i]:
                break
        flags.append(class_names.index(labels[i]))
        index += window
    return np.array(flags), class_names

def speaker_diarization(filename, n_speakers, mid_window=1.0, mid_step=0.1,
                        short_window=0.1, lda_dim=0, plot_res=False):
    """
    ARGUMENTS:
        - filename:        the name of the WAV file to be analyzed
        - n_speakers       the number of speakers (clusters) in
                           the recording (<=0 for unknown)
        - mid_window (opt)    mid-term window size
        - mid_step (opt)    mid-term window step
        - short_window  (opt)    short-term window size
        - lda_dim (opt     LDA dimension (0 for no LDA)
        - plot_res         (opt)   0 for not plotting the results 1 for plotting
    """
    sampling_rate, signal = audioBasicIO.read_audio_file(filename)
    signal = audioBasicIO.stereo_to_mono(signal)
    duration = len(signal) / sampling_rate

    base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "data/models")

    classifier_all, mean_all, std_all, class_names_all, _, _, _, _, _ = \
        at.load_model(os.path.join(base_dir, "svm_rbf_speaker_10"))
    classifier_fm, mean_fm, std_fm, class_names_fm, _, _, _, _,  _ = \
        at.load_model(os.path.join(base_dir, "svm_rbf_speaker_male_female"))

    mid_feats, st_feats, _ = \
        mtf.mid_feature_extraction(signal, sampling_rate,
                                  mid_window * sampling_rate,
                                  mid_step * sampling_rate,
                                  round(sampling_rate * 0.05),
                                  round(sampling_rate * 0.05))

    mid_term_features = np.zeros((mid_feats.shape[0] + len(class_names_all) +
                                  len(class_names_fm), mid_feats.shape[1]))
    for index in range(mid_feats.shape[1]):
        feature_norm_all = (mid_feats[:, index] - mean_all) / std_all
        feature_norm_fm = (mid_feats[:, index] - mean_fm) / std_fm
        _, p1 = at.classifier_wrapper(classifier_all, "svm_rbf", feature_norm_all)
        _, p2 = at.classifier_wrapper(classifier_fm, "svm_rbf", feature_norm_fm)
        start = mid_feats.shape[0]
        end = mid_feats.shape[0] + len(class_names_all)
        mid_term_features[0:mid_feats.shape[0], index] = mid_feats[:, index]
        mid_term_features[start:end, index] = p1 + 1e-4
        mid_term_features[end::, index] = p2 + 1e-4

    # Normalize features:
    scaler = StandardScaler()
    mid_feats_norm = scaler.fit_transform(mid_term_features.T)

    # Remove outliers:
    dist_all = np.sum(distance.squareform(distance.pdist(mid_feats_norm.T)), axis=0)
    m_dist_all = np.mean(dist_all)
    i_non_outliers = np.nonzero(dist_all < 1.1 * m_dist_all)[0]

    mt_feats_norm_or = mid_feats_norm
    mid_feats_norm = mid_feats_norm[:, i_non_outliers]

    # LDA dimensionality reduction:
    if lda_dim > 0:
        window_ratio = int(round(mid_window / short_window))
        step_ratio = int(round(short_window / short_window))
        mt_feats_to_red = []
        num_of_features = len(st_feats)
        num_of_stats = 2
        for index in range(num_of_stats * num_of_features):
            mt_feats_to_red.append([])

        for index in range(num_of_features):
            cur_pos = 0
            feat_len = len(st_feats[index])
            while cur_pos < feat_len:
                n1 = cur_pos
                n2 = cur_pos + window_ratio
                if n2 > feat_len:
                    n2 = feat_len
                short_features = st_feats[index][n1:n2]
                mt_feats_to_red[index].append(np.mean(short_features))
                mt_feats_to_red[index + num_of_features].append(np.std(short_features))
                cur_pos += step_ratio
        mt_feats_to_red = np.array(mt_feats_to_red)
        mt_feats_to_red_2 = np.zeros((mt_feats_to_red.shape[0] + len(class_names_all) +
                                      len(class_names_fm), mt_feats_to_red.shape[1]))
        limit = mt_feats_to_red.shape[0] + len(class_names_all)
        for index in range(mt_feats_to_red.shape[1]):
            feature_norm_all = (mt_feats_to_red[:, index] - mean_all) / std_all
            feature_norm_fm = (mt_feats_to_red[:, index] - mean_fm) / std_fm
            _, p1 = at.classifier_wrapper(classifier_all, "svm_rbf", feature_norm_all)
            _, p2 = at.classifier_wrapper(classifier_fm, "svm_rbf", feature_norm_fm)
            mt_feats_to_red_2[0:mt_feats_to_red.shape[0], index] = mt_feats_to_red[:, index]
            mt_feats_to_red_2[mt_feats_to_red.shape[0]:limit, index] = p1 + 1e-4
            mt_feats_to_red_2[limit::, index] = p2 + 1e-4
        mt_feats_to_red = mt_feats_to_red_2
        scaler = StandardScaler()
        mt_feats_to_red = scaler.fit_transform(mt_feats_to_red.T).T
        labels = np.zeros((mt_feats_to_red.shape[1], ))
        lda_step = 1.0
        lda_step_ratio = lda_step / short_window
        for index in range(labels.shape[0]):
            labels[index] = int(index * short_window / lda_step_ratio)
        clf = lda.LinearDiscriminantAnalysis(n_components=lda_dim)
        mid_feats_norm = clf.fit_transform(mt_feats_to_red.T, labels)

    if n_speakers <= 0:
        s_range = range(2, 10)
    else:
        s_range = [n_speakers]
    cluster_labels = []
    sil_all = []

    for speakers in s_range:
        k_means = sklearn.cluster.KMeans(n_clusters=speakers)
        k_means.fit(mid_feats_norm)
        cls = k_means.labels_
        cluster_labels.append(cls)
        sil_1 = []
        sil_2 = []
        for c in range(speakers):
            clust_per_cent = np.nonzero(cls == c)[0].shape[0] / float(len(cls))
            if clust_per_cent < 0.020:
                sil_1.append(0.0)
                sil_2.append(0.0)
            else:
                mt_feats_norm_temp = mid_feats_norm[cls == c, :]
                dist = distance.pdist(mt_feats_norm_temp.T)
                sil_1.append(np.mean(dist) * clust_per_cent)
                sil_temp = []
                for c2 in range(speakers):
                    if c2 != c:
                        mt_feats_norm_temp2 = mid_feats_norm[cls == c2, :]
                        dist2 = distance.cdist(mt_feats_norm_temp, mt_feats_norm_temp2, 'euclidean')
                        sil_temp.append(np.mean(dist2))
                if sil_temp:  # Check if sil_temp is not empty
                    sil_2.append(min(sil_temp))
                else:
                    sil_2.append(float('inf'))  # Assign a high value if sil_temp is empty
        sil_all.append(np.mean(np.array(sil_2) - np.array(sil_1)))
    if len(s_range) > 1:
        speakers = s_range[np.argmax(sil_all)]
    cls = cluster_labels[np.argmax(sil_all)]
    seg_start, seg_end = aF.labels_to_segments(cls, mid_step)
    cls = [str(c) for c in cls]

    if plot_res:
        timeX = np.arange(0, duration, mid_step)
        # Recalculate timeX to match the shape of mid_feats
        timeX = np.arange(0, mid_feats.shape[1] * mid_step, mid_step)
        timeX = timeX[:mid_feats.shape[1]]  # Ensure timeX matches mid_feats in length

        plt.subplot(2, 1, 1)
        plt.plot(timeX, mid_feats[0, :])
        plt.xlabel("time (seconds)")
        plt.ylabel("feature values")
        plt.subplot(2, 1, 2)
        cls_unique = list(set(cls))
        class_labels = np.zeros((len(cls), len(cls_unique)))
        for i, c in enumerate(cls_unique):
            class_labels[:, i] = np.array([1 if cls[j] == c else 0 for j in range(len(cls))])
        plt.imshow(class_labels.T, aspect='auto', origin='lower')
        plt.xlabel("time (seconds)")
        plt.ylabel("speaker id")
        plt.show()

    return cls, seg_start, seg_end

# Run the speaker diarization
cls, seg_start, seg_end = speaker_diarization(filename, n_speakers, mid_window, mid_step, short_window, lda_dim, plot_res)

# Display the results
print("Class Labels:", cls)
print("Segment Start Times:", seg_start)
print("Segment End Times:", seg_end)