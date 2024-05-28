"""!----------------WARNING----------------!
              DO NOT USE THIS CODE
   !----------------WARNING----------------!
"""
import numpy as np
import csv
import audioBasicIO
import MidTermFeatures as mtf
import hmmlearn.hmm
import glob, os, pickle

def segments_to_labels(start_times, end_times, labels, window):
    """
    This function converts segment endpoints and respective segment
    labels to fix-sized class labels.
    ARGUMENTS:
     - start_times:  segment start points (in seconds)
     - end_times:    segment endpoints (in seconds)
     - labels:       segment labels
     - window:      fix-sized window (in seconds)
    RETURNS:
     - flags:    np array of class indices
     - class_names:    list of classnames (strings)
    """
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


def read_segmentation_gt(gt_file):
    """
    This function reads a segmentation ground truth file,
    following a simple CSV format with the following columns:
    <segment start>,<segment end>,<class label>

    ARGUMENTS:
     - gt_file:       the path of the CSV segment file
    RETURNS:
     - seg_start:     a np array of segments' start positions
     - seg_end:       a np array of segments' ending positions
     - seg_label:     a list of respective class labels (strings)
    """
    with open(gt_file, 'rt') as f_handle:
        reader = csv.reader(f_handle, delimiter='\t')
        start_times = []
        end_times = []
        labels = []
        for row in reader:
            if len(row) == 3:
                start_times.append(float(row[0]))
                end_times.append(float(row[1]))
                labels.append((row[2]))
    return np.array(start_times), np.array(end_times), labels


def train_hmm_compute_statistics(features, labels):
    """
    This function computes the statistics used to train
    an HMM joint segmentation-classification model
    using a sequence of sequential features and respective labels

    ARGUMENTS:
     - features:  a np matrix of feature vectors (numOfDimensions x n_wins)
     - labels:    a np array of class indices (n_wins x 1)
    RETURNS:
     - class_priors:            matrix of prior class probabilities
                                (n_classes x 1)
     - transmutation_matrix:    transition matrix (n_classes x n_classes)
     - means:                   means matrix (numOfDimensions x 1)
     - cov:                     deviation matrix (numOfDimensions x 1)
    """
    unique_labels = np.unique(labels)
    n_comps = len(unique_labels)

    n_feats = features.shape[0]

    if features.shape[1] < labels.shape[0]:
        print("trainHMM warning: number of short-term feature vectors "
              "must be greater or equal to the labels length!")
        labels = labels[0:features.shape[1]]

    # compute prior probabilities:
    class_priors = np.zeros((n_comps,))
    for i, u_label in enumerate(unique_labels):
        class_priors[i] = np.count_nonzero(labels == u_label)
    # normalize prior probabilities
    class_priors = class_priors / class_priors.sum()

    # compute transition matrix:
    transmutation_matrix = np.zeros((n_comps, n_comps))
    for i in range(labels.shape[0]-1):
        transmutation_matrix[int(labels[i]), int(labels[i + 1])] += 1
    # normalize rows of transition matrix:
    for i in range(n_comps):
        transmutation_matrix[i, :] /= transmutation_matrix[i, :].sum()

    means = np.zeros((n_comps, n_feats))
    for i in range(n_comps):
        means[i, :] = \
            np.array(features[:,
                     np.nonzero(labels == unique_labels[i])[0]].mean(axis=1))

    cov = np.zeros((n_comps, n_feats))
    for i in range(n_comps):
        """
        cov[i, :, :] = np.cov(features[:, np.nonzero(labels == u_labels[i])[0]])
        """
        # use line above if HMM using full gaussian distributions are to be used
        cov[i, :] = np.std(features[:,
                           np.nonzero(labels == unique_labels[i])[0]],
                           axis=1)

    return class_priors, transmutation_matrix, means, cov


def train_hmm_from_file(wav_file, gt_file, hmm_model_name, mid_window, mid_step):
    """
    This function trains a HMM model for segmentation-classification
    using a single annotated audio file
    ARGUMENTS:
     - wav_file:        the path of the audio filename
     - gt_file:         the path of the ground truth filename
                       (a csv file of the form <segment start in seconds>,
                       <segment end in seconds>,<segment label> in each row
     - hmm_model_name:   the name of the HMM model to be stored
     - mt_win:          mid-term window size
     - mt_step:         mid-term window step
    RETURNS:
     - hmm:            an object to the resulting HMM
     - class_names:     a list of class_names

    After training, hmm, class_names, along with the mt_win and mt_step
    values are stored in the hmm_model_name file
    """

    seg_start, seg_end, seg_labs = read_segmentation_gt(gt_file)
    flags, class_names = segments_to_labels(seg_start, seg_end, seg_labs, mid_step)
    sampling_rate, signal = audioBasicIO.read_audio_file(wav_file)
    features, _, _ = \
        mtf.mid_feature_extraction(signal, sampling_rate,
                                   mid_window * sampling_rate,
                                   mid_step * sampling_rate,
                                   round(sampling_rate * 0.050),
                                   round(sampling_rate * 0.050))
    class_priors, transumation_matrix, means, cov = \
        train_hmm_compute_statistics(features, flags)
    hmm = hmmlearn.hmm.GaussianHMM(class_priors.shape[0], "diag")

    hmm.covars_ = cov
    hmm.means_ = means
    hmm.startprob_ = class_priors
    hmm.transmat_ = transumation_matrix

    save_hmm(hmm_model_name, hmm, class_names, mid_window, mid_step)

    return hmm, class_names


def train_hmm_from_directory(folder_path, hmm_model_name, mid_window, mid_step):
    """
    This function trains a HMM model for segmentation-classification using
    a where WAV files and .segment (ground-truth files) are stored
    ARGUMENTS:
     - folder_path:     the path of the data diretory
     - hmm_model_name:  the name of the HMM model to be stored
     - mt_win:          mid-term window size
     - mt_step:         mid-term window step
    RETURNS:
     - hmm:            an object to the resulting HMM
     - class_names:    a list of class_names

    After training, hmm, class_names, along with the mt_win
    and mt_step values are stored in the hmm_model_name file
    """

    flags_all = np.array([])
    class_names_all = []
    for i, f in enumerate(glob.glob(folder_path + os.sep + '*.wav')):
        # for each WAV file
        wav_file = f
        gt_file = f.replace('.wav', '.segments')
        if os.path.isfile(gt_file):
            seg_start, seg_end, seg_labs = read_segmentation_gt(gt_file)
            flags, class_names = \
                segments_to_labels(seg_start, seg_end, seg_labs, mid_step)
            for c in class_names:
                # update class names:
                if c not in class_names_all:
                    class_names_all.append(c)
            sampling_rate, signal = audioBasicIO.read_audio_file(wav_file)
            feature_vector, _, _ = \
                mtf.mid_feature_extraction(signal, sampling_rate,
                                           mid_window * sampling_rate,
                                           mid_step * sampling_rate,
                                           round(sampling_rate * 0.050),
                                           round(sampling_rate * 0.050))

            flag_len = len(flags)
            feat_cols = feature_vector.shape[1]
            min_sm = min(feat_cols, flag_len)
            feature_vector = feature_vector[:, 0:min_sm]
            flags = flags[0:min_sm]

            flags_new = []
            # append features and labels
            for j, fl in enumerate(flags):
                flags_new.append(class_names_all.index(class_names_all[flags[j]]))

            flags_all = np.append(flags_all, np.array(flags_new))

            if i == 0:
                f_all = feature_vector
            else:
                f_all = np.concatenate((f_all, feature_vector), axis=1)

    # compute HMM statistics
    class_priors, transmutation_matrix, means, cov = \
        train_hmm_compute_statistics(f_all, flags_all)
    # train the HMM
    hmm = hmmlearn.hmm.GaussianHMM(class_priors.shape[0], "diag")
    hmm.covars_ = cov
    hmm.means_ = means
    hmm.startprob_ = class_priors
    hmm.transmat_ = transmutation_matrix

    save_hmm(hmm_model_name, hmm, class_names_all, mid_window, mid_step)

    return hmm, class_names_all


def save_hmm(hmm_model_name, model, classes, mid_window, mid_step):
    """Save HMM model"""
    with open(hmm_model_name, "wb") as f_handle:
        pickle.dump(model, f_handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(classes, f_handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(mid_window, f_handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(mid_step, f_handle, protocol=pickle.HIGHEST_PROTOCOL)