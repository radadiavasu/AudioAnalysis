# from sklearn.cluster import KMeans
# import numpy as np
# import os
# import glob
# import audioBasicIO as audioBasicIO
# import MidTermFeatures as mtf

# def segment_audio_using_clustering(input_folder, mid_window, mid_step, n_clusters=2):
#     """
#     Segment audio using k-means clustering on mid-term features and generate a .segments file.

#     This function reads an audio file, extracts mid-term features, and applies k-means clustering
#     to segment the audio based on feature similarities. The segments are saved to a .segments file
#     with the same base name as the audio file.

#     Parameters:
#     wav_file (str): Path to the input .wav audio file.
#     n_clusters (int): Number of clusters for k-means clustering. Default is 2.
#     mt_win (float): Mid-term window size in seconds. Default is 1.0.
#     mt_step (float): Mid-term window step in seconds. Default is 0.5.

#     Returns:
#     None: This function saves the segmentation results to a .segments file into your .wav files containing folder.

#     Example:
#     >>> segment_audio_using_clustering('audio.wav')
#     The function will create 'audio.segments' with the segments identified by k-means clustering.

#     Note:
#     - Ensure that ffmpeg or avconv is installed and accessible via PATH. I don't have installed ffmpeg or arcnov but it still works, 
#       i ain't know how.
#     - The duration of the segments will not exceed the length of the audio file.
#     """
#     for wav_file in glob.glob(os.path.join(input_folder, '*.wav')):
#         sampling_rate, signal = audioBasicIO.read_audio_file(wav_file)
#         actual_duration = len(signal) / sampling_rate

#         feature_vector, _, _ = mtf.mid_feature_extraction(
#             signal, sampling_rate,
#             mid_window * sampling_rate,
#             mid_step * sampling_rate,
#             round(sampling_rate * 0.050),
#             round(sampling_rate * 0.050)
#         )

#         feature_vector = feature_vector.T  # Transpose to shape (n_samples, n_features)
#         kmeans = KMeans(n_clusters=n_clusters)
#         labels = kmeans.fit_predict(feature_vector)

#         # Generate .segments file
#         segment_file = wav_file.replace('.wav', '.segments')
#         with open(segment_file, 'w') as f:
#             for i, label in enumerate(labels):
#                 start_time = i * mid_step
#                 end_time = (i + 1) * mid_step
#                 if end_time > actual_duration:
#                     end_time = actual_duration
#                 if start_time >= actual_duration:
#                     break
#                 f.write(f"{start_time},{end_time},{label}\n")

# if __name__ == "__main__":
#     input_folder = r'G:\pyAudioAnalysis\pyAudioAnalysis\data\speechEmotion'
#     mid_window = 1.0  # Mid-term window size
#     mid_step = 0.5  # Mid-term window step
#     segment_audio_using_clustering(input_folder, mid_window, mid_step, n_clusters=2)


from sklearn.cluster import KMeans
import numpy as np
import os
import glob
import audioBasicIO as audioBasicIO
import MidTermFeatures as mtf
from pyAudioAnalysis import audioTrainTest as aT

def classify_segment(features, classifier_model):
    """
    Classify a segment using a pre-trained classifier model.

    Parameters:
    features (np.array): The feature vector of the segment.
    classifier_model (tuple): The classifier model and the respective MEAN and STD values for feature normalization.

    Returns:
    str: The predicted label for the segment.
    """
    model, MEAN, STD = classifier_model
    features_normalized = (features - MEAN) / STD
    result, _ = aT.classifier_wrapper(model, "knn", features_normalized)
    return result

def segment_audio_using_clustering(input_folder, mid_window, mid_step, n_clusters=2, classifier_model=None):
    """
    Segment audio using k-means clustering on mid-term features and generate a .segments file with tags.

    Parameters:
    input_folder (str): Path to the folder containing .wav audio files.
    mid_window (float): Mid-term window size in seconds.
    mid_step (float): Mid-term window step in seconds.
    n_clusters (int): Number of clusters for k-means clustering. Default is 2.
    classifier_model (tuple): The classifier model and the respective MEAN and STD values for feature normalization.

    Returns:
    None: This function saves the segmentation results to a .segments file in the input folder.
    """
    for wav_file in glob.glob(os.path.join(input_folder, '*.wav')):
        sampling_rate, signal = audioBasicIO.read_audio_file(wav_file)
        actual_duration = len(signal) / sampling_rate

        feature_vector, _, _ = mtf.mid_feature_extraction(
            signal, sampling_rate,
            mid_window * sampling_rate,
            mid_step * sampling_rate,
            round(sampling_rate * 0.050),
            round(sampling_rate * 0.050)
        )

        feature_vector = feature_vector.T  # Transpose to shape (n_samples, n_features)
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(feature_vector)

        # Generate .segments file
        segment_file = wav_file.replace('.wav', '.segments')
        with open(segment_file, 'w') as f:
            for i, label in enumerate(labels):
                start_time = i * mid_step
                end_time = (i + 1) * mid_step
                if end_time > actual_duration:
                    end_time = actual_duration
                if start_time >= actual_duration:
                    break
                
                # Classify the segment if a classifier model is provided
                if classifier_model:
                    segment_features = feature_vector[i]
                    tag = classify_segment(segment_features, classifier_model)
                else:
                    tag = "unknown"

                f.write(f"{start_time},{end_time},{label},{tag}\n")

if __name__ == "__main__":
    input_folder = r'G:\pyAudioAnalysis\pyAudioAnalysis\data\speechEmotion'
    mid_window = 1.0  # Mid-term window size
    mid_step = 0.5  # Mid-term window step

    # Load a pre-trained classifier model (example: KNN model)
    model_name = r"regression.pkl_arousalMEANS"
    model_path = r"G:\pyAudioAnalysis\pyAudioAnalysis\data"
    classifier_model, MEAN, STD, class_names, _ = aT.load_model(model_path, model_name)

    segment_audio_using_clustering(input_folder, mid_window, mid_step, n_clusters=2, classifier_model=(classifier_model, MEAN, STD))
