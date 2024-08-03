import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
from pyAudioAnalysis import audioTrainTest as aT

def train_hmm_model(data_folders, model_name):
    # Extract features and train HMM
    aT.extract_features_and_train(data_folders, 1.0, 1.0, 0.05, 0.05, "hmm", model_name, False)

if __name__ == "__main__":
    data_folders = [
        r"G:\pyAudioAnalysis\pytests\test_data\3_class\speech",
        r"G:\pyAudioAnalysis\pytests\test_data\3_class\music",
        r"G:\pyAudioAnalysis\pytests\test_data\3_class\silence"
    ]
    model_name = "hmm_4class"
    train_hmm_model(data_folders, model_name)
