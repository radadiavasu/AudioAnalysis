import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioSegmentation as aS
import numpy as np


def test_speaker_diarization():
    labels, purity_cluster_m, purity_speaker_m = \
        aS.speaker_diarization("test_data/diarizationExample.wav", 
                                4, plot_res=False)
    assert purity_cluster_m > 0.9, "Diarization cluster purity is low"
    assert purity_speaker_m > 0.9, "Diarization speaker purity is low"


def test_mt_file_classification():
    labels, class_names, accuracy, cm = aS.mid_term_file_classification(
                                     "test_data/scottish.wav", 
                                     "test_data/svm_rbf_sm", "svm_rbf", False, 
                                     "test_data/scottish.segments")
    assert accuracy > 0.95, "Segment-level classification accuracy is low"
    

def save_hmm(hmm_model_name, hmm, class_names, mid_window, mid_step):
    """
    Save the HMM model to a file.
    Arguments:
    hmm_model_name -- Path to save the HMM model
    hmm -- Trained HMM object
    class_names -- List of class names
    mid_window -- Mid-term window size (in seconds)
    mid_step -- Mid-term window step (in seconds)
    """
    # Ensure that dir exists.
    try:
        import os
        os.makedirs(os.path.dirname(hmm_model_name), exist_ok=True)
    except FileExistsError as e:
        print("File already exists.", e)
    except OSError as e:
        print("OS error while creating directories:", e)

    try:
        import pickle
        with open(hmm_model_name, 'wb') as f:
            pickle.dump({'hmm': hmm, 'class_names': class_names, 'mid_window': mid_window, 'mid_step': mid_step}, f)
    except FileNotFoundError as e:
        print(f"Error: The file {hmm_model_name} not found in desired directory.", e)
    except PermissionError as e:
        print(f"Error: Permission denied while trying to write the file named {hmm_model_name}.", e)
    except pickle.PicklingError as e:
        print(f"Error: Failed to pickle object.", e)
    except Exception as e:
        print("An unexpected error occurred while saving the model.", e)

# Test Cases
def test_save_hmm():
    hmm = "dummy_hmm"  # Replace with actual HMM object for real testing
    class_names = ["class1", "class2"]
    mid_window = 1.0
    mid_step = 0.5

    # Test FileNotFoundError (use a path with invalid characters)
    print("Testing FileNotFoundError:")
    save_hmm(r"G:/invalid_path<>*?/hmm_model.pkl", hmm, class_names, mid_window, mid_step)

    # Test PermissionError (try saving to a system directory, which typically requires admin rights)
    print("\nTesting PermissionError:")
    save_hmm(r"C:/Windows/System32/hmm_model.pkl", hmm, class_names, mid_window, mid_step)

    # Test PicklingError
    print("\nTesting PicklingError:")
    save_hmm(r"G:/pyAudioAnalysis/pyAudioAnalysis/data/models/hmm_model.pkl", test_save_hmm, class_names, mid_window, mid_step)  # Passing a function to induce PicklingError

    # Test general exception with an unhandled error
    print("\nTesting general exception:")
    try:
        save_hmm("", hmm, class_names, mid_window, mid_step)  # This should raise an error due to empty file path
    except Exception as e:
        print("Caught exception:", e)

# Run the tests
test_save_hmm()
    