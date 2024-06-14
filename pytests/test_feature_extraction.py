import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
from pyAudioAnalysis import MidTermFeatures


def test_feature_extraction_short():
    [fs, x] = audioBasicIO.read_audio_file(r"G:\pyAudioAnalysis\pytests\test_data\1_sec_wav.wav")
    F, f_names = ShortTermFeatures.feature_extraction(x, fs, 
                                                      0.050 * fs, 0.050 * fs)
    print("Feature matrix shape:", F.shape)
    print("Feature names:", f_names)
    assert F.shape[1] == 20, "Wrong number of mid-term windows"
    assert F.shape[0] == len(f_names), "Number of features and feature " \
                                       "names are not the same"


def test_feature_extraction_segment():
    print("Short-term feature extraction")
    [fs, x] = audioBasicIO.read_audio_file(r"G:\pyAudioAnalysis\pytests\test_data\5_sec_wav.wav")
    mt, st, mt_names = MidTermFeatures.mid_feature_extraction(x, fs, 
                                                              1 * fs,
                                                              1 * fs,
                                                              0.05 * fs,
                                                              0.05 * fs)
    print("Mid-term feature matrix shape:", mt.shape)
    print("Short-term feature matrix shape:", st.shape)
    print("Mid-term feature names:", mt_names)
    assert mt.shape[1] == 5, "Wrong number of short-term windows"
    assert mt.shape[0] == len(mt_names),  "Number of features and feature " \
                                          "names are not the same"

if __name__ == "__main__":
    test_feature_extraction_short()
    test_feature_extraction_segment()
    print("All tests passed.")