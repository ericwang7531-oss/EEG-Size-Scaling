import numpy as np
import mne
raw_eeg_path = r"D:\SUSTech_Lab\PanDeng\脑电受试数据\LZN_EEGNet_raw.set"  # 替换为你的 EEG 数据文件路径

print("Loading EEG data from:", raw_eeg_path)
eeg = mne.io.read_raw_eeglab(raw_eeg_path, preload=True)
raw_data = eeg.get_data()
print("EEG data shape:", raw_data.shape)
print("Maximum voltage:", np.max(raw_data))
print("Minimum voltage:", np.min(raw_data))