import os
import mne
import numpy as np
import pandas as pd

# ================= 配置区 =================
# 1. 你的原始 fif 文件存放的根目录 (请根据你的实际情况修改)
# 假设你的文件命名规律是: subject1_large.fif, subject1_small.fif
fif_data_dir = r"D:\SUSTech_Lab\PanDeng\OMI Decoding I\src\data" 

# 2. 需要转换的被试列表和实验次数 (对应 settings.py 里的字典)
# 键为受试名，值为其 turn (实验次数) 的列表
subject_dict = {
                'WZWY': [
                    1,  # 范式版本: v1.0
                ],
                'ZEM': [
                    1,  # 范式版本: v1.0
                ],
                'LBB': [
                    1,  # 范式版本: v1.0
                ],
                'LZN': [
                    1,  # 范式版本: v1.0
                ],
                'HYY': [
                    1,  # 范式版本: v1.0
                ],
                'LHZ': [
                    1,  # 范式版本: v1.0
                ],
                'DTY': [
                    1,  # 范式版本: v1.0
                ],
                'LRH': [
                    1,  # 范式版本: v1.0
                ]
            }                                # 所有受试名和受试实验次数

# 3. 对应 settings.py 里的参数配置，用于严格还原路径结构
bandpass_range = [0.1, 30]  # 滤波频段
eog_method = 'ICA artifact rejection'  # 去眼电方式
# ==========================================

# 预计算一些固定路径字符串
path_main = os.getcwd()  # 获取当前项目根目录
path_data_root = os.path.join(path_main, 'data')
path_fr = f"[{bandpass_range[0]}-{bandpass_range[1]}]"

print("开始批量转换 fif 文件...")

for subj_name, turns in subject_dict.items():
    for subj_turn in turns:
        print(f"\n正在处理被试: {subj_name}, 实验次数: {subj_turn}")
        
        # --- 步骤 1: 读取 FIF 文件 ---
        # 这里的命名规则请根据你实际的文件名修改！
        # 例如你保存的是 "王梓文扬_1_large.fif"
        fif_path_large = os.path.join(fif_data_dir, f"{subj_name}_EEGNet_Trigger2_64chs-epo.fif")
        fif_path_small = os.path.join(fif_data_dir, f"{subj_name}_EEGNet_Trigger3_64chs-epo.fif")
        
        try:
            epochs_large = mne.read_epochs(fif_path_large, preload=True, verbose=False)
            epochs_small = mne.read_epochs(fif_path_small, preload=True, verbose=False)
        except FileNotFoundError as e:
            print(f"  [警告] 找不到对应的 fif 文件，跳过此轮次: {e}")
            continue

        # --- 步骤 2: 提取数据与生成标签 ---
        # .get_data() 返回形状: (trials, channels, times)
        data_large = epochs_large.get_data(copy=True)
        data_small = epochs_small.get_data(copy=True)
        
        # 假设：变大 (large) 标签设为 0，变小 (small) 标签设为 1
        # (请确保这里的标签设置与你的实验假设一致)
        label_large = np.zeros(data_large.shape[0])
        label_small = np.ones(data_small.shape[0])
        
        # 沿着 trial 维度 (axis=0) 合并
        eeg_combined = np.concatenate((data_large, data_small), axis=0)
        label_combined = np.concatenate((label_large, label_small), axis=0)
        
        print(f"  合并后 EEG 矩阵形状: {eeg_combined.shape}")
        print(f"  合并后 Label 矩阵形状: {label_combined.shape}")

        # --- 步骤 3: 严格按照 tools.py 构建目标路径并保存 ---
        # 1. 保存预处理 EEG 数据的路径
        # 对应 tools.py: data/名字/turn/Processed data-Imagine/[0.1-30]/ICA artifact rejection/
        eeg_save_dir = os.path.join(path_data_root, subj_name, str(subj_turn), 
                                    'Processed data-Imagine', path_fr, eog_method)
        os.makedirs(eeg_save_dir, exist_ok=True)
        eeg_save_path = os.path.join(eeg_save_dir, f"{subj_name}_{subj_turn}.npy")
        np.save(eeg_save_path, eeg_combined)
        print(f"  已保存 EEG -> {eeg_save_path}")

        # 2. 保存预处理 Label 数据的路径
        # 对应 tools.py: data/名字/turn/Processed label/
        label_save_dir = os.path.join(path_data_root, subj_name, str(subj_turn), 'Processed label')
        os.makedirs(label_save_dir, exist_ok=True)
        label_save_path = os.path.join(label_save_dir, f"{subj_name}_{subj_turn}.npy")
        np.save(label_save_path, label_combined)
        print(f"  已保存 Label -> {label_save_path}")
        
        # 3. 生成并保存空的 bad_trial.csv
        # tools.py 中的 load_dataset 强制要求读取坏段记录，否则会报错
        # 对应 tools.py 路径同 eeg_save_dir
        bad_trial_save_path = os.path.join(eeg_save_dir, f"{subj_name}_{subj_turn}_bad_trial.csv")
        # 创建一个只有列名没有数据的 DataFrame，转成 NumPy 时就是个空数组，不会干扰训练
        pd.DataFrame(columns=['bad_trial_index']).to_csv(bad_trial_save_path, index=False)
        print(f"  已保存空坏段记录 -> {bad_trial_save_path}")

print("\n全部转换完成！现在你可以去运行 main.py 启动 trainer 了。")