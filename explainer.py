"""解释模型的输出"""
import os
import sys
import json
import numpy as np
import pandas as pd
import shap.explainers
from networks import *
from sklearn.model_selection import train_test_split
from tools import set_random_seed, load_dataset, standardize_eeg, get_path


class WrappedModel(nn.Module):
    """一些模型有多输出(如DBConformer返回(fused, out)), 因此在解释这类模型前要对其进行包装, 确保只有一个输出(预测标签)"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        """重写模型输出"""
        _, out = self.model(x)
        return out


if __name__ == '__main__':

    # region 初始化
    set_random_seed()  # 设置随机种子

    S = Settings()                                   # 获取参数
    device = torch.device(S.training_device)         # 训练设备
    subject_name = sys.argv[1]                       # 受试名
    subject_turn = int(sys.argv[2])                  # 受试第几次实验(在S.concat_turns为真且受试实验次数大于1的情况下, 这个参数无效)
    if S.concat_turns and len(S.subject_to_process_dictionary[subject_name]) > 1:
        subject_turn = S.subject_to_process_dictionary[subject_name]
    is_observe = bool(int(sys.argv[3]))              # 是否使用观察期数据进行解码
    bandpass_filter_range = json.loads(sys.argv[4])  # 带通滤波频段
    data_time_range = json.loads(sys.argv[5])        # 所需数据时间范围
    network_name = sys.argv[6]                       # 网络名
    print(f'--------------------'
          f'Model explanation: '
          f'{subject_name} '
          f'{subject_turn} '
          f'{"Observe" if is_observe else "Imagine"} '
          f'[{bandpass_filter_range[0]}-{bandpass_filter_range[1]}] '
          f'({data_time_range[0]}-{data_time_range[1]}) '
          f'{network_name}'
          f'--------------------')

    path_training = get_path(
        path_spec='training',
        is_file=False,
        s_name=subject_name,
        s_turn=subject_turn,
        obs=is_observe,
        fr=bandpass_filter_range,
        tr=data_time_range,
        net=network_name
    )

    os.makedirs(path_training, exist_ok=True)
    # endregion

    # region 载入数据集
    print('Loading dataset...')

    eeg, label, num_deleted_trial = [], [], 0
    for turn in (subject_turn if isinstance(subject_turn, list) else [subject_turn]):
        try:
            eeg_turn, label_turn, bad_trial = load_dataset(subject_name, turn, is_observe, bandpass_filter_range)
        except FileNotFoundError:
            print(f'    Missing data : {subject_name}, '
                  f'{turn}, '
                  f'{"observe" if is_observe else "imagine"}, '
                  f'[{bandpass_filter_range[0]}-{bandpass_filter_range[1]}].')
            continue
        bad_trial_indices = [
            int((bad_trial[idx, 0] - 1) * S.num_trial_subject_turn[subject_name][str(turn)] + (bad_trial[idx, 1] - 1))
            for idx in range(bad_trial.shape[0])
        ]                                  # 计算坏段索引
        num_deleted_trial += len(bad_trial_indices)                # 坏段数量
        eeg_turn = np.delete(eeg_turn, bad_trial_indices, axis=0)  # 删除坏段
        label_turn = np.delete(label_turn, bad_trial_indices, axis=0)
        eeg_turn = eeg_turn[:, :,
        int(S.new_sampling_rate * data_time_range[0]):int(S.new_sampling_rate * data_time_range[1])]                                   # 截取所需时间段
        eeg.append(eeg_turn)
        label.append(label_turn)
    if eeg == [] or label == []:
        print(f'    No available data for {subject_name}, '
              f'{"observe" if is_observe else "imagine"}, '
              f'[{bandpass_filter_range[0]}-{bandpass_filter_range[1]}], '
              f'subprocess ended.')
        exit()
    eeg = np.concatenate(eeg, axis=0)
    label = np.concatenate(label, axis=0)

    print(f'    Dataset shape : {eeg.shape}, {label.shape}.')
    print(f'    {num_deleted_trial} bad trial{"s have" if num_deleted_trial > 1 else " has"} been deleted.')
    # endregion

    # region 划分数据集
    print('Splitting dataset...')

    training_eeg, test_eeg, training_label, test_label = train_test_split(
        eeg, label, test_size=S.test_ratio, shuffle=True, random_state=S.random_seed
    )  # 划分出测试集作为带解释数据
    train_eeg, validation_eeg, _, _ = train_test_split(
        training_eeg, training_label, test_size=S.test_ratio, shuffle=True, random_state=S.random_seed
    )  # 划分出训练集作为参考数据, 不使用验证集
    train_eeg, validation_eeg, test_eeg = standardize_eeg(train_eeg, validation_eeg, test_eeg)  # 标准化数据集
    del training_eeg, training_label, validation_eeg
    background = torch.tensor(train_eeg, dtype=torch.float32).to(device)
    target = torch.tensor(test_eeg, dtype=torch.float32).to(device)

    print(f'    Train EEG shape : {train_eeg.shape}.')
    print(f'    Test EEG shape  : {test_eeg.shape}.')
    # endregion

    # region 载入模型
    print('Loading model...')

    path_best_hyperparameters = get_path(
        path_spec='best_hp',
        is_file=True,
        s_name=subject_name,
        s_turn=subject_turn,
        obs=is_observe,
        fr=bandpass_filter_range,
        tr=data_time_range,
        net=network_name
    )
    path_best_model_weights = get_path(
        path_spec='best_mw',
        is_file=True,
        s_name=subject_name,
        s_turn=subject_turn,
        obs=is_observe,
        fr=bandpass_filter_range,
        tr=data_time_range,
        net=network_name
    )

    best_hyperparameters = pd.read_csv(path_best_hyperparameters).iloc[0].to_dict()  # 载入最佳超参
    best_model_weights = torch.load(path_best_model_weights, map_location=device)    # 载入最佳权重

    if network_name == 'EEGNet_LSTM':
        network_hyperparameter = {
            'dropout_rate': best_hyperparameters['dropout_rate'],
            'block1_kernel1_size': best_hyperparameters['block1_kernel1_size'],
            'block1_kernel1_num': best_hyperparameters['block1_kernel1_num'],
            'lstm_unit_num': best_hyperparameters['lstm_unit_num']
        }
        network = EEGNet_LSTM(n_input=train_eeg.shape[1], n_output=2, custom_para=network_hyperparameter)  # 使用最佳超参构建优化器
    elif network_name == 'DBConformer':
        network_hyperparameter = {
            'patch_size_ratio': best_hyperparameters['patch_size_ratio'],
        }
        network = DBConformer(
            ch_num=train_eeg.shape[1],
            time_sample_num=train_eeg.shape[2],
            patch_size=int(train_eeg.shape[2] * network_hyperparameter['patch_size_ratio']),
            n_classes=2
        )
    else:
        print(f'    Model "{network_name}" is not defined, subprocess ended.')
        exit()

    network.load_state_dict(best_model_weights)
    if network_name == 'DBConformer': network = WrappedModel(network)  # 包装DBConformer
    network.to(device)
    network.eval()

    print(f'    Best hyperparameters loaded from : {path_best_hyperparameters}')
    print(f'    Best model weights loaded from   : {path_best_model_weights}')
    # endregion

    # region 计算SHAP值
    print('Calculating SHAP...')

    explainer = shap.explainers.GradientExplainer(
        model=network,
        data=background,
        batch_size=train_eeg.shape[0]
    )
    explanation = explainer(target).values
    explanation = np.squeeze(explanation)
    path_explanation = get_path(
        path_spec='explanation',
        is_file=True,
        s_name=subject_name,
        s_turn=subject_turn,
        obs=is_observe,
        fr=bandpass_filter_range,
        tr=data_time_range,
        net=network_name
    )
    np.save(path_explanation, explanation)

    print(f'    SHAP calculated, shape   : {explanation.shape}.')
    print(f'    Explanation was saved to : {path_explanation}.')
    # endregion
