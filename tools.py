"""包含项目通用的类型与函数"""
import os
import sys
import time
import torch
import ctypes
import random
import pathlib
import numpy as np
import pandas as pd
from datetime import datetime
from settings import Settings
from sklearn.preprocessing import StandardScaler

_winmm = ctypes.WinDLL('winmm')
_winmm.timeBeginPeriod(1)  # 请求 1 ms 计时器精度


class Logger(object):
    """用于记录控制台输出"""
    def __init__(self):
        s = Settings()
        path = get_path(
            is_file=True,
            path_spec='log',
            s_name=s.current_subject_name,
            s_turn=s.current_subject_turn,
            fr=s.bandpass_filter_frequency_range_list,
            subp_list=s.subprocess_to_run.keys()
        )
        self.terminal = sys.stdout
        self.log_file = open(path, 'w', encoding='utf-8')

    def write(self, message):
        """将消息写入控制台与文件"""
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # 强制立即写入磁盘

    def flush(self):
        """把缓存区里的数据立刻写进目的地"""
        self.terminal.flush()
        self.log_file.flush()


class RunTimer:
    """计时器, 初始化时便开始计时"""
    def __init__(self):
        self.start_time = time.time()  # 起始时间

    def stop(self, prompt:str, show_hours:bool=False, newline:bool=True) -> None:
        """
        停止计时并提示运行时间
        :param prompt: 提示语
        :param show_hours: 是否展示小时, 默认为False
        :param newline: 提示语结尾是否换行, 默认为True
        """
        runtime = time.time() - self.start_time  # 总运行时间
        if show_hours:
            hours = runtime // 3600  # 运行小时数
            minutes = runtime // 60 - hours * 60  # 运行分钟数
            seconds = round((runtime - hours * 3600 - minutes * 60), 1)  # 运行秒数
            text = prompt + str(hours) + 'h ' + str(minutes) + 'min ' + str(seconds) + 's.'
        else:
            minutes = runtime // 60
            seconds = round((runtime - minutes * 60), 1)
            text = prompt + str(minutes) + 'min ' + str(seconds) + 's.'
        if not newline:
            print(text, end='')
        else:
            print(text)


def precise_wait(t: float):
    """
    精确等待, 占用线程
    :param t: 等待时间, 单位为秒
    """
    start_time = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start_time
        remaining = t - elapsed
        if remaining <= 0: break
        if remaining > 0.002: time.sleep(0.001)  # 剩余时间大于2ms时, 用sleep让出CPU给其他线程
        else: pass  # 最后 2 ms 改用忙等，保证高精度终止


def set_random_seed():
    """设置随机种子, 确保可复现性"""
    seed = Settings().random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_path(
        path_spec:str='', is_file:bool=False,
        s_name:str|None=None, s_turn:int|list|None=None, obs:bool|None=None, fr:list|None=None, tr:list|None=None,
        net:str|None=None, subp_list:list|None=None
):
    """
    获取指定路径
    :param path_spec: 路径具体内容(
        'colormap', 'log', 'chan_loc', 'acc_table', 'comb_acc_table'
        'ori_eeg', 'ori_label',
        'processed_label', 'processed_eeg', 'bad_trial', 'ica_topo_map', 'trial_plot',
        'neu_rep', 'neu_rep_data',
        'training', 'loss_curve', 'acc_curve', 'optimize', 'best_hp', 'best_mw',
        'result', 'test_truth', 'test_pred', 'test_truth_pred', 'explanation', 'eeg_shap_heatmap'
        )
    :param is_file: 是否为文件路径(若为否则为文件夹路径)
    :param s_name: 受试名
    :param s_turn: 受试第几次实验
    :param obs: 是否使用观察期数据进行解码
    :param fr: 带通滤波范围, 一维(一个范围)或二维(多个范围)列表
    :param tr: 截取时间范围, 只能是一维列表
    :param net: 网络名称
    :param subp_list: 子进程列表
    """
    s = Settings()
    if path_spec == 'colormaps':
        return os.path.join(s.path_result, 'colormaps.mat')
    elif path_spec == 'acc_table' and fr is not None and net is not None:
        return os.path.join(s.path_result, f'Test_accuracy_[{fr[0]}-{fr[1]}]_{net}.xlsx')
    elif path_spec == 'comb_acc_table' and fr is not None and net is not None:
        return os.path.join(s.path_result, f'Test_accuracy_[{fr[0]}-{fr[1]}]_{net}_combined.xlsx')
    elif s_name is not None and s_turn is not None:
        path_name_turn = os.path.join(s_name, str(s_turn))  # 受试名与受试第几次实验
        path_ori_data = os.path.join(s.path_data, path_name_turn, 'Original data')
        if path_spec == 'log' and fr is not None and subp_list is not None:
            subp_names = '_'
            for subprocess in subp_list:
                if s.subprocess_to_run[subprocess]:
                    if subp_names != '_':
                        subp_names += f'-{subprocess}'
                    else:
                        subp_names += f'{subprocess}'
                    if subprocess == 'experiment':
                        subp_names += f'({s_name}{s_turn})'
                    elif subprocess == 'preprocessor':
                        for r_idx in range(len(fr)):
                            subp_names += f'[{fr[r_idx][0]}-{fr[r_idx][1]}]'
            if subp_names == '_':
                subp_names += 'no_subprocess'
            path = os.path.join(s.path_main, 'log')
            if is_file: path = os.path.join(path, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}{subp_names}.log')
            return path
        # region 原始数据路径
        elif path_spec == 'chan_loc':
            return os.path.join(s.path_data, path_name_turn, 'channel_location.sfp')
        elif path_spec == 'ori_eeg':
            path = os.path.join(path_ori_data, 'EEG')
            if is_file:
                # 直接在 EEG 文件夹下寻找对应的 .set 文件
                path = os.path.join(path, f'{s_name}.set')
            return path
        elif path_spec == 'ori_label':
            path = os.path.join(path_ori_data, 'Label')
            if is_file: path = os.path.join(path, f'{s_name}_{s_turn}.csv')
            return path
        # endregion
        elif path_spec == 'processed_label':
            path = os.path.join(s.path_data, path_name_turn, 'Processed label')
            if is_file: return os.path.join(path, f'{s_name}_{s_turn}.npy')
            return path
        elif obs is not None:
            path_img_obs_data = 'Processed data-Observe' if obs else 'Processed data-Imagine'  # 想象或观察路径
            path_img_obs_short = 'Observe' if obs else 'Imagine'
            if fr is not None:
                path_fr = f'[{fr[0]}-{fr[1]}]'
                path_eog = 'Regression artifact rejection' if s.remove_eog_method == 'reg' else 'ICA artifact rejection'
                # region 预处理路径
                path_processed_data = os.path.join(s.path_data, path_name_turn, path_img_obs_data, path_fr, path_eog)
                if path_spec == 'processed_eeg':
                    if is_file: return os.path.join(path_processed_data, f'{s_name}_{s_turn}.npy')
                    return path_processed_data
                elif path_spec == 'bad_trial':
                    return os.path.join(path_processed_data, f'{s_name}_{s_turn}_bad_trial.csv')
                elif path_spec == 'ica_topo_map':
                    return os.path.join(path_processed_data, 'ICA topography map')
                elif path_spec == 'trial_plot':
                    return os.path.join(path_processed_data, 'Trial time-domain plot')
                # endregion
                # region 神经表征数据路径
                path_neu_rep = os.path.join(s.path_data, path_name_turn, 'Neural representation', path_fr, path_eog)
                if path_spec == 'neu_rep':
                    return path_neu_rep
                elif path_spec == 'neu_rep_data':
                    return os.path.join(path_neu_rep, 'Data')
                # endregion
                elif tr is not None and net is not None:
                    path_tr = f'({tr[0]}-{tr[1]})'
                    # region 训练路径
                    path_training = os.path.join(s.path_training, path_name_turn, path_img_obs_short, path_fr, path_tr, path_eog, net)
                    if path_spec == 'training':
                        return path_training
                    elif path_spec == 'loss_curve':
                        return os.path.join(path_training, f'loss_curve.png')
                    elif path_spec == 'acc_curve':
                        return os.path.join(path_training, f'acc_curve.png')
                    elif path_spec == 'optimize':
                        return os.path.join(path_training, 'Optimizer training curve')
                    elif path_spec == 'best_hp':
                        return os.path.join(path_training, 'best_hyperparameters.csv')
                    elif path_spec == 'best_mw':
                        return os.path.join(path_training, f'best_model_weights.pth')
                    # endregion
                    # region 结果路径
                    path_result = os.path.join(s.path_result, path_name_turn, path_img_obs_short, path_fr, path_tr, path_eog, net)
                    if path_spec == 'result':
                        return path_result
                    elif path_spec == 'test_truth':
                        return os.path.join(path_result, f'{s_name}_test_truth.npy')
                    elif path_spec == 'test_pred':
                        return os.path.join(path_result, f'{s_name}_test_prediction.npy')
                    elif path_spec == 'test_truth_pred':
                        return os.path.join(path_result, f'{s_name}_test_truth_and_prediction.csv')
                    elif path_spec == 'explanation':
                        return os.path.join(path_result, f'{s_name}_explanation.npy')
                    elif path_spec == 'confusion_matrix':
                        return os.path.join(path_result, f'{s_name}_confusion_matrix.xlsx')
                    elif path_spec == 'eeg_shap_heatmap':
                        return os.path.join(path_result, 'EEG-SHAP heatmap')
                    elif path_spec == 'shap_topomap':
                        return os.path.join(path_result, 'SHAP topographic map')
                    # endregion

    raise AttributeError(f'Invalid or missing parameters.')


def load_dataset(subj_name:str, subj_turn:int|list, observe:bool, frange:list):
    """
    载入指定数据
    :param subj_name: 受试名
    :param subj_turn: 受试第几次实验
    :param observe: 是否使用观察期数据进行解码
    :param frange: 带通滤波范围
    """
    path_eeg = get_path(is_file=True, path_spec='processed_eeg', s_name=subj_name, s_turn=subj_turn, obs=observe, fr=frange)
    path_label = get_path(is_file=True, path_spec='processed_label', s_name=subj_name, s_turn=subj_turn, obs=observe)
    path_bad_trial = get_path(is_file=True, path_spec='bad_trial', s_name=subj_name, s_turn=subj_turn, obs=observe, fr=frange)
    eeg = np.load(path_eeg)                                      # 载入脑电数据
    label = np.load(path_label)                                  # 载入训练标签
    bad_trial = pd.read_csv(path_bad_trial).to_numpy(dtype=int)  # 载入坏段表
    return eeg, label, bad_trial


def standardize_eeg(train_eeg:np.ndarray, val_eeg:np.ndarray, test_eeg:np.ndarray|None=None):
    """
    对EEG数据进行标准化(Z-score)
    注意: 必须使用训练集的统计量来标准化验证集和测试集, 避免数据泄露
    :param train_eeg: 训练集EEG, 形状(n_train, n_chan, eeg_len)
    :param val_eeg: 验证集EEG, 形状(n_val, n_chan, eeg_len)
    :param test_eeg: 测试集EEG, 形状(n_test, n_chan, eeg_len)
    :return: 标准化后的数据 (保持原始形状)
    """
    n_train, n_chan, eeg_len = train_eeg.shape
    n_val = val_eeg.shape[0]

    train_eeg_flat = np.transpose(train_eeg, (1, 0, 2)).reshape(n_chan, -1).T  # (n_train, n_chan, eeg_len) -> (n_train, n_chan*eeg_len)
    val_eeg_flat = np.transpose(val_eeg, (1, 0, 2)).reshape(n_chan, -1).T

    scaler = StandardScaler()
    scaler.fit(train_eeg_flat)  # 仅在训练集上拟合

    train_eeg_scaled = scaler.transform(train_eeg_flat).T.reshape(n_chan, n_train, eeg_len).transpose(1, 0, 2)  # 标准化数据集并恢复原有形状
    val_eeg_scaled = scaler.transform(val_eeg_flat).T.reshape(n_chan, n_val, eeg_len).transpose(1, 0, 2)

    if test_eeg is not None:
        n_test = test_eeg.shape[0]
        test_eeg_flat = np.transpose(test_eeg, (1, 0, 2)).reshape(n_chan, -1).T
        test_eeg_scaled = scaler.transform(test_eeg_flat).T.reshape(n_chan, n_test, eeg_len).transpose(1, 0, 2)
        return train_eeg_scaled, val_eeg_scaled, test_eeg_scaled

    return train_eeg_scaled, val_eeg_scaled