"""预处理EEG与轨迹数据"""
import os
import mne
import sys
import json
from mne import epochs
import numpy as np
import pandas as pd
from typing import List, cast
from settings import Settings
import matplotlib.pyplot as plt
from tools import set_random_seed, get_path


def offline_preprocess(subj_name:str, subj_turn:int, bad_chan_dict:dict, bp_fr:list):
    """
    预处理, 参数不再统一设置, 分别在每个代码段设置
    :param subj_name: 受试名
    :param subj_turn: 受试第几次实验
    :param bad_chan_dict: 坏导 (使用通道名称, 而非编号)
    :param bp_fr: 带通滤波器频段
    """

    s = Settings()

    # region EEG预处理
    # region 载入EEG数据
    print('Loading EEG data...')

    path_eeg_file = get_path(path_spec='ori_eeg', is_file=True, s_name=subj_name, s_turn=subj_turn)  # 原始EEG文件路径
    eeg = mne.io.read_raw_eeglab(input_fname=path_eeg_file, preload=True)                               # 载入原始EEG文件


    global_bad_chan_list = []  # 全局坏导列表
    block_bad_chan_list = []   # 局部(组块)坏导列表
    for bad_chan_idx, bad_chan in enumerate(bad_chan_dict):
        if bad_chan_dict[bad_chan] == 'all':
            global_bad_chan_list.append(bad_chan)
        else:
            block_bad_chan_list.append(bad_chan)

    eeg.info['bads'] = global_bad_chan_list  # 标记全局坏导(只有一部分组块损坏的通道不计入在内)

    print(f'    Original EEG path  : {path_eeg_file}')
    print(f'    Channel number     : {eeg.info["nchan"]}')
    print(f'    Sampling rate      : {int(eeg.info["sfreq"])} Hz')
    print(f'    Global bad channel{"s :" if len(eeg.info["bads"]) > 1 else " :"} {eeg.info["bads"]}')
    print(f'    Local bad channel{"s  :" if len(block_bad_chan_list) > 1 else "  :"} {block_bad_chan_list}')
    # endregion

    # region 电极定位
    print('Setting channel locations...')

    locs_info_path = r"D:\SUSTech_Lab\PanDeng\脑电受试数据\gtec_64+1_channels.loc"
    new_chan_names = np.loadtxt(locs_info_path, dtype=str, usecols=3)
    # 新旧导联映射并更新
    old_chan_names = eeg.info["ch_names"]
    chan_dict = {old_chan_names[i]: new_chan_names[i] for i in range(64)}
    eeg.rename_channels(chan_dict)
    #eeg.set_channel_types(s.non_eeg_channel)                        # 设置通道类型
    eeg.set_montage(mne.channels.read_custom_montage(locs_info_path))  # 导入位置文件

    montage = eeg.get_montage()
    montage = cast(mne.channels.DigMontage, montage)  # 提醒sb检查器这不是None类型, 不然有弱警告
    pos_dict = montage.get_positions()['ch_pos']      # 获取电极三维笛卡尔坐标
    path_chan_loc = get_path(path_spec='chan_loc', s_name=subj_name, s_turn=subj_turn)
    with open(path_chan_loc, 'w', encoding='utf-8') as f:  # 保存电极排布为.sfp格式, 可用于EEGLab
        for i, name in enumerate(eeg.ch_names):
            if name in pos_dict and pos_dict[name] is not None:
                x, y, z = pos_dict[name][0] * 100, pos_dict[name][1] * 100, pos_dict[name][2] * 100  # 转换为EEGLab常用的厘米
            else:
                x, y, z = 0.0, 0.0, 0.0  # EOG, ECG或其他没有位置的通道, 坐标设为0
            f.write(f'{name}\t{x}\t{y}\t{z}\n')

    #used_chan_type = ['eeg']
    #if s.use_eog_channel: used_chan_type.append('eog')
    #if s.use_ecg_channel: used_chan_type.append('ecg')
    #eeg.pick(used_chan_type)  # 选择需要的通道类型, 不需要的被丢弃

    eeg_chan_num = len(montage.ch_names)
    print(f'    Location{"s" if eeg_chan_num > 1 else ""} of {eeg_chan_num} EEG channel{"s have" if eeg_chan_num > 1 else " has"} been loaded.')
    #print(f'    EOG electrode{"s : " if s.use_eog_channel and len(eeg.copy().pick("eog").ch_names) > 1 else "  :  "}{eeg.copy().pick("eog").ch_names if s.use_eog_channel else "Not used"}.')
    #print(f'    ECG electrode{"s : " if s.use_ecg_channel and len(eeg.copy().pick("ecg").ch_names) > 1 else "  :  "}{eeg.copy().pick("ecg").ch_names if s.use_ecg_channel else "Not used"}.')
    # endregion

    # region FIR滤波
    print('FIR filtering...')

    eeg.notch_filter(freqs=s.notch_filter_frequency, notch_widths=s.notch_filter_width, method='fir')
    eeg.filter(l_freq=bp_fr[0], h_freq=bp_fr[1], method='fir')

    print(f'    Notch filter    : {s.notch_filter_frequency-(s.notch_filter_width+1)/2}-{s.notch_filter_frequency+(s.notch_filter_width+1)/2} Hz.')
    print(f'    Bandpass filter : {bp_fr[0]}-{bp_fr[1]} Hz.')
    # endregion

    # region 插值全局坏导
    print(f'Interpolating global bad channel{"s" if len(global_bad_chan_list) > 1 else ""}...')

    eeg_bad_chan = [ch for ch in global_bad_chan_list if ch in eeg.copy().pick('eeg').ch_names]  # EEG坏导
    #non_eeg_bad_chan = [ch for ch in global_bad_chan_list if ch not in eeg_bad_chan]             # 非EEG坏导
    if len(eeg.info['bads']) > 0:
        eeg.interpolate_bads(reset_bads=True, mode='accurate', method={'eeg': 'spline'})  # 插值损坏的EEG通道
    #    eeg.drop_channels(non_eeg_bad_chan)                                               # 删除损坏的EOG/ECG通道

    print(f'    {len(eeg_bad_chan)} EEG channel{"s have" if len(eeg_bad_chan) > 1 else " has"} been interpolated{" : " + str(eeg_bad_chan) if len(eeg_bad_chan) > 0 else ""}.')
    #print(f'    {len(non_eeg_bad_chan)} non-EEG channel{"s have" if len(non_eeg_bad_chan) > 1 else " has"} been deleted{" : " + str(non_eeg_bad_chan) if len(non_eeg_bad_chan) > 0 else ""}.')
    # endregion

    # region 重参考
    print('Re-referencing...')

    eeg.set_eeg_reference(ref_channels=s.rereference_channel, ch_type='eeg')
    if s.rereference_channel == "average":
        print('    Average reference applied.')
    else:
        print(f'    {" Reference channels " if len(s.rereference_channel) > 1 else "  Reference channel "}: {s.rereference_channel}')
    print(f'    EEG maximum voltage : {np.max(eeg.get_data(picks="eeg")*1e6):10.2f} uV / {np.max(eeg.get_data(picks="eeg")):5.2f} V;')
    print(f'    EEG average voltage : {np.mean(eeg.get_data(picks="eeg")*1e6):10.2f} uV / {np.mean(eeg.get_data(picks="eeg")):5.2f} V;')
    print(f'    EEG minimum voltage : {np.min(eeg.get_data(picks="eeg")*1e6):10.2f} uV / {np.min(eeg.get_data(picks="eeg")):5.2f} V.')
    #if s.use_eog_channel:
    #    print(f'    EOG maximum voltage : {np.max(eeg.get_data(picks="eog") * 1e6):10.2f} uV / {np.max(eeg.get_data(picks="eog")):5.2f} V;')
    #    print(f'    EOG average voltage : {np.mean(eeg.get_data(picks="eog") * 1e6):10.2f} uV / {np.mean(eeg.get_data(picks="eog")):5.2f} V;')
    #    print(f'    EOG minimum voltage : {np.min(eeg.get_data(picks="eog") * 1e6):10.2f} uV / {np.min(eeg.get_data(picks="eog")):5.2f} V.')
    #if s.use_ecg_channel:
    #    print(f'    ECG maximum voltage: {np.max(eeg.get_data(picks="ecg") * 1e6):10.2f} uV / {np.max(eeg.get_data(picks="ecg")):5.2f} V;')
    #    print(f'    ECG average voltage : {np.mean(eeg.get_data(picks="ecg") * 1e6):10.2f} uV / {np.mean(eeg.get_data(picks="ecg")):5.2f} V;')
    #    print(f'    ECG minimum voltage : {np.min(eeg.get_data(picks="ecg") * 1e6):10.2f} uV / {np.min(eeg.get_data(picks="ecg")):5.2f} V.')
    # endregion

    # region 分离组块
    print('Extracting blocks...')

    events, event_dict = mne.events_from_annotations(eeg)    
    # 【新增】强行打印出你数据里真实的 Trigger 字典，看看它们到底叫什么名字！
    print(f"    [DEBUG] 当前数据中真实存在的标签字典: {event_dict}")
    
    # 动态获取 trigger 的内部代号
    trig1_id = event_dict.get('Trigger 1')  # 基线开始 (作为 Block 的起点)
    trig5_id = event_dict.get('Trigger 5')  # 组间大休息 (作为 Block 的终点)

    # 【新增】安全锁：如果找不到标签 1，直接报错提醒，而不是继续往下跑导致除以 0
    if trig1_id is None:
        print("    [ERROR] 找不到名为 'Trigger 1' 的标签！请查看上方 [DEBUG] 打印出的字典，把代码里的 'Trigger 1' 改成你字典里真正的名字。")
        sys.exit()


    block_start_trig = []
    block_end_trig = []

    # 使用状态机逻辑，精准提取每个 Block 的头尾时间点
    in_block = False
    for event in events:
        if not in_block and event[2] == trig1_id:
            # 当不在 block 中，且遇到第一个 trigger 1 时，记录为当前 block 的物理起点
            block_start_trig.append(event[0])
            in_block = True
        elif in_block and event[2] == trig5_id:
            # 当在 block 中，遇到 trigger 5 时，记录为当前 block 的物理终点
            block_end_trig.append(event[0])
            in_block = False

    # 防止由于实验意外终止，导致最后一个 Block 缺少 trigger 5 的情况
    if len(block_start_trig) > len(block_end_trig):
        block_end_trig.append(events[-1, 0])

    block_list = []  # 存储所有组块
    block_len = []   # 记录组块长度

    for block_idx in range(len(block_start_trig)):
        # 加上 s.time_block_extra 余量，并除以采样率转换为秒
        block_start_time = block_start_trig[block_idx] / eeg.info['sfreq'] - s.time_block_extra
        block_end_time = block_end_trig[block_idx] / eeg.info['sfreq'] + s.time_block_extra
        
        # 安全锁：防止加减 extra 余量后，时间超出原始数据的总长度导致 MNE 截取报错
        block_start_time = max(0, block_start_time)
        block_end_time = min(eeg.times[-1], block_end_time)

        block = eeg.copy().crop(tmin=block_start_time, tmax=block_end_time)

       
        block_eeg_bad_chan = []
        block_non_eeg_bad_chan = []
        if len(block_bad_chan_list) > 0:  # 处理局部坏导
            for block_bad_chan in block_bad_chan_list:
                start_block = bad_chan_dict[block_bad_chan][0]  # 坏导开始组块
                end_block = bad_chan_dict[block_bad_chan][1]    # 坏导结束组块
                if start_block <= block_idx + 1 <= end_block:
                    if block_bad_chan in block.copy().pick('eeg').ch_names:  # 插值EEG坏导
                        block.info['bads'] = [block_bad_chan]
                        block.interpolate_bads(reset_bads=True, mode='accurate', method={'eeg': 'spline'})
                        block_eeg_bad_chan.append(block_bad_chan)
                    else:  # 删除非EEG坏导
                        block.drop_channels(block_bad_chan)
                        block_non_eeg_bad_chan.append(block_bad_chan)

        block_list.append(block)
        block_len.append(block_end_time - block_start_time)

        print(f'    Block {block_idx + 1:2}, '
              f'{"no" if len(block_eeg_bad_chan) == 0 else ""} '
              f'channel{"s" if len(block_eeg_bad_chan) > 1 else ""} '
              f'{"was " if len(block_eeg_bad_chan) == 0 else ""}'
              f'interpolated{"" if len(block_eeg_bad_chan) == 0 else ": " + str(block_eeg_bad_chan)}, '
              f'{"no" if len(block_non_eeg_bad_chan) == 0 else ""} '
              f'channel{"s" if len(block_non_eeg_bad_chan) > 1 else ""} '
              f'{"was " if len(block_non_eeg_bad_chan) == 0 else ""}'
              f'deleted{"" if len(block_non_eeg_bad_chan) == 0 else ": " + str(block_non_eeg_bad_chan)}.')

    print(f'    {len(block_start_trig)} block{"s were" if len(block_start_trig) > 1 else " was"} extracted, '
          f'average length: {sum(block_len) / len(block_len):.1f}s.')
    # endregion

    # region 去除伪迹
    print('Removing artifacts...')

    if s.remove_eog_method == 'reg':
        path_ica_topo = ''
    else:
        path_ica_topo = get_path(path_spec='ica_topo_map', s_name=subj_name, s_turn=subj_turn, obs=False, fr=bp_fr)     # ICA组分图路径
    path_neural_rep = get_path(path_spec='neu_rep', s_name=subj_name, s_turn=subj_turn, obs=False, fr=bp_fr)          # 神经表征路径
    path_neural_rep_data = get_path(path_spec='neu_rep_data', s_name=subj_name, s_turn=subj_turn, obs=False, fr=bp_fr)  # 用于神经表征的数据路径

    if s.remove_eog_method == 'reg':
        pass
    else:
        os.makedirs(path_ica_topo, exist_ok=True)
    os.makedirs(path_neural_rep, exist_ok=True)
    os.makedirs(path_neural_rep_data, exist_ok=True)

    block_list_raw = []  # 保存一份未经过伪迹去除的数据, 用于后续绘图时进行对比
    for block_idx, block in enumerate(block_list):  # 这里的block直接指向block_list[block_idx], 对block的修改会作用于block_list
        block_list_raw.append(block.copy())         # 复制数据
        if s.remove_eog_method == 'reg':                                                                           # 回归法去眼电
            eog_reg_model = mne.preprocessing.EOGRegression(picks='eeg', picks_artifact='eog').fit(block)
            block_list[block_idx] = eog_reg_model.apply(block)
        else:                                                                                                      # ICA去眼电
            block_copy = block.copy()                                                                              # 复制一份数据, 用于高通滤波
            block_copy.filter(l_freq=1, h_freq=None, method='fir')                                                 # ICA对低频漂移敏感, 须先进行高通滤波

            ica = mne.preprocessing.ICA(method=s.ica_method, random_state=s.random_seed)                           # 初始化ICA模型
            ica.fit(block_copy, picks='eeg')                                                                       # 拟合ICA

            # 1. 提取虚拟眼电通道 FP1 的数据与所有 ICA 组分的波形数据
            eog_data = block_copy.get_data(picks='FP1').squeeze()
            ica_sources = ica.get_sources(block_copy).get_data()
            
            # 2. 计算每个 ICA 组分与 FP1 的皮尔逊相关系数
            corrs = np.array([np.corrcoef(eog_data, comp)[0, 1] for comp in ica_sources])
            
            # 3. 根据 settings.py 设定的测量方式得出最终得分 (当前为 'correlation')
            scores_val = (corrs - np.mean(corrs)) / np.std(corrs) if s.ica_measure == 'zscore' else corrs
                
            # 4. 找到绝对值大于设定阈值 (当前为 0.5) 的组分索引
            bad_idx = np.where(np.abs(scores_val) > s.ica_threshold)[0]
            
            # 5. 完美复刻 MNE 底层逻辑：按得分绝对值从大到小排序，确保后续截取时优先剔除相关性最强的
            sorted_bad_idx = bad_idx[np.argsort(np.abs(scores_val[bad_idx]))[::-1]]
            
            eog_indices = sorted_bad_idx.tolist()  # 导出为原本要求的列表格式
            scores = [scores_val]                  # 包装成原生的列表格式，防止后续打印和绘图报错
            # ---------- 手动 ICA 眼电匹配模块 结束 ----------

            eog_indices: List[int]                                                                                 # MNE似乎没有指定清楚返回类型, 所以通过类型注解, 避免IDE产生警告
            scores: List[np.ndarray]
            num_eog_artifact = len(eog_indices)                                                                    # 标记为眼电伪迹的成分数量

            if num_eog_artifact > 0 and num_eog_artifact > s.ica_max_prune_amount:                                 # 限制去除的组分数量 (MNE会按score的绝对值大小, 降序排列eog_indices)
                eog_indices = eog_indices[:s.ica_max_prune_amount]
                print(f'    {num_eog_artifact} components in block {int(block_idx+1):2} were labeled as eye artifacts, '
                      f'only the top {s.ica_max_prune_amount} components with the highest correlation were removed.')

            ica.exclude = eog_indices                                                                              # 标记要去除的组分

            topo_plots = ica.plot_components(show=False)                                                           # 保存ICA组分图(包含将要去除的伪迹)
            if not isinstance(topo_plots, list): topo_plots = [topo_plots]                                         # 若组分较少, 只有一张图, 则将其转换为列表便于后续操作
            for plot_idx, topo in enumerate(topo_plots):
                topo.savefig(os.path.join(path_ica_topo, f'B{block_idx + 1}C{plot_idx + 1}'), dpi=600)
            plt.close('all')
            ica.apply(block)                                                                                       # 去除组分

            print(f'    Block {int(block_idx+1):2} has {len(scores[0]):3} component{"s" if len(scores[0]) > 1 else ""}, '
                  f'{len(eog_indices):2} of them {"were " if len(eog_indices) > 1 else "was "}labeled as eye artifact{"s " if len(eog_indices) > 1 else "   "}'
                  f'{": "+str(eog_indices) if len(eog_indices) > 0 else ""}.')

        if bp_fr == s.neural_representation_target_band:  # 保存组块数据, 用于神经表征
            block.save(
                fname=os.path.join(path_neural_rep_data, f'B{int(block_idx + 1)}_raw.fif'),
                fmt='double', overwrite=True
            )
    print(f'    All blocks have been saved to : {path_neural_rep_data}')
    # endregion

    path_processed_label = get_path(path_spec='processed_label', s_name=subj_name, s_turn=subj_turn)  # 训练标签路径
    os.makedirs(path_processed_label, exist_ok=True)
    for observe in [False]:
        # region 创建路径
        path_processed_eeg = get_path(path_spec='processed_eeg', s_name=subj_name, s_turn=subj_turn, obs=observe, fr=bp_fr)  # 处理后EEG路径
        path_trial_plot = get_path(path_spec='trial_plot', s_name=subj_name, s_turn=subj_turn, obs=observe, fr=bp_fr)        # 时域信号图路径

        os.makedirs(path_processed_eeg, exist_ok=True)
        os.makedirs(path_trial_plot, exist_ok=True)
        # endregion

        # region 提取试次
        print(f'Extracting trials ({"Observe" if observe else "Imagine"})...')

        total_num_imag_trig = 0
        total_num_epoch = 0
        total_num_bad_trial = 0
        epoched_block_list = []  # 记录所有提取过试次的组块(已剔除伪迹)

        target_events = []

        path_bad_trial_csv = os.path.join(path_processed_eeg, f'{subj_name}_{subj_turn}_bad_trial.csv')         # 坏段记录表路径
        pd.DataFrame(columns=['block_idx', 'trial_idx']).to_csv(path_bad_trial_csv, index=False, mode='w')      # 创建坏段记录表 (覆盖原有)

        path_eeg_label_save = os.path.join(path_processed_label, f'{subj_name}_{subj_turn}_from_EEG.csv')       # 保存从EEG数据中提取出来的训练标签
        if not observe: pd.DataFrame(columns=[s.name_csv_columns[0]]).to_csv(path_eeg_label_save, index=False)  # 只保存一次

        for block_idx, (block, block_raw) in enumerate(zip(block_list, block_list_raw)):
            block_events, event_dict = mne.events_from_annotations(block)                # 获取标签列表
            imag_large_id = event_dict.get('Trigger 2')                        # 想象标签ID
            imag_small_id = event_dict.get('Trigger 3')                        # 想象标签ID

            num_imag_large = len(block_events[block_events[:, 2] == imag_large_id])  # 想象大标签数量
            num_imag_small = len(block_events[block_events[:, 2] == imag_small_id])  # 想象小标签数量
            num_imag_trig = num_imag_large + num_imag_small
            total_num_imag_trig += num_imag_trig

            target_events = []  # 记录所有想象标签
            for event_idx, event in enumerate(block_events):
                if event[2] in [imag_large_id, imag_small_id]:
                    target_events.append(event)
            target_events = np.array(target_events, dtype=int)


            epoched_block = mne.Epochs(      # 提取试次(剔除伪迹的数据)
                block, target_events, event_id={"imag_large": imag_large_id, "imag_small": imag_small_id},
                tmin=-0.5,tmax=5.0,
                baseline= (-0.5, 0), # 基线范围, 注意这里不会删除基线时间段
                picks='eeg', preload=True
            )
            epoched_block_raw = mne.Epochs(  # 提取试次(未剔除伪迹的数据, 参数与上面的epoched_block完全一样)
                block_raw, target_events, event_id={"imag_large": imag_large_id, "imag_small": imag_small_id},
                tmin=-0.5, tmax=5.0,
                baseline=(-0.5, 0),
                picks='eeg', preload=True
            )

            epoched_block_list.append(epoched_block)
            total_num_epoch += len(epoched_block)  # 记录总试次数量

            data_clean = epoched_block.copy().pick('eeg').crop(           # 获取去除伪迹的EEG数据(不包含基线)
                tmin=0, tmax=5).get_data(copy=True)
            data_raw = epoched_block_raw.copy().pick('eeg').crop(         # 获取未去除伪迹的EEG数据
                tmin=0, tmax=5).get_data(copy=True)
            plot_times = epoched_block.copy().pick('eeg').crop(           # 绘图时间点
                tmin=0, tmax=5).times
            ch_names = epoched_block.copy().pick('eeg').info['ch_names']  # 通道名称

            bad_trial = np.any(np.abs(data_clean) > s.reject_threshold, axis=(1, 2))  # 排查每个试次的幅值是否超过阈值
            bad_trial_indices = np.where(bad_trial)[0]
            total_num_bad_trial += len(bad_trial_indices)
            if len(bad_trial_indices) > 0:
                df_bad = pd.DataFrame({'block_idx': block_idx + 1, 'trial_idx': bad_trial_indices + 1})
                df_bad.to_csv(path_bad_trial_csv, mode='a', header=False, index=False)

            offsets = np.array(np.arange(len(ch_names) - 1, -1, -1) * s.epoch_plot_scale)                    # 用于分开每个通道, 免得挤在一起
            for epoch_idx in range(len(epoched_block)):                                                      # 绘制每个试次的时域信号并保存
                fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
                ax.plot(plot_times, data_raw[epoch_idx].T + offsets, color='red', linewidth=0.5)       # 绘制未去除伪迹的数据(红色)
                ax.plot(plot_times, data_clean[epoch_idx].T + offsets, color='black', linewidth=0.5)   # 绘制去除伪迹的数据(黑色)
                ax.set_xlim(left=-s.time_observe if observe else 0, right=0 if observe else s.time_imagine)  # 设置X轴范围
                ax.set_ylim(-1.5 * s.epoch_plot_scale, offsets[0] + 1.5 * s.epoch_plot_scale)                # 图框上下留白
                ax.set_yticks(offsets)                                                                       # 设置通道间隔
                ax.set_yticklabels(ch_names, fontsize=8)                                                     # 标出通道名称
                plt.tight_layout()                                                                           # 紧凑视图
                fig.savefig(os.path.join(path_trial_plot, f'B{block_idx + 1}T{epoch_idx + 1}.png'))          # 保存图片
                plt.close(fig)                                                                               # 释放内存

            print(f'    Block {block_idx+1:2}')
            print(f'        Data length                               : {block.n_times}')
            print(f'        Number of triggers (total)                : {len(block_events)}')
            print(f'        Number of triggers (imagine large & small): {num_imag_trig}')
            print(f'        Number of extracted trials                : {len(epoched_block)}')
            print(f'        Number of rejected trials                 : {len(bad_trial_indices)}')

        combined_epochs = mne.concatenate_epochs(epoched_block_list)                             # 合并所有试次
        combined_epochs.crop(tmin=0, tmax=5.0)                                    # 删除基线数据(想象数据)

        print(f'    Extracted {total_num_epoch} trials in total, {total_num_bad_trial} of them were rejected.')
        print(f'    Bad trial record sheet created at : {path_bad_trial_csv}')
        # endregion

        # region 降采样
        print('Resampling...')

        ori_sfreq = combined_epochs.info["sfreq"]
        combined_epochs.resample(s.new_sampling_rate)

        print(f'    Data resampled from {ori_sfreq:.0f} Hz to {s.new_sampling_rate} Hz.')
        # endregion

        # region 保存EEG数据
        print('Saving EEG data...')

        path_eeg_save = os.path.join(path_processed_eeg, f'{subj_name}_{subj_turn}.npy')
        final_data = combined_epochs.get_data(picks='eeg')
        np.save(path_eeg_save, final_data)

        print(f'    Data saved to    : {path_eeg_save}.')
        print(f'    Final data shape : {final_data.shape}.')
        # endregion
    # endregion

    # region 标签预处理
    print('Extracting labels directly from triggers...')
    
    # 重新获取全局事件字典，以防局部变量丢失
    _, global_event_dict = mne.events_from_annotations(eeg)
    imag_large_id_global = global_event_dict.get('Trigger 2')  # 想象大标签ID
    
    # 从剔除坏段后的纯净事件列表中拿出所有真实的事件ID
    final_events = combined_epochs.events[:, 2]
    
    # 将 2(变大) 映射为网络需要的 1，其余的(变小/3) 映射为 0
    label_npy = np.where(final_events == imag_large_id_global, 1, 0)

    print(f'    Shape of final clean labels : {label_npy.shape}')
    
    # 保存原生的完美对齐标签
    print('Saving labels')
    path_label_save = os.path.join(path_processed_label, f'{subj_name}_{subj_turn}')
    np.save(path_label_save + '.npy', label_npy)
    pd.DataFrame(label_npy).to_csv(path_label_save + '.csv', index=False, header=False)
    print(f'    Labels have been saved to : {path_label_save}')
    # endregion


if __name__ == '__main__':

    set_random_seed()  # 设置随机种子

    subject_name = sys.argv[1]                        # 受试名
    subject_turn = int(sys.argv[2])                   # 受试第几次实验
    bad_channel_dictionary = json.loads(sys.argv[3])  # 坏导名称字典
    bandpass_filter_range = json.loads(sys.argv[4])   # 带通滤波频段
    print(f'--------------------'
          f'Preprocessing: '
          f'{subject_name} '
          f'{subject_turn} '
          f'[{bandpass_filter_range[0]}-{bandpass_filter_range[1]}]'
          f'--------------------')

    if Settings().mne_shut_up:
        mne.set_log_level('ERROR')  # 关闭MNE的输出, 不用在每个函数中使用verbose=0

    offline_preprocess(
        subj_name=subject_name,
        subj_turn=subject_turn,
        bad_chan_dict=bad_channel_dictionary,
        bp_fr=bandpass_filter_range
    )
