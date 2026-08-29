"""训练模型"""
import os
import sys
import copy
import json
import optuna
import warnings
import numpy as np
import pandas as pd
from networks import *
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from tools import RunTimer, set_random_seed, load_dataset, standardize_eeg, get_path


def epoch_train(m, dl, opti, crit, dev):
    """
    训练一个轮次
    :param m: 模型
    :param dl: 数据生成器
    :param opti: 优化器
    :param crit: 损失函数
    :param dev: 训练使用的硬件设备
    :return 训练损失, 训练平均绝对误差, 训练准确率
    """
    m.train()
    loss, correct, total = 0.0, 0, 0
    for x, y in dl:
        x = x.to(dev, dtype=torch.float)
        y = y.to(dev, dtype=torch.long)
        opti.zero_grad()
        outputs = m(x)
        if isinstance(outputs, tuple):
            outputs = outputs[1]  # DBConformer会传回两个变量
        batch_loss = crit(outputs, y)
        batch_loss.backward()
        opti.step()
        loss += batch_loss.item() * x.size(0)  # x.size(0)就是批次大小, 乘以批次大小以便得到每个样本的损失之和
        _, pred = torch.max(outputs, 1)  # 取概率最大的类别
        total += y.size(0)
        correct += (pred == y).sum().item()
    loss = loss / len(dl.dataset)
    acc = correct / total
    return loss, acc


def epoch_evaluate(m, dl, crit, dev):
    """
    评估一个轮次
    :param m: 模型
    :param dl: 数据生成器
    :param crit: 损失函数
    :param dev: 训练使用的硬件设备
    :return: 验证损失, 验证平均绝对误差, 验证准确率, 真值, 预测值
    """
    m.eval()  # 调至评估模式, 避免影响模型
    loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():  # 不更新参数
        for x, y in dl:
            x = x.to(dev, dtype=torch.float)
            y = y.to(dev, dtype=torch.long)
            outputs = m(x)
            if isinstance(outputs, tuple):
                outputs = outputs[1]  # DBConformer会传回两个变量
            batch_loss = crit(outputs, y)
            loss += batch_loss.item() * x.size(0)  # x.size(0)就是批次大小, 乘以批次大小以便得到每个样本的损失之和

            _, pred = torch.max(outputs, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()

            y_true.append(y.cpu().numpy())
            y_pred.append(torch.softmax(outputs, dim=1).cpu().numpy())  # 有torch.no_grad()就不需要detach()了
    loss = loss / len(dl.dataset)
    acc = correct / total
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    return loss, acc, y_true, y_pred


def plot_training_curve(t:list, v:list, typ:str, ep:int, title:str, p:str, fs:tuple[int, int]=(10, 5), dpi:int=300):
    """
    绘制并保存训练过程曲线
    :param t: 所有轮次的训练指标
    :param v: 所有轮次的验证指标
    :param typ: 指标类型
    :param ep: 最佳轮次
    :param title: 图标题
    :param p: 保存路径, 需包含文件名
    :param fs: 图像尺寸
    :param dpi: 图像分辨率
    """
    plt.figure(figsize=fs, dpi=dpi)
    plt.plot(t, label=f'Training {typ}')
    plt.plot(v, label=f'Validation {typ}')
    plt.axvline(x=ep, color='r', linestyle='--', label='Best Epoch')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(typ)
    plt.legend()
    plt.savefig(p)
    plt.close()


def create_optimize_objective(x:np.ndarray, y:np.ndarray, n_out:int, net_name:str, path_fig:str, dev):
    """
    创建超参搜索目标
    由于方法objective只能接受一个参数(trial), 因此使用闭包来给它传入更多参数
    :param x: 用于调参的训练数据(EEG)
    :param y: 用于调参的标签(位移向量)
    :param n_out: 模型输出数量
    :param net_name: 网络名称
    :param path_fig: 损失曲线图保存路径
    :param dev: 硬件设备
    """
    def objective(trial) -> float:
        """Optuna的超参调优目标函数"""
        timer = RunTimer()
        s = Settings()

        # region 定义超参空间
        bs = trial.suggest_categorical('batch_size', s.optimize_space['batch_size'])  # 定义批次大小空间
        lr = trial.suggest_float(                                                     # 定义学习率空间
            'learning_rate',
            s.optimize_space['learning_rate'][0],
            s.optimize_space['learning_rate'][1],
            log=True
        )
        dr = trial.suggest_categorical('dropout_rate', s.optimize_space['dropout_rate'])  # 定义丢弃率空间
        psr = None

        if net_name == 'EEGNet_LSTM':  # TODO 8: 封装为函数get_network_parameter()
            s_b1k1 = trial.suggest_categorical('block1_kernel1_size', s.optimize_space[net_name]['block1_kernel1_size'])  # 定义第一层卷积核大小空间
            n_b1k1 = trial.suggest_categorical('block1_kernel1_num', s.optimize_space[net_name]['block1_kernel1_num'])    # 定义第一层卷积核数量空间
            n_lstm_u = trial.suggest_categorical('lstm_unit_num', s.optimize_space[net_name]['lstm_unit_num'])            # 定义LSTM单元数量空间
            net_hp = {'dropout_rate': dr, 'block1_kernel1_size': s_b1k1, 'block1_kernel1_num': n_b1k1, 'lstm_unit_num': n_lstm_u}
        elif net_name == 'DBConformer':
            psr = trial.suggest_categorical('patch_size_ratio', s.optimize_space[net_name]['patch_size_ratio'])
            net_hp = {'patch_size_ratio': psr}
        else:
            net_hp = None

        if lr <= s.small_learning_rate_threshold and bs >= s.big_batch_size_threshold:
            return float('inf')  # 跳过小学习率与大批次的组合

        net_inf = ''  # 网络超参提示信息
        if net_hp is not None:
            if net_name == 'EEGNet_LSTM':
                net_inf = (f's_b1k1={net_hp["block1_kernel1_size"]:3}, '
                            f'n_b1k1={net_hp["block1_kernel1_num"]:3}, '
                            f'n_lstm_u={net_hp["lstm_unit_num"]:3}')
            elif net_name == 'DBConformer':
                net_inf = f'psr={net_hp["patch_size_ratio"]}'
        # endregion

        # region k折验证
        kf = KFold(n_splits=s.num_k_folds, shuffle=True, random_state=s.random_seed)  # k折验证
        f_min_v_loss, f_max_v_acc = [], []                                            # 记录每一折的最佳性能指标
        for f, (t_idx, v_idx) in enumerate(kf.split(x)):
            t_loss_rec, v_loss_rec, t_acc_rec, v_acc_rec = [], [], [], []  # 记录指标历史

            # region 准备数据集
            x_t, x_v = x[t_idx], x[v_idx]  # 划分训练集与验证集
            y_t, y_v = y[t_idx], y[v_idx]
            x_t, x_v = standardize_eeg(x_t, x_v)  # 标准化EEG
            t_dl = DataLoader(TensorDataset(torch.from_numpy(x_t), torch.from_numpy(y_t)), batch_size=bs, shuffle=True)             # 训练集数据生成器
            v_dl = DataLoader(TensorDataset(torch.from_numpy(x_v), torch.from_numpy(y_v)), batch_size=x_v.shape[0], shuffle=False)  # 验证集数据生成器, 不用打乱或分批次
            # endregion

            # region 初始化网络架构, 优化器与损失函数
            if net_name == 'EEGNet_LSTM':  # TODO 8: 封装为函数get_network()
                net = EEGNet_LSTM(n_input=x_t.shape[1], n_output=n_out, custom_para=net_hp)
            elif net_name == 'DBConformer':
                net = DBConformer(
                    ch_num=x_t.shape[1],
                    time_sample_num=x_t.shape[2],
                    patch_size=int(x_t.shape[2] * psr),
                    n_classes=n_out
                )
            else:
                print(f'    Model "{net_name}" is not defined, subprocess ended.')
                exit()
            net.to(dev)                                       # 将模型放到训练设备上
            opti = torch.optim.Adam(net.parameters(), lr=lr)  # 优化器
            crit = nn.CrossEntropyLoss()                      # 损失函数
            # endregion

            # region 训练模型
            min_v_loss = float('inf')  # 最低验证集损失
            max_v_acc = -1             # 最高验证集准确率
            pat_cnt = 0                # 早停耐心计数器
            best_ep = 0                # 最佳轮次
            for ep in range(s.max_training_epoch):
                t_loss, t_acc = epoch_train(net, t_dl, opti, crit, dev)
                v_loss, v_acc, _, _ = epoch_evaluate(net, v_dl, crit, dev)
                t_loss_rec.append(t_loss)
                v_loss_rec.append(v_loss)
                t_acc_rec.append(t_acc)
                v_acc_rec.append(v_acc)
                # region 早停
                if min_v_loss - v_loss > s.min_loss_improvement:  # 判断该轮训练是否有进步
                    min_v_loss = v_loss                   # 记录新的最佳损失
                    max_v_acc = v_acc
                    pat_cnt = 0                           # 若有进步就重置耐心计数器
                    best_ep = ep
                else:
                    pat_cnt += 1                          # 若无进步就开始等待
                    if pat_cnt >= s.early_stop_patience:  # 耗尽耐心就停止训练
                        break
                # endregion
            f_min_v_loss.append(min_v_loss)
            f_max_v_acc.append(max_v_acc)
            plot_training_curve(  # 绘制损失曲线
                t=t_loss_rec, v=v_loss_rec, typ='Loss', ep=best_ep,
                title=f'Trial {trial.number + 1:2} Fold {f + 1} (Loss: {min_v_loss:.3f})\n'
                      f'bs={bs}, lr={lr:.6f}, dr={dr}, {net_inf}',
                p=os.path.join(path_fig, f'trial_{trial.number + 1}_fold_{f + 1}_loss.png')
            )
            plot_training_curve(  # 绘制准确率曲线
                t=t_acc_rec, v=v_acc_rec, typ='Acc', ep=best_ep,
                title=f'Trial {trial.number + 1} Fold {f + 1} (Acc: {max_v_acc:.3f})\n'
                      f'bs={bs}, lr={lr:.6f}, dr={dr}, {net_inf}',
                p=os.path.join(path_fig, f'trial_{trial.number + 1}_fold_{f + 1}_acc.png')
            )
            # endregion
        # endregion

        avg_f_min_v_loss = np.mean(f_min_v_loss)
        avg_f_max_v_acc = np.mean(f_max_v_acc)

        print(f'    '
              f'Trial {trial.number+1:3}, '
              f'bs={bs:3}, '
              f'lr={lr:.6f}, '
              f'dr={dr}, '
              f'{net_inf}, '
              f'v_loss={avg_f_min_v_loss:.3f}, '
              f'v_acc={avg_f_max_v_acc:.3f}, ', end='')
        timer.stop(prompt='耗时')

        return float(avg_f_min_v_loss)  # 返回折平均验证损失
    return objective


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
          f'Model training: '
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
    path_optimize = get_path(
        path_spec='optimize',
        is_file=False,
        s_name=subject_name,
        s_turn=subject_turn,
        obs=is_observe,
        fr=bandpass_filter_range,
        tr=data_time_range,
        net=network_name
    )
    path_result = get_path(
        path_spec='result',
        is_file=False,
        s_name=subject_name,
        s_turn=subject_turn,
        obs=is_observe,
        fr=bandpass_filter_range,
        tr=data_time_range,
        net=network_name
    )
    os.makedirs(path_training, exist_ok=True)
    os.makedirs(path_optimize, exist_ok=True)
    os.makedirs(path_result, exist_ok=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)  # 让optuna闭嘴
    warnings.filterwarnings("ignore", message="Using padding='same' with even kernel lengths")  # 让那个卷积核性能提示也闭嘴
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

    k_fold_eeg, test_eeg, k_fold_label, test_label = train_test_split(
        eeg, label, test_size=S.test_ratio, shuffle=True, random_state=S.random_seed
    )  # k_fold_eeg, k_fold_label用于k折验证, test_eeg, test_label用于测试

    print(f'    Training set shape : {k_fold_eeg.shape}, {k_fold_label.shape}.')
    print(f'    Test set shape     : {test_eeg.shape}, {test_label.shape}.')
    # endregion

    # region 超参调优
    print('Tuning hyperparameters...')

    optimizer_timer = RunTimer()  # 开始计时
    optimize_study = optuna.create_study(direction='minimize')
    optimize_study.optimize(
        create_optimize_objective(x=k_fold_eeg, y=k_fold_label, n_out=2, net_name=network_name, path_fig=path_optimize, dev=device),
        n_trials=S.num_optimize_trial
    )  # 进行超参搜索
    best_hyperparameters = optimize_study.best_params  # 获取最佳参数
    path_best_hyperparameter = get_path(
        path_spec='best_hp',
        is_file=True,
        s_name=subject_name,
        s_turn=subject_turn,
        obs=is_observe,
        fr=bandpass_filter_range,
        tr=data_time_range,
        net=network_name
    )
    pd.DataFrame([best_hyperparameters]).to_csv(path_best_hyperparameter, index=False)  # 保存最佳参数

    optimizer_timer.stop(prompt='    Hyperparameter tuning time         : ', show_hours=True)
    print(f'    Best hyperparameters               : {best_hyperparameters}')
    print(f'    Best hyperparameters were saved to : {path_best_hyperparameter}')
    # endregion

    # region 训练模型
    print('Training model...')

    train_timer = RunTimer()
    train_eeg, validation_eeg, train_label, validation_label = train_test_split(
        k_fold_eeg, k_fold_label, test_size=S.test_ratio, shuffle=True, random_state=S.random_seed
    )  # 再次划分训练集与验证集

    train_eeg, validation_eeg, test_eeg = standardize_eeg(train_eeg, validation_eeg, test_eeg)  # 标准化数据集

    train_data_loader = DataLoader(  # 使用最佳批次大小构建训练集数据生成器
        TensorDataset(torch.from_numpy(train_eeg), torch.from_numpy(train_label)),
        batch_size=best_hyperparameters['batch_size'], shuffle=True
    )
    validation_data_loader = DataLoader(  # 验证集数据生成器
        TensorDataset(torch.from_numpy(validation_eeg), torch.from_numpy(validation_label)),
        batch_size=validation_eeg.shape[0], shuffle=False
    )

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
    network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=best_hyperparameters['learning_rate'])  # 使用最佳学习率构建优化器
    criterion = nn.CrossEntropyLoss()                                                             # 损失函数

    train_loss_record, train_accuracy_record, validation_loss_record, validation_accuracy_record = [], [], [], []
    minimum_validation_loss = float('inf')                    # 最佳验证集损失
    maximum_validation_accuracy = -1                          # 最佳验证集准确率
    best_model_weights = copy.deepcopy(network.state_dict())  # 最佳模型参数
    patience_counter = 0                                      # 早停耐心计数器
    best_epoch = 0                                            # 记录最佳轮次用于绘图

    for epoch in range(S.max_training_epoch):
        train_loss, train_accuracy = epoch_train(network, train_data_loader, optimizer, criterion, device)
        validation_loss, validation_accuracy, _, _ = epoch_evaluate(network, validation_data_loader, criterion, device)

        train_loss_record.append(train_loss)
        train_accuracy_record.append(train_accuracy)
        validation_loss_record.append(validation_loss)
        validation_accuracy_record.append(validation_accuracy)
        print(f'    Epoch {epoch + 1:4}/{S.max_training_epoch} : '
              f'Train loss={train_loss:.3f}, '
              f'Train acc={train_accuracy:.3f}, '
              f'Val loss={validation_loss:.3f}, '
              f'Val acc={validation_accuracy:.3f}.')

        if minimum_validation_loss - validation_loss > S.min_loss_improvement:  # 判断该轮训练是否有进步
            minimum_validation_loss = validation_loss  # 如果有就记录新的最佳损失
            maximum_validation_accuracy = validation_accuracy
            best_model_weights = copy.deepcopy(network.state_dict())  # 记录最佳模型参数
            best_epoch = epoch                                        # 记录最佳训练轮次
            patience_counter = 0                                      # 重置早停耐心计数器
        else:
            patience_counter += 1
            if patience_counter >= S.early_stop_patience:
                print(f'    Epoch {epoch + 1} early stopped, '
                      f'best val loss={minimum_validation_loss:.3f}, '
                      f'best val acc={maximum_validation_accuracy:.3f}.')
                break

    path_model = get_path(
        path_spec='best_mw',
        is_file=True,
        s_name=subject_name,
        s_turn=subject_turn,
        obs=is_observe,
        fr=bandpass_filter_range,
        tr=data_time_range,
        net=network_name
    )
    torch.save(best_model_weights, path_model)

    path_loss_curve = get_path(
        path_spec='loss_curve',
        is_file=True,
        s_name=subject_name,
        s_turn=subject_turn,
        obs=is_observe,
        fr=bandpass_filter_range,
        tr=data_time_range,
        net=network_name
    )
    path_acc_curve = get_path(
        path_spec='acc_curve',
        is_file=True,
        s_name=subject_name,
        s_turn=subject_turn,
        obs=is_observe,
        fr=bandpass_filter_range,
        tr=data_time_range,
        net=network_name
    )
    plot_training_curve(  # 绘制损失曲线
        t=train_loss_record, v=validation_loss_record, typ='Loss', ep=best_epoch,
        title=f'Loss: {minimum_validation_loss:.3f}',
        p=path_loss_curve
    )
    plot_training_curve(  # 绘制准确率曲线
        t=train_accuracy_record, v=validation_accuracy_record, typ='Accuracy', ep=best_epoch,
        title=f'Accuracy: {maximum_validation_accuracy:.3f}',
        p=path_acc_curve
    )

    print(f'    Model weights were saved to  : {path_model}')
    print(f'    Loss curve were saved to     : {path_loss_curve}')
    print(f'    Accuracy curve were saved to : {path_acc_curve}')
    train_timer.stop(prompt='    Model training time          : ', show_hours=True)
    # endregion

    # region 测试模型
    print('Testing model...')

    network.load_state_dict(best_model_weights)  # 载入最佳模型参数
    test_data_loader = DataLoader(               # 测试集数据生成器
        TensorDataset(torch.from_numpy(test_eeg), torch.from_numpy(test_label)),
        batch_size=test_eeg.shape[0], shuffle=False
    )
    _, _, test_truth, test_prediction = epoch_evaluate(network, test_data_loader, criterion, device)

    np.save(get_path(
        path_spec='test_truth',
        is_file=True,
        s_name=subject_name,
        s_turn=subject_turn,
        obs=is_observe,
        fr=bandpass_filter_range,
        tr=data_time_range,
        net=network_name
    ), test_truth)  # 保存真实值与预测值
    np.save(get_path(
        path_spec='test_pred',
        is_file=True,
        s_name=subject_name,
        s_turn=subject_turn,
        obs=is_observe,
        fr=bandpass_filter_range,
        tr=data_time_range,
        net=network_name
    ), test_prediction)
    dataframe = pd.DataFrame(np.column_stack([test_truth, test_prediction]))
    column_title = ['Truth'] + [f'Prediction_{i + 1}' for i in range(test_prediction.shape[1])]
    dataframe.columns = column_title
    dataframe.to_csv(get_path(
        path_spec='test_truth_pred',
        is_file=True,
        s_name=subject_name,
        s_turn=subject_turn,
        obs=is_observe,
        fr=bandpass_filter_range,
        tr=data_time_range,
        net=network_name
    ), index=False)

    test_predicted_labels = np.argmax(test_prediction, axis=1)  # 将模型输出转为预测类别(0或1)
    test_acc = np.mean(test_predicted_labels == test_truth)  # 计算准确率

    print(f'    Test set result              : Accuracy={test_acc:.6f}')
    print(f'    Test set result was saved to : {path_result}')
    # endregion
