"""定义整个项目的参数, 最好不要在运行代码的时候修改"""
import os


class Settings:
    """定义所有项目参数"""
    def __init__(self):
        # region 通用
        self.path_main = os.getcwd()                                   # 项目目录 (不要直接执行main.py之外的脚本, 不然会导致路径错误!)
        self.path_data = os.path.join(self.path_main, 'data')          # 数据目录
        self.path_training = os.path.join(self.path_main, 'training')  # 训练目录
        self.path_result = os.path.join(self.path_main, 'result')      # 结果目录
        self.subprocess_to_run = {
            'experiment': False,    # 实验范式
            'preprocessor': False,  # 数据预处理
            'trainer': True,       # 模型训练与测试
            'explainer': False,     # 特征贡献分析(解释)
            'analyzer': False,      # 结果分析
        }                                 # 要运行的子进程
        self.subject_dictionary = {
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
        self.subject_to_process_dictionary = {
            'WZWY': [
                1,  # 范式版本: v1.0
            ],
            #'ZEM': [
            #    1,  # 范式版本: v1.0
            #],
            #'LBB': [
            #    1,  # 范式版本: v1.0
            #],
            'LZN': [
                1,  # 范式版本: v1.0
            ],
            'HYY': [
                1,  # 范式版本: v1.0
            ],
            #'LHZ': [
            #    1,  # 范式版本: v1.0
            #],
            #'DTY': [
            #    1,  # 范式版本: v1.0
            #],
            #'LRH': [
            #    1,  # 范式版本: v1.0
            #]
        }                     # 需要处理的受试名和受试实验次数
        self.use_observe_data = [
            False
        ]                                  # 是否使用观察期数据进行解码
        self.random_seed = 6                                           # 随机种子, 用于保证可复现性
        # endregion

        # region 范式
        self.current_subject_name = 'Test'          # 当前受试名
        self.current_subject_turn = 1               # 当下受试的第几次实验

        self.address_unity = '127.0.0.1'            # 与Unity的C#代码交互的IP地址
        self.address_synchronizer = '192.168.3.19'  #'192.168.3.19' '192.168.31.218'  # 同步器IP地址
        self.port_unity = 1111                      # 与Unity的C#代码交互的端口号
        self.port_synchronizer = 4321               # 同步器端口
        self.timeout_synchronizer = 1.0             # 同步器超时

        self.num_block = 8                  # 组块数量
        self.num_trial = 30                   # 每个组块的试次数量(需为偶数以确保两种标签数量相同)
        self.num_trial_subject_turn = {
            'WZWY': {
                '1': 30,  # 范式版本: v1.0
            },
            'ZEM': {
                '1': 30,  # 范式版本: v1.0
            },
            'LBB': {
                '1': 30,  # 范式版本: v1.0
            },
            'LZN': {
                '1': 30,  # 范式版本: v1.0
            },
            'HYY': {
                '1': 30,  # 范式版本: v1.0
            },
            'LHZ': {
                '1': 30,  # 范式版本: v1.0
            },
            'DTY': {
                '1': 30,  # 范式版本: v1.0
            },
            'LRH': {
                '1': 30,  # 范式版本: v1.0
            }
        }  # 由于每次实验的试次数量可能不同, 为避免影响坏段索引计算, 在此记录每个受试每次实验的试次数量

        self.time_prepare = 2.0     # 阶段时长(准备)
        self.time_imagine = 5.0     # 阶段时长(想象)
        self.time_rest = 3.0        # 阶段时长(休息)

        self.speed_accelerate_max = 30.0    # 物体加速时最大速度
        self.speed_accelerate_min = -1.5     # 物体加速时最小速度
        self.speed_accelerate_rate = 2.5      # 物体加速时速度变化率
        self.speed_accelerate_delay = -0.888089202  # 物体加速延迟

        self.speed_decelerate_max = 30.0    # 物体减速时最大速度
        self.speed_decelerate_min = -3     # 物体减速时最小速度
        self.speed_decelerate_rate = 0.5      # 物体减速时速度变化率
        self.speed_decelerate_delay = 1.691585463   # 物体减速延迟

        self.command_experiment_start = 'experiment_start'  # Unity指令(实验开始)
        self.command_prepare = 'prepare'                    # Unity指令(准备)
        self.command_observe = 'observe'                    # Unity指令
        self.command_imagine = 'imagine'                    # Unity指令(想象)
        self.command_rest = 'rest'                          # Unity指令(休息)
        self.command_experiment_end = 'experiment_end'      # Unity指令(实验结束)

        self.trigger_prepare = 1             # EEG标签(准备)
        self.trigger_size_expanding = 2  # EEG标签(物体膨胀)
        self.trigger_size_shrinking = 3  # EEG标签(物体缩小)
        self.trigger_rest = 4             # EEG标签(休息)

        self.label_size_expanding = '+'  # 训练标签(物体膨胀)
        self.label_size_shrinking = '-'  # 训练标签(物体缩小)

        self.name_csv_columns = ['label', 'block', 'trial', 'time']  # CSV列标题
        # endregion

        # region EEG预处理
        self.mne_shut_up = True    # 关闭MNE的运行输出

        self.montage = [#'standard_1020'
             r"D:\SUSTech_Lab\PanDeng\脑电受试数据\gtec_64+1_channels.loc"]  #g.tech                                                                    # 电极排布类型
        self.non_eeg_channel = {'ECG': 'ecg', 'HEOR': 'eog', 'HEOL': 'eog', 'VEOU': 'eog', 'VEOL': 'eog'}  # 标记非EEG通道
        self.use_eog_channel = False                                                                        # 是否使用EOG通道
        self.use_ecg_channel = False                                                                       # 是否使用ECG通道

        self.notch_filter_frequency = 50                    # FIR陷波滤波器频率(Hz)
        self.notch_filter_width = 3                         # FIR陷波滤波器带宽(Hz)
        self.bandpass_filter_frequency_range_list = [
            [0.1, 30],
            # [0.1, 13],
        ]   # FIR带通滤波器频段列表(Hz)
        self.neural_representation_target_band = [0.1, 30]  # 用于神经表征的频段

        self.bad_channel_dictionary = {
            'WZWY': {
                '1': {},
            },
            'ZEM': {
                '1': {},
            },
            'LBB': {
                '1': {},
            },
            'LZN': {
                '1': {},
            },
            'HYY': {
                '1': {},
            },
            'LHZ': {
                '1': {},
            },
            'DTY': {
                '1': {},
            },
            'LRH': {
                '1': {},
            },
        }   # 受试每次实验的坏导名称与出现坏导的组块范围(包括起始与结束组块, 若为'all'则所有组块均损坏)

        self.rereference_channel = 'average'  # 重参考方法, 可改为参考电极列表

        self.time_block_extra = 10  # 提取组块时, 开头结尾留出的多余时长(s)

        self.remove_eog_method = 'ica'    # 去除眼电的方法 ('ica', 'reg')
        self.ica_method = 'picard'        # ICA方法 ('fastica', 'infomax', 'picard')
        self.ica_measure = 'correlation'  # 判断伪迹的方法
        self.ica_threshold = 0.5          # 一个ICA组分与眼电电极相关性达到多少即判定为眼电伪迹
        self.ica_max_prune_amount = 3     # 最多去除多少个ICA组分

        self.baseline_time = 1.0       # 基线时间(s)
        self.reject_threshold = 150e-6  # 判断为坏段的阈值(V)

        self.epoch_plot_scale = 50e-6  # 缩放比例尺(V)
        self.new_sampling_rate = 100   # 新采样率(Hz)
        # endregion

        # region 默认网络架构
        self.EEGNet_LSTM = {
            'dropout_rate': 0.2,        # 丢弃率
            'block1_kernel1_num': 8,    # Block 1 二维卷积核数量
            'block1_kernel1_size': 64,  # Block 1 二维卷积核尺寸 (1, block1_kernel1_size)
            'block1_depth': 2,          # Block 1 深度卷积层深度
            'block1_pool_size': 4,      # Block 1 均值池化尺寸 (1, block1_pool_size)
            'block2_kernel_num': 8,     # Block 2 可分离卷积核数量
            'block2_pool_size': 8,      # Block 2 均值池化尺寸 (1, block2_pool_size)
            'lstm_unit_num': 64,        # LSTM单元数量
            'lstm_layer_num': 2,        # LSTM层数
        }  # EEGNet-LSTM默认超参
        # endregion

        # region 训练
        self.concat_turns = True                             # 是否将每个受试的所有数据合并在一起训练
        self.training_device = 'cuda'                         # 训练设备, 'cuda'或'cpu'
        self.imagine_data_time_range_to_process_list = [
            [0, 5],
            #[0, 3],
            #[1, 4],
            #[2, 5],
        ]  # 需要处理的想象期数据时间范围列表(s)
        self.observe_data_time_range_to_process_list = [
            [0, 3],
            [0, 1],
        ]  # 需要处理的观察期数据时间范围列表(s)
        self.imagine_data_time_range_list = [
            [0, 5],
            #[0, 3],
            #[1, 4],
            #[2, 5],
        ]             # 完整的想象期数据时间范围列表(s)
        self.observe_data_time_range_list = [
            [0, 3],
            [0, 1],
        ]             # 完整的观察期数据时间范围列表(s)
        self.network_list = [
            #'DBConformer',
            'EEGNet_LSTM',
        ]                             # 要训练的网络名称(与上述超参字典名相同)
        self.test_ratio = 0.1                                 # 测试集占总数据集的比例
        self.num_k_folds = 5                                  # K折验证折数
        self.num_optimize_trial = 50                          # 贝叶斯优化算法尝试次数(越多越准)
        self.max_training_epoch = 5000                        # 最大训练轮次
        self.early_stop_patience = 35                         # 早停耐心值
        self.min_loss_improvement = 0.0                       # 最小进步量, 若比这个小则不认为训练有进步
        self.small_learning_rate_threshold = 5e-4             # 小学习率判断阈值, 用于排除大批次和小学习率的组合
        self.big_batch_size_threshold = 512                   # 大批次判断阈值, 用于排除大批次和小学习率的组合
        self.optimize_space = {
            'batch_size': [4, 8, 16, 32, 64],             # 批次大小选项
            'learning_rate': [1e-5, 1e-2],                # 学习率范围(对数均匀分布), 注意该列表长度必须为2, 若需要固定一个值, 就设置两个相同的元素
            'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],    # 丢弃率选项
            'EEGNet_LSTM': {
                'block1_kernel1_size': [16, 32, 64, 128],  # 第一层卷积核大小选项
                'block1_kernel1_num': [8, 16, 32],         # 第一层卷积核数量选项
                'lstm_unit_num': [8, 16, 32, 64, 128],     # LSTM单元数量
            },                         # EEGNet-LSTM超参空间
            'DBConformer': {
                'patch_size_ratio': [1/5, 1/10, 1/20],   # 模型参数patch_size占窗口尺寸的比例, 其倒数就是patch数量
            },  # DBConformer超参空间
        }                           # 超参空间(注意, 与网络架构相关的键名称要与默认超参字典中的一致)
        # endregion

        # region 分析
        self.plot_font = 'Times New Roman'
        self.analysis_task = {
            'model_performance': True,       # 模型效果(准确率, 混淆矩阵, etc.)
            'neural_representation': False,  # 神经表征(ERP, ERSP, etc.)
            'feature_contribution': True,    # 特征贡献(EEG-SHAP热力图, SHAP地形图, etc.)
        }                          # 需要进行的任务

        self.accuracy_table_param = {
            'bad_accuracy_threshold': 0.7,    # 低于该阈值的将被标记为低准确率
            'good_accuracy_threshold': 0.8,   # 高于该阈值的将被标记为高准确率
            'bad_accuracy_color': 'FF0000',   # 低准确率标记色
            'good_accuracy_color': '0000FF',  # 高准确率标记色
            'first_column_width': 20,         # 第一列(行标题)宽度
        }                   # 准确率表格设置

        self.eeg_shap_heatmap_param = {
            'overwrite': False,  # 是否覆写已有图像
            # region 图窗
            'dpi': 300,                 # 分辨率
            'figure_size': (14, 7),     # 图窗大小
            'left': 0.03,               # 左沿位置
            'right': 0.97,              # 右沿位置
            'bottom': 0.07,             # 下沿位置
            'top': 0.92,                # 上沿位置
            'wspace': 0.1,              # 子图水平间距
            'colormap': 'RedGreyBlue',  # 使用的颜色图(来自colormaps.mat)
            # endregion
            # region 线条
            'linewidth': 1.5,        # 线宽
            'average_linewidth': 3,  # 绘制试次间平均时的线宽
            # endregion
            # region 字体大小
            'main_title_fontsize': 18,  # 主标题字号
            'subtitle_fontsize': 15,    # 子图标题字号
            'axis_label_fontsize': 15,  # 轴标签字号
            'xtick_fontsize': 10,       # 横轴刻度字号
            'ytick_fontsize': 10,       # 纵轴刻度字号
            # endregion
        }                 # EEG-SHAP热力图设置
        self.shap_topomap_sphere_radius = 0.056             # SHAP地形图半径
        self.shap_topomap_param = {
            'overwrite': False,  # 是否覆写已有图像
            # region 图窗
            'dpi': 300,                  # 分辨率
            'figure_size': (14, 7),      # 图窗大小
            'left': 0.03,                # 左沿位置
            'right': 0.97,               # 右沿位置
            'bottom': 0.07,              # 下沿位置
            'top': 0.92,                 # 上沿位置
            'wspace': 0.1,               # 子图水平间距
            'colormap': 'RedWhiteBlue',  # 使用的颜色图(来自colormaps.mat)
            # endregion
            # region 字体大小
            'main_title_fontsize': 18,  # 主标题字号
            'subtitle_fontsize': 15,    # 子图标题字号
            'axis_label_fontsize': 15,  # 轴标签字号
            'xtick_fontsize': 10,       # 横轴刻度字号
            # endregion
        }                     # SHAP地形图设置
        # endregion