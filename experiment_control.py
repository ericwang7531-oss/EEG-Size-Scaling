"""
物体线速度想象范式
本脚本从向C#脚本发送指令, 控制Unity渲染视觉刺激
"""
import os
import time
import queue
import socket
import random
import pandas as pd
from threading import Thread
from settings import Settings
from pynput.keyboard import Listener, Key
from tools import precise_wait, RunTimer, get_path


enter_pressed = False
def on_press(key):
    """监听是否按下回车键, 用于结束练习"""
    if key == Key.enter:
        global enter_pressed
        enter_pressed = True
        print('练习即将结束')


class UnityCommandThread(Thread):
    """通过UDP向Unity发送指令的子线程"""
    def __init__(self, ip:str, port:int):
        """
        :param ip: IP地址
        :param port: 端口号
        """
        super().__init__(daemon=True)
        self._port = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (ip, port)
        self._cmd_queue = queue.Queue()  # 指令队列

    def run(self):
        """运行子线程"""
        while True:
            cmd = self._cmd_queue.get()
            if cmd is None: break
            try:
                self._port.sendto(cmd.encode('utf-8'), self._addr)
            except Exception as e:
                print(f'指令发送错误: {e}')
        self._port.close()

    def set_command(self, cmd:str):
        """
        将新Unity指令加入队列
        :param cmd: 新Unity指令
        """
        self._cmd_queue.put(cmd)

    def stop(self):
        """停止子线程"""
        self._cmd_queue.put(None)
        self.join()


class TriggerThread(Thread):
    """通过TCP向EEG数据打标的子线程"""
    def __init__(self, ip:str, port:int, timeout:float):
        """
        :param ip: 同步器IP地址
        :param port: 端口号
        :param timeout: 超时时长
        """
        super().__init__(daemon=True)
        self._port = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._port.settimeout(timeout)
        self._port.connect((ip, port))

        self._trig_queue = queue.Queue()  # 标签队列

    def run(self):
        """运行子线程"""
        while True:
            trig = self._trig_queue.get()
            if trig is None:
                break
            try:
                self._port.sendall(bytes([0x01, 0xE1, 0x01, 0x00, trig]))
            except Exception as e:
                print(f'打标错误: {e}')
        self._port.close()

    def set_trigger(self, trig: int):
        """
        将新标签加入队列
        :param trig: 新标签
        """
        self._trig_queue.put(trig)

    def stop(self):
        """停止子线程"""
        self._trig_queue.put(None)
        self.join()


class CSVRecordThread(Thread):
    """将试次标签记录至CSV文件的子线程"""
    def __init__(self, col:list, path:str, fn:str):
        """
        :param col: CSV列标题
        :param path: CSV保存路径
        :param fn: 文件名(包含保存路径)
        """
        super().__init__(daemon=True)
        self.is_running = False  # 线程运行状态
        self.col = col           # CSV列标题
        self.path = path         # CSV保存路径
        self.fn = fn             # 文件名(包含保存路径)

        os.makedirs(self.path, exist_ok=True)  # 创建CSV保存路径
        pd.DataFrame(columns=self.col).to_csv(self.fn, index=False)

        self._info_queue = queue.Queue()  # CSV内容队列

    def run(self):
        """运行子线程"""
        self.is_running = True
        while self.is_running:
            info = self._info_queue.get()
            if info is None:
                break
            df = pd.DataFrame(
                data=[[info[0], info[1], info[2], time.perf_counter()]],
                columns=self.col
            )
            try:
                df.to_csv(self.fn, mode='a', header=False, index=False)
            except Exception as e:
                print(f'标签写入错误: {e}')

    def set_info(self, label: str, block:int, trial: int):
        """
        将需要写入CSV的新内容加入队列
        :param label: 新标签
        :param block: 组块序号
        :param trial: 试次序号
        """
        self._info_queue.put([label, block, trial])

    def stop(self):
        """停止子线程"""
        self._info_queue.put(None)
        self.join()


if __name__ == "__main__":

    # TODO 7: 视觉想象生动度测试
    experiment_timer = RunTimer()  # 开始计时
    S = Settings()                 # 获得参数
    path_eeg_save = get_path(path_spec='ori_eeg', is_file=False, s_name=S.current_subject_name, s_turn=S.current_subject_turn)
    os.makedirs(path_eeg_save, exist_ok=True)  # 创建EEG数据保存路径

    # region 初始化子线程
    unity_command_thread = UnityCommandThread(ip=S.address_unity, port=S.port_unity)                                     # 初始化Unity控制线程
    trigger_thread = TriggerThread(ip=S.address_synchronizer, port=S.port_synchronizer, timeout=S.timeout_synchronizer)  # 初始化打标线程
    csv_record_thread = CSVRecordThread(                                                                                 # 初始化CSV记录线程
        col=S.name_csv_columns,
        path=get_path(path_spec='ori_label', is_file=False, s_name=S.current_subject_name, s_turn=S.current_subject_turn),
        fn=get_path(path_spec='ori_label', is_file=True, s_name=S.current_subject_name, s_turn=S.current_subject_turn)
    )
    keyboard_listener = Listener(on_press=on_press)                                                                      # 初始化键盘监听器

    unity_command_thread.start()  # 启动Unity控制线程
    trigger_thread.start()        # 启动打标线程
    csv_record_thread.start()     # 启动CSV记录线程
    keyboard_listener.start()     # 启动键盘监听器
    # endregion

    # region 实验前练习
    print('按回车键退出练习')
    print('请不要最小化Python窗口, 否则会增加打标时间误差!')

    trial_count = 0
    while True:
        speed_initial = S.speed_accelerate_min if trial_count % 2 == 0 else S.speed_decelerate_max
        speed_final = S.speed_accelerate_max if trial_count % 2 == 0 else S.speed_decelerate_min
        speed_change_rate = S.speed_accelerate_rate if trial_count % 2 == 0 else S.speed_decelerate_rate
        speed_change_delay = S.speed_accelerate_delay if trial_count % 2 == 0 else S.speed_decelerate_delay
        command_observe = f'{S.command_observe}:{speed_initial}:{speed_final}:{speed_change_rate}:{speed_change_delay}:{S.time_observe}'

        unity_command_thread.set_command(cmd=S.command_prepare)  # 准备阶段
        precise_wait(t=S.time_prepare)
        unity_command_thread.set_command(cmd=command_observe)    # 观察阶段
        precise_wait(t=S.time_observe)
        unity_command_thread.set_command(cmd=S.command_imagine)  # 想象阶段
        precise_wait(t=S.time_imagine)
        unity_command_thread.set_command(cmd=S.command_rest)     # 休息阶段
        precise_wait(t=S.time_rest)

        trial_count += 1
        if enter_pressed:
            keyboard_listener.stop()
            break
    # endregion

    # region 正式实验
    print('实验正式开始')
    unity_command_thread.set_command(cmd=S.command_experiment_start)  # 开始正式实验
    for block_idx in range(S.num_block):
        trigger_thread.set_trigger(S.trigger_block_start)
        # region 设置试次标签
        list_label = [S.label_observe_accelerate] * (S.num_trial // 2) + [S.label_observe_decelerate] * (S.num_trial // 2)  # 一半试次为加速, 另一半为减速
        random.shuffle(list_label)  # 随机打乱顺序

        list_trigger = [S.trigger_observe_accelerate if label == S.label_observe_accelerate else S.trigger_observe_decelerate for label in list_label]
        list_initial_speed = [S.speed_accelerate_min if label == S.label_observe_accelerate else S.speed_decelerate_max for label in list_label]
        list_final_speed = [S.speed_accelerate_max if label == S.label_observe_accelerate else S.speed_decelerate_min for label in list_label]
        list_speed_change_rate = [S.speed_accelerate_rate if label == S.label_observe_accelerate else S.speed_decelerate_rate for label in list_label]
        list_speed_change_delay = [S.speed_accelerate_delay if label == S.label_observe_accelerate else S.speed_decelerate_delay for label in list_label]
        # endregion

        for trial_idx in range(S.num_trial):
            trigger_thread.set_trigger(trig=S.trigger_prepare)         # 准备阶段
            unity_command_thread.set_command(cmd=S.command_prepare)
            precise_wait(t=S.time_prepare)

            trigger_thread.set_trigger(trig=list_trigger[trial_idx])   # 观察阶段
            unity_command_thread.set_command(cmd=f'{S.command_observe}:'
                                                 f'{list_initial_speed[trial_idx]}:'
                                                 f'{list_final_speed[trial_idx]}:'
                                                 f'{list_speed_change_rate[trial_idx]}:'
                                                 f'{list_speed_change_delay[trial_idx]}:'
                                                 f'{S.time_observe}')
            precise_wait(t=S.time_observe)

            trigger_thread.set_trigger(trig=S.trigger_imagine)         # 想象阶段
            unity_command_thread.set_command(cmd=S.command_imagine)
            precise_wait(t=S.time_imagine)

            trigger_thread.set_trigger(trig=S.trigger_rest)            # 休息阶段
            csv_record_thread.set_info(label=list_label[trial_idx], block=block_idx, trial=trial_idx)
            if trial_idx != S.num_trial - 1:  # 组块的最后一个试次不进行短休息, 直接进入长休息
                unity_command_thread.set_command(cmd=S.command_rest)
                precise_wait(t=S.time_rest)

        if block_idx != S.num_block - 1:  # 最后一个组块不进行长休息, 直接结束
            unity_command_thread.set_command(cmd=f'{S.command_rest}:'  # 组块间长休息
                                                 f'{S.time_block_end}:'
                                                 f'{S.num_block-1-block_idx}')
            precise_wait(t=S.time_block_end)
    # endregion

    # region 实验结束
    experiment_timer.stop(prompt='实验结束, 耗时:', show_hours=True)
    unity_command_thread.set_command(cmd=f'{S.command_experiment_end}:End of experiment!')
    unity_command_thread.stop()
    trigger_thread.stop()
    csv_record_thread.stop()
    # endregion
