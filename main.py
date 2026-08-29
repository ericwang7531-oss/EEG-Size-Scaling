"""主进程 (适配新版 settings.py)"""
import sys
import json
import subprocess
from settings import Settings
from tools import Logger, RunTimer

main_timer = RunTimer()
sys.stdout = Logger()
S = Settings()

if S.subprocess_to_run.get('experiment'):  # 范式
    process = subprocess.Popen(
        args=['python', '-u', 'src\\experiment_control.py'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8')
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end='')
    process.wait()

if S.subprocess_to_run.get('preprocessor'):  # 预处理
    for subject_name in S.subject_to_process_dictionary:
        for subject_turn in S.subject_to_process_dictionary[subject_name]:
            # 获取对应的坏导信息，注意 settings.py 里 turn 是字符串格式作为 key
            bad_channel = S.bad_channel_dictionary.get(subject_name, {}).get(str(subject_turn), {})
            for bandpass_filter_range in S.bandpass_filter_frequency_range_list:
                preprocess_timer = RunTimer()
                process = subprocess.Popen(
                    args=[
                        'python', '-u', 'src\\preprocessor.py',
                        subject_name,
                        str(subject_turn),
                        json.dumps(bad_channel),
                        json.dumps(bandpass_filter_range),
                    ],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding='utf-8', errors='replace'
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end='')
                process.wait()
                preprocess_timer.stop(prompt='Preprocessing time: ')

if S.subprocess_to_run.get('trainer'):  # 训练
    for subject_name in S.subject_to_process_dictionary:
        skip_turns = False  # 如果S.concat_turns为真, 则只会循环一次subject_turn
        for subject_turn in S.subject_to_process_dictionary[subject_name]:
            if skip_turns:
                continue
            for is_observe in S.use_observe_data:
                # 兼容可能缺失的合并数据配置，默认不合并
                merge_obs_img = getattr(S, 'merge_observe_imagine_data', False)
                if merge_obs_img and is_observe:
                    continue  # 如果合并观察与想象期数据, 且is_observe为真, 则跳过, 避免重复训练
                
                if is_observe:
                    data_time_range_list = S.observe_data_time_range_to_process_list
                else:
                    data_time_range_list = S.imagine_data_time_range_to_process_list
                
                for bandpass_filter_range in S.bandpass_filter_frequency_range_list:
                    for data_time_range in data_time_range_list:
                        if merge_obs_img:
                            merge_range = getattr(S, 'merge_observe_data_time_range', [0, 0])
                            if data_time_range[1] - data_time_range[0] != merge_range[1] - merge_range[0]:
                                continue  # 如果合并观察与想象期数据, 且观察与想象不同, 则跳过, 避免重复训练
                        
                        for network_name in S.network_list:
                            train_timer = RunTimer()
                            process = subprocess.Popen(
                                args=[
                                    'python', '-u', 'src\\trainer.py',
                                    subject_name,
                                    str(subject_turn),  # 在S.concat_turns为真且受试实验次数大于1的情况下, 这个参数无效
                                    str(int(is_observe)),
                                    json.dumps(bandpass_filter_range),
                                    json.dumps(data_time_range),
                                    network_name,
                                ],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace'
                            )
                            assert process.stdout is not None
                            for line in process.stdout:
                                print(line, end='')
                            process.wait()
                            train_timer.stop(prompt='Training time: ')
            
            # 使用 getattr 适配 settings.py 中的 concat_turns 变量，避免报错
            if getattr(S, 'concat_turns', False):
                skip_turns = True

if S.subprocess_to_run.get('explainer'):  # 解释模型
    for subject_name in S.subject_to_process_dictionary:
        skip_turns = False  
        for subject_turn in S.subject_to_process_dictionary[subject_name]:
            if skip_turns:
                continue
            for is_observe in S.use_observe_data:
                merge_obs_img = getattr(S, 'merge_observe_imagine_data', False)
                if merge_obs_img and is_observe:
                    continue  
                
                if is_observe:
                    data_time_range_list = S.observe_data_time_range_to_process_list
                else:
                    data_time_range_list = S.imagine_data_time_range_to_process_list
                
                for bandpass_filter_range in S.bandpass_filter_frequency_range_list:
                    for data_time_range in data_time_range_list:
                        if merge_obs_img:
                            merge_range = getattr(S, 'merge_observe_data_time_range', [0, 0])
                            if data_time_range[1] - data_time_range[0] != merge_range[1] - merge_range[0]:
                                continue  
                        
                        for network_name in S.network_list:
                            explain_timer = RunTimer()
                            process = subprocess.Popen(
                                args=[
                                    'python', '-u', 'src\\explainer.py',
                                    subject_name,
                                    str(subject_turn),  
                                    '1',  # paradigm_version
                                    str(int(is_observe)),
                                    json.dumps(bandpass_filter_range),
                                    json.dumps(data_time_range),
                                    network_name,
                                ],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8'
                            )
                            assert process.stdout is not None
                            for line in process.stdout:
                                print(line, end='')
                            process.wait()
                            explain_timer.stop(prompt='Explaining time: ')
            
            if getattr(S, 'concat_turns', False):
                skip_turns = True

if S.subprocess_to_run.get('analyzer'):  # 分析并可视化结果
    for bandpass_filter_range in S.bandpass_filter_frequency_range_list:
        analyzer_timer = RunTimer()
        process = subprocess.Popen(
            args=[
                'python', '-u', 'src\\analyzer.py',
                json.dumps(bandpass_filter_range),
            ],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8'
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end='')
        process.wait()
        analyzer_timer.stop(prompt='Analyzing time: ')

main_timer.stop(prompt='Main process time: ', show_hours=True)