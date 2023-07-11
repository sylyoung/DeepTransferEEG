# -*- coding: utf-8 -*-
# @Time    : 2022/1/12 12:44 下午
# @Author  : wenzhang
# @File    : CsvRecord.py
import pandas as pd
import numpy as np
import os
import csv
from datetime import datetime
from datetime import timedelta, timezone


class CsvRecord:
    def __init__(self, args):
        self.data_env = args.data_env
        self.data_str = args.data
        self.N = args.N
        self.file_str = args.file_str

    def init(self):
        name_list = ['file', 'data', 'time'] + [str(i + 1) for i in range(self.N)] + ['Avg', 'Std']

        acc_str_list = ['-' for _ in range(self.N)]
        if self.data_env == 'local':
            self.time_str = datetime.utcnow().replace(tzinfo=timezone.utc).astimezone(
                timezone(timedelta(hours=8), name='Asia/Shanghai')).strftime("%m-%d_%H_%M_%S")
        if self.data_env == 'gpu':
            self.time_str = datetime.utcnow().replace(tzinfo=timezone.utc).strftime("%m-%d_%H_%M_%S")
        output_str_row = np.array([self.file_str, self.data_str, self.time_str] + acc_str_list + ['-', '-'])
        output_pd = pd.DataFrame(dict(zip(name_list, output_str_row.T)), index=[0])

        # 检测是否存在该文件，如果存在则不init
        self.save_path = './csv/acc_log_' + self.data_str + '.csv'
        if not os.path.exists(self.save_path):
            output_pd.to_csv(self.save_path, index=None)

    def record(self, acc_array_raw):
        acc_str_list = [str(i) for i in np.round(acc_array_raw, 2)]
        mean_acc = np.round(np.mean(acc_array_raw), 2)
        std_acc = np.round(np.std(acc_array_raw), 2)
        output_str_row = np.array(
            [self.file_str, self.data_str, self.time_str] + acc_str_list + [str(mean_acc), str(std_acc)])
        with open(self.save_path, mode='a', newline='', encoding='utf8') as cfa:
            csv.writer(cfa).writerow(output_str_row)


if __name__ == '__main__':
    import argparse

    args = argparse.Namespace(data_env='local', data='MI2-4')
    args.N = 9
    args.file_str = 'demo_test'

    csv_log = CsvRecord(args)
    csv_log.init()

    sub_acc_all = np.array([69.097, 29.167, 80.208, 41.319, 38.889, 36.111, 64.583, 74.306, 67.014])
    csv_log.record(sub_acc_all)
