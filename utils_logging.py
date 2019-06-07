# coding:utf8
import time


class Logging:
    process_num = 0
    process = []
    process_start_time = []

    def __init__(self):
        pass

    def start(self, x):
        self.info(x + ' start...')
        self.process_num += 1
        self.process.append(x)
        self.process_start_time.append(time.time())

    def info(self, x):
        print('    ' * self.process_num + x)

    def end(self):
        self.process_num -= 1
        self.info(self.process[-1] + ' end. It cost %s seconds' % (time.time() - self.process_start_time[-1]))
        self.process.pop()
        self.process_start_time.pop()