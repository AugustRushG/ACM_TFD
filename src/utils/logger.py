import os
import time


class Logger:
    """Minimal logger that prints to stdout and appends to `<log_dir>/<log_name>/<log_name>_{train,eval}.txt`."""

    def __init__(self, log_name, training, log_dir='../logs'):
        self.small_log_dir = os.path.join(log_dir, log_name)
        os.makedirs(self.small_log_dir, exist_ok=True)

        suffix = '_train.txt' if training else '_eval.txt'
        self.log_file = os.path.join(self.small_log_dir, log_name + suffix)
        with open(self.log_file, 'w') as f:
            f.write(f'Log file created at {time.ctime()}\n')

    def get_log_folder_path(self):
        return self.small_log_dir

    def info(self, message):
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
        print(message)
