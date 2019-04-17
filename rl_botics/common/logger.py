import os
import tensorflow as tf
import csv


class Logger(object):
    def __init__(self, sess):
        self.sess = sess
        cur_dir = os.getcwd()
        log_dir = str(cur_dir) + '/logs/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.log_dct = {}
        self.filename = log_dir + 'log.txt'
        self.writer = tf.summary.FileWriter(logdir=log_dir)

    def _write_tensorboard(self):
        def summary_val(k, v):
            kwargs = {'tag': k, 'simple_value': float(v)}
            return tf.Summary.Value(**kwargs)
        summary = tf.Summary(value=[summary_val(k, v) for k, v in self.log_dct.items()])

        self.writer.add_summary(summary, self.log_dct['ep'])
        self.writer.flush()

    def _write_file(self):
        with open(self.filename, "a") as f:
            writer = csv.writer(f)
            for k, v in self.log_dct.items():
                writer.writerow([k,v])

    def _clear(self):
        self.log = {}

    def log(self, dict):
        self.log_dct.update(dict)
        self._write_tensorboard()
        self._write_file()
        self._clear()
