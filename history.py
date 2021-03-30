import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
# import os


class History(object):
    """
    class to save `loss` and  `accuracy`.
    """

    def __init__(self, name=None):
        self.name = name
        self.epoch = []
        self.acc = []
        self.auc = []
        self.acc_bin = []
        self.loss = []
        self.ptp01 = []
        self.ptp05 = []
        self.recent = None
        self.axes = []

    def add(self, logs, epoch):
        self.recent = logs
        self.epoch.append(epoch)
        if 'loss' in logs.keys():
            self.loss.append(logs['loss'])
        if 'acc' in logs.keys():
            self.acc.append(logs['acc'])
        if 'auc' in logs.keys():
            self.auc.append(logs['auc'])
        if 'acc_bin' in logs.keys():
            self.acc_bin.append(logs['acc_bin'])
        if 'ptp01' in logs.keys():
            self.ptp01.append(logs['ptp01'])
        if 'ptp05' in logs.keys():
            self.ptp05.append(logs['ptp05'])

    def set_axes(self, axes=None):
        if axes:
            self.axes = axes
        # new figure and axis
        else:
            plt.figure()
            num = int((len(self.acc) != 0) + (len(self.auc) != 0) + (len(self.loss) != 0) + (len(self.acc_bin) != 0)
                      + (len(self.ptp01) != 0) + (len(self.ptp05) != 0))
            for i in range(num):
                self.axes.append(plt.subplot(num, 1, i + 1))

    def _get_tick(self):
        tick_max = np.max(self.epoch)
        ticks_int = np.arange(0, tick_max, np.ceil(tick_max / 5))
        if max(ticks_int) != tick_max:
            ticks_int = np.append(ticks_int, tick_max)
        return ticks_int

    def plot(self, axes=None, show=True):
        """
        plot loss and acc in subplots
        :param axes:
        :param show:
        :return:
        """
        self.set_axes(axes=axes)
        ticks = self._get_tick()
        cnt = 0
        if len(self.loss) != 0:
            self.axes[cnt].plot(self.epoch, self.loss)
            self.axes[cnt].legend([self.name + '/loss'])
            self.axes[cnt].set_xticks(ticks)
            self.axes[cnt].set_xticklabels([str(e) for e in ticks])
            cnt += 1
        if len(self.acc) != 0:
            self.axes[cnt].plot(self.epoch, self.acc)
            self.axes[cnt].legend([self.name + '/acc'])
            self.axes[cnt].set_xticks(ticks)
            self.axes[cnt].set_xticklabels([str(e) for e in ticks])
            cnt += 1
        if len(self.auc) != 0:
            self.axes[cnt].plot(self.epoch, self.auc)
            self.axes[cnt].legend([self.name + '/auc'])
            self.axes[cnt].set_xticks(ticks)
            self.axes[cnt].set_xticklabels([str(e) for e in ticks])
            cnt += 1
        if len(self.acc_bin) != 0:
            self.axes[cnt].plot(self.epoch, self.acc_bin)
            self.axes[cnt].legend([self.name + '/acc_bin'])
            self.axes[cnt].set_xticks(ticks)
            self.axes[cnt].set_xticklabels([str(e) for e in ticks])
            cnt += 1
        if len(self.ptp01) != 0:
            self.axes[cnt].plot(self.epoch, self.ptp01)
            self.axes[cnt].legend([self.name + '/ptp01'])
            self.axes[cnt].set_xticks(ticks)
            self.axes[cnt].set_xticklabels([str(e) for e in ticks])
            cnt += 1
        if len(self.ptp05) != 0:
            self.axes[cnt].plot(self.epoch, self.ptp05)
            self.axes[cnt].legend([self.name + '/ptp05'])
            self.axes[cnt].set_xticks(ticks)
            self.axes[cnt].set_xticklabels([str(e) for e in ticks])
            cnt += 1

        plt.show() if show else None

    def clc_plot(self, axes=None, show=True):
        """
        clear output before plot, using in jupyter notebook to dynamically plot.
        :param axes:
        :param show:
        :return:
        """
        clear_output(wait=True)
        self.plot(axes=axes, show=show)

    def clear(self):
        clear_output(wait=True)

