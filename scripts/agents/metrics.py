# By Nick Erickson
# Contains functions related to metrics

import sys

import numpy as np

from utils import graph_helper
from utils.data_utils import save_class


class Metrics:  # TODO: Save this to a pickle file?
    def __init__(self, metric_type='a3c'):
        self.type = metric_type
        # self.survival_times = []
        # self.survival_times_last_10 = []
        # self.survival_times_full_mean = []
        self.runs = MetricInfoSurvival(name='survival')
        self.Q = []
        self.loss = []
        # self.total_size = 0
        # self.size = 0
        # self.max_survival = -1
        self.V = []
        self.V_episode = []

        if metric_type == 'a3c':
            self.a3c = MetricsA3C()
        else:
            self.a3c = None

    def save(self, save_location, name=None):
        if name is None:
            name = ''

        save_class(self, save_location + name + '.dat')

    """
    def update(self, survival_time):
        self.size += 1
        self.total_size += 1
        if survival_time > self.max_survival:
            self.max_survival = survival_time
            
        #self.survival_times.append(survival_time)
        #self.survival_times_last_10.append(np.mean(self.survival_times[-10:]))
        #self.survival_times_full_mean.append(np.mean(self.survival_times)) # make this better...
    """

    def display_metrics(self, frame, use_rate, total_saved=0, epsilon=0):
            if np.sum(use_rate) != 0:
                use_rate = use_rate / np.sum(use_rate)
            framerate = frame/self.runs.times[-1]
            print('R' + str(self.runs.total_size) + ': ' + "%.2f" % self.runs.times[-1] + 's' + ', %.2f fps' % framerate + ', key: ', ['%.2f' % k for k in use_rate], end='')
            print(' Mean: %.2f' % self.runs.mean[-1], 'Last 10: %.2f' % self.runs.last10[-1], 'Max: %.2f' % self.runs.max, "TS", total_saved, "E %.2f" % epsilon)

            sys.stdout.flush()

    def save_metrics_training(self, save_location):
        graph_helper.graph_simple([np.arange(len(self.Q))], [self.Q], ['Q Value'], 'Q Value', 'Q Value', 'Replay (10^2)',
                                  savefig_name=save_location + 'Q')
        graph_helper.graph_simple([np.arange(len(self.loss))], [self.loss], ['Loss'], 'Loss', 'Loss', 'Replay (10^2)',
                                  savefig_name=save_location + 'loss')

    def save_metrics_v(self, save_location):
        graph_helper.graph_simple([np.arange(len(self.V))], [self.V], ['Value'], 'Value', 'Value', 'Run',
                                  savefig_name=save_location + 'V')
        graph_helper.graph_simple([np.arange(len(self.V_episode))], [self.V_episode], ['Value'], 'Value', 'Value',
                                 'Frame', savefig_name=save_location + 'V_episode')

    # def save_metrics(self, save_location):
    #    # TODO: Remove these logs, just export the data directly
    #    graphHelper.graphSimple([np.arange(self.size),np.arange(self.size),np.arange(self.size)], [self.survival_times, self.survival_times_last_10, self.survival_times_full_mean], ['Survival Time', 'Survival Rolling 10 Mean', 'Survival Mean'], 'Survival Times', 'Time(s)', 'Run', savefigName=save_location + 'survival')


class MetricInfoSurvival:
    def __init__(self, total_size=0, name='survival'):
        self.name = name
        self.total_size = total_size
        self.size = 0
        self.max = -1
        self.mean = []
        self.last10 = []
        self.times = []

    def update(self, time):
        if self.max < time:
            self.max = time

        self.times.append(time)
        self.last10.append(np.mean(self.times[-10:]))

        if self.size == 0:
            self.mean.append(time)
        else:
            self.mean.append((self.mean[-1] * self.size + time) / (self.size+1))

        self.size += 1
        self.total_size += 1

    def graph(self, save_location, name=None):
        if name is None:
            name = self.name
        # TODO: Remove these logs, just export the data directly
        graphRange = np.arange(self.total_size - self.size, self.total_size)
        graph_helper.graph_simple([graphRange, graphRange, graphRange], [self.times, self.last10, self.mean],
                                  ['Time', 'Rolling 10 Mean', 'Mean'], 'Survival Times', 'Time(s)', 'Run',
                                  savefig_name=save_location + name)


class MetricInfo:
    def __init__(self, name):
        self.name = name
        self.mean = []
        self.max = []
        self.min = []
        self.size = 0

    def update(self, data):
        self.mean.append(np.mean(data))
        self.max.append(np.max(data))
        self.min.append(np.min(data))
        self.size += 1

    def graph_mean(self, save_location):
        graph_helper.graph_simple([np.arange(self.size)], [self.mean], [self.name], self.name, self.name, 'Batch',
                                  savefig_name=save_location + self.name)


class MetricsA3C:
    def __init__(self):
        self.L  = MetricInfo('Loss Total') # Loss Total
        self.Pr = MetricInfo('Log Probability') # Log Prob
        self.Po = MetricInfo('Loss Probability') # Loss Prob
        self.V  = MetricInfo('Loss Value') # Loss Value
        self.E  = MetricInfo('Loss Entropy') # Loss Entropy

        self.metrics = []
        self.metrics.extend([self.L, self.Pr, self.Po, self.V, self.E])

    def update(self, l, pr, po, v, e):
        self.L.update(l)
        self.Pr.update(pr)
        self.Po.update(po)
        self.V.update(v)
        self.E.update(e)

    def graph_all(self, save_location):
        for metric in self.metrics:
            try:
                metric.graph_mean(save_location)
            except:
                print(metric.name)
