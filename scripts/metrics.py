# By Nick Erickson
# Contains functions related to metrics

import numpy as np
import graphHelper
import sys

class Metrics: # TODO: Save this to a pickle file?
    def __init__(self, metric_type='a3c'):
        self.survival_times = []
        self.survival_times_last_10 = []
        self.survival_times_full_mean = []
        self.Q = []
        self.loss = []
        self.total_size = 0 
        self.size = 0
        self.max_survival = -1
        self.V = []
        self.V_episode = []

        if metric_type == 'a3c':
            self.a3c = A3C_Metrics()
        else:
            self.a3c = None
   
    def update(self, survival_time):
        self.size += 1
        self.total_size += 1
        if survival_time > self.max_survival:
            self.max_survival = survival_time
        self.survival_times.append(survival_time)
        self.survival_times_last_10.append(np.mean(self.survival_times[-10:]))
        self.survival_times_full_mean.append(np.mean(self.survival_times)) # make this better...

    def display_metrics(self, frame, useRate, total_saved=0, epsilon=0):
            if np.sum(useRate) != 0:
                useRate = useRate/np.sum(useRate)
            framerate = frame/self.survival_times[-1]
            print('R' + str(self.total_size) + ': ' + "%.2f" % self.survival_times[-1] + 's' + ', %.2f fps' % framerate + ', key: ', ['%.2f' % k for k in useRate], end='')
            print(' Mean: %.2f' % self.survival_times_full_mean[-1], 'Last 10: %.2f' % self.survival_times_last_10[-1], 'Max: %.2f' % self.max_survival, "TS", total_saved, "E %.2f" % epsilon)
            
            sys.stdout.flush()
            
    def save_metrics_training(self, save_location):
        graphHelper.graphSimple([np.arange(len(self.Q))], [self.Q], ['Q Value'], 'Q Value', 'Q Value', 'Replay (10^2)', savefigName=save_location + 'Q')
        graphHelper.graphSimple([np.arange(len(self.loss))], [self.loss], ['Loss'], 'Loss', 'Loss', 'Replay (10^2)', savefigName=save_location + 'loss')        
    
    def save_metrics_v(self, save_location):
        graphHelper.graphSimple([np.arange(len(self.V))], [self.V], ['Value'], 'Value', 'Value', 'Run', savefigName=save_location + 'V')
        graphHelper.graphSimple([np.arange(len(self.V_episode))], [self.V_episode], ['Value'], 'Value', 'Value', 'Frame', savefigName=save_location + 'V_episode')
        
    def save_metrics(self, save_location):
        # TODO: Remove these logs, just export the data directly
        graphHelper.graphSimple([np.arange(self.size),np.arange(self.size),np.arange(self.size)], [self.survival_times, self.survival_times_last_10, self.survival_times_full_mean], ['Survival Time', 'Survival Rolling 10 Mean', 'Survival Mean'], 'Survival Times', 'Time(s)', 'Run', savefigName=save_location + 'survival')
        
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
        graphHelper.graphSimple([np.arange(self.size)], [self.mean], [self.name], self.name, self.name, 'Batch', savefigName=save_location + self.name)
        
        
class A3C_Metrics:
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
            metric.graph_mean(save_location)
        
        

            





            