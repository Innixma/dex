# By Nick Erickson
# Contains functions related to metrics

import numpy as np
import graphHelper

class Metrics: # TODO: Save this to a pickle file?
    def __init__(self):
        self.survival_times = []
        self.survival_times_last_10 = []
        self.survival_times_full_mean = []
        self.Q = []
        self.loss = []
        self.size = 0
        self.max_survival = -1
   
    def update(self, survival_time):
        self.size += 1
        if survival_time > self.max_survival:
            self.max_survival = survival_time
        self.survival_times.append(survival_time)
        self.survival_times_last_10.append(np.mean(self.survival_times[-10:]))
        self.survival_times_full_mean.append(np.mean(self.survival_times))

    def display_metrics(self, frame, useRate, total_saved=0, epsilon=0):
            if np.sum(useRate) != 0:
                useRate = useRate/np.sum(useRate)
            framerate = frame/self.survival_times[-1]
            print('R' + str(self.size) + ': ' + "%.2f" % self.survival_times[-1] + 's' + ', %.2f fps' % framerate + ', key: ', ['%.2f' % k for k in useRate], end='')
            print(' Mean: %.2f' % self.survival_times_full_mean[-1], 'Last 10: %.2f' % self.survival_times_last_10[-1], 'Max: %.2f' % self.max_survival, "TS", total_saved, "E %.2f" % epsilon)
    
    def save_metrics_training(self, save_location):
        graphHelper.graphSimple([np.arange(len(self.Q))], [self.Q], ['Q Value'], 'Q Value', 'Q Value', 'Replay (10^2)', savefigName=save_location + 'Q')
        graphHelper.graphSimple([np.arange(len(self.loss))], [self.loss], ['Loss'], 'Loss', 'Loss', 'Replay (10^2)', savefigName=save_location + 'loss')        
        
    def save_metrics(self, save_location):
        # TODO: Remove these logs, just export the data directly
        graphHelper.graphSimple([np.arange(self.size),np.arange(self.size),np.arange(self.size)], [self.survival_times, self.survival_times_last_10, self.survival_times_full_mean], ['Survival Time', 'Survival Rolling 10 Mean', 'Survival Mean'], 'Survival Times', 'Time(s)', 'Run', savefigName=save_location + 'survival')
                
            
            
            