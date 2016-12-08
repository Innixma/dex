# By Nick Erickson, Benjamin Hillmann, Sivaraman Rajaganapathy
# Contains main functions for selecting the optimal action

import numpy as np

meanLight = 0
meanDiff = 0
maxDiff = 0
prevMean = 0
c = 0

# Resets globals
def reset_globs():
    global prevMean, meanDiff, meanLight, c, maxDiff
    prevMean = 0
    meanDiff = 0 # Currently unused
    maxDiff = 0 # Current unused
    prevMean = 0
    c = 0
    
# Gets the optimal move
def get_move(data, moves):
    global prevMean, meanDiff, meanLight, c, maxDiff
    c += 1
    
    ##############################################################
    # Check if you lost, screen goes white when you lose, so check
    curMean = np.mean(data)
    if c <= 20:
        maxDiff = 0
        meanLight += curMean
        if c == 20:
            meanLight = meanLight / 20
    else:
        meanLight = meanLight * (9/10) + curMean * (1/10)
        diff = curMean - prevMean
        meanDiff = (meanDiff*((c-1)/c) + abs(diff)*(1/c))
        maxDiff = max(maxDiff, abs(diff))
        #print(meanDiff, diff)
        threshold = (255 - meanLight) / 2
        #print(threshold, diff, maxDiff)
        #print(diff, threshold)
        if diff > threshold and c > 20:
            return 'esc' # Lost!
    prevMean = curMean
    
    ##############################################################
    
    ##############################################################
    # Compute the optimal move
    optimal_move = moves[(c % 2) + 1] # Placeholder
    ##############################################################
    
    return optimal_move
    