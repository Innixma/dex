# By Nick Erickson, Benjamin Hillmann, Sivaraman Rajaganapathy
# Contains main functions for detecting a game loss

# TODO: Remake this to be better, use preparedImage instead of raw
import numpy as np

meanLight = 0
meanDiff = 0
maxDiff = 0
maxMean = 0
prevMean = 0
c = 0

thresholdMax = 120 / 255

# Resets globals
def reset_globs():
    global prevMean, meanDiff, meanLight, c, maxDiff, maxMean
    prevMean = 0
    meanDiff = 0 # Currently unused
    meanLight = 0
    maxDiff = 0 # Current unused
    maxMean = 0
    prevMean = 0
    c = 0
    
# Checks for terminal
def check_terminal(data):
    global prevMean, meanDiff, meanLight, c, maxDiff, maxMean
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
        maxMean = max(curMean, maxMean)
        #print(meanDiff, diff)
        
        
        
        threshold = (thresholdMax - meanLight) / 2
        #print(threshold, diff, maxDiff)
        #print(diff, threshold)
        if diff > threshold:
            reset_globs()
            return 1 # Lost!
            
    prevMean = curMean
    return 0
    