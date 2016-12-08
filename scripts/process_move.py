# By Nick Erickson, Benjamin Hillmann, Sivaraman Rajaganapathy
# Contains main functions for selecting the optimal action

import numpy as np

meanLight = 0
meanDiff = 0
maxDiff = 0
maxMean = 0
prevMean = 0
c = 0

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
    
# Gets the optimal move
def get_move(data, moves, model):
    global prevMean, meanDiff, meanLight, c, maxDiff, maxMean
    c += 1
    
    ##############################################################
    # Check if you lost, screen goes white when you lose, so check
    curMean = np.mean(data)
    if c <= 60:
        maxDiff = 0
        meanLight += curMean
        if c == 60:
            meanLight = meanLight / 60
    else:
        meanLight = meanLight * (9/10) + curMean * (1/10)
        diff = curMean - prevMean
        meanDiff = (meanDiff*((c-1)/c) + abs(diff)*(1/c))
        maxDiff = max(maxDiff, abs(diff))
        maxMean = max(curMean, maxMean)
        #print(meanDiff, diff)
        threshold = (230 - meanLight) / 2
        #print(threshold, diff, maxDiff)
        #print(diff, threshold)
<<<<<<< HEAD
        if diff > threshold and c > 60:
            print(diff, threshold)
=======
        if diff > threshold and c > 20:
>>>>>>> 5d9e6839bded9089ccc713fd72bf5ca74475cb57
            return 'esc' # Lost!
    prevMean = curMean
    
    ##############################################################
    
    ##############################################################
    # Compute the optimal move
    #optimal_move = moves[(c % 2) + 1] # Placeholder
    if model:
        results = model.predict(data.reshape(1, 1, 200, 200, 3)[:,:,:,:,0])
        #print(results)
        optimal_move = moves[np.argmax(results)]
        print(optimal_move, results)
    else:
        optimal_move = moves[np.random.randint(0,3)]
    ##############################################################
    
    return optimal_move
    