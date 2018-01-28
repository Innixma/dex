# By Nick Erickson, Benjamin Hillmann, Sivaraman Rajaganapathy
# Contains main functions for detecting a game loss

# TODO: Remake this to be better, use preparedImage instead of raw
import numpy as np

mean_light = 0
mean_diff = 0
max_diff = 0
max_mean = 0
prev_mean = 0
c = 0

thresholdMax = 120 / 255


# Resets globals
def reset_globs():
    global prev_mean, mean_diff, mean_light, c, max_diff, max_mean
    prev_mean = 0
    mean_diff = 0  # Currently unused
    mean_light = 0
    max_diff = 0  # Current unused
    max_mean = 0
    prev_mean = 0
    c = 0


# Checks for terminal
def check_terminal(data):
    global prev_mean, mean_diff, mean_light, c, max_diff, max_mean
    c += 1

    # Check if you lost, screen goes white when you lose, so check
    cur_mean = np.mean(data)
    if c <= 20:
        max_diff = 0
        mean_light += cur_mean
        if c == 20:
            mean_light = mean_light / 20
    else:
        mean_light = mean_light * (9 / 10) + cur_mean * (1 / 10)
        diff = cur_mean - prev_mean
        mean_diff = (mean_diff * ((c - 1) / c) + abs(diff) * (1 / c))
        max_diff = max(max_diff, abs(diff))
        max_mean = max(cur_mean, max_mean)
        # print(meanDiff, diff)

        threshold = (thresholdMax - mean_light) / 2
        # print(threshold, diff, maxDiff)
        # print(diff, threshold)
        if diff > threshold:
            reset_globs()
            return 1  # Lost!

    prev_mean = cur_mean
    return 0
