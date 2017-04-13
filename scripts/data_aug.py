# By Nick Erickson

# Data augmentation code

import numpy as np

def full_augment(total_data):
    
    total_data = total_data + pixel(total_data)
    
    total_data = total_data + rotate4(total_data)
    
    return total_data + flip(total_data)

def rotate4(data): # Rotates data 90 degrees 3 times, quadrupling the data, returns the new data.
    d_new = []
    data_shape = data[0][0].shape
    img_channels = data_shape[-1]
    for d in data:
        s0 = np.zeros(data_shape, dtype='float16')
        s0_ = np.zeros(data_shape, dtype='float16')
        s1 = np.zeros(data_shape, dtype='float16')
        s1_ = np.zeros(data_shape, dtype='float16')
        s2 = np.zeros(data_shape, dtype='float16')
        s2_ = np.zeros(data_shape, dtype='float16')
        for i in range(img_channels):
            #print(i)
            p0 = np.rot90(d[0][:,:,i])
            p1 = np.rot90(p0)
            p2 = np.rot90(p1)
            
            p0_ = np.rot90(d[3][:,:,i])
            p1_ = np.rot90(p0_)
            p2_ = np.rot90(p1_)
            
            s0[:,:,i] = p0
            s0_[:,:,i] = p0_
            
            s1[:,:,i] = p1
            s1_[:,:,i] = p1_
            
            s2[:,:,i] = p2
            s2_[:,:,i] = p2_
            
        d_new.append([s0, d[1], d[2], s0_, d[4]])
        d_new.append([s1, d[1], d[2], s1_, d[4]])
        d_new.append([s2, d[1], d[2], s2_, d[4]])
    return d_new
        
# NOTE: Assumes action is 0 = none, 1 = left, 2 = right, must swap
def flip(data): # Flips data lr
    d_new = []
    for d in data:
        if d[1] == 0:
            d1 = 0
        elif d[1] == 1:
            d1 = 2
        else:
            d1 = 1
        d_new.append([np.fliplr(d[0]), d1 % 2, d[2], np.fliplr(d[3]), d[4]])
    return d_new
            
def pixel(data): # Lacks diagonal
    d_new = []
    rows = [-1, 1]
    cols = [-1, 1]
    for d in data:
        for row in rows:
            s0 = np.roll(d[0], row, 0)
            s0_ = np.roll(d[3], row, 0)
            d_new.append([s0, d[1], d[2], s0_, d[4]])
           
        for col in cols:
            s0 = np.roll(d[0], col, 1)
            s0_ = np.roll(d[3], col, 1)
            d_new.append([s0, d[1], d[2], s0_, d[4]])
        """
        for row in rows:
            for col in cols:
                if row != 0 or col != 0:
                    s0 = np.zeros(data_shape, dtype='float16')
                    s0_ = np.zeros(data_shape, dtype='float16')
                    
                    s0 = [0:-1]
                    
                    print((row, col))
                    s0 = np.roll(d[0], (row, col), axis=(0,1))
                    s0_ = np.roll(d[3], [row, col], axis=[0,1])
                    d_new.append([s0, d[1], d[2], s0_, d[4]])
        """
    return d_new















































