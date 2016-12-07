import win32gui, win32ui, win32con, win32api, win32com.client
import random
import numpy as np
import scipy.misc as smp
import time
#import process_move
import process_moveSIVR as process_move





hwnd = win32gui.FindWindow(None, 'Open Hexagon 1.92 - by vittorio romeo')
shell = win32com.client.Dispatch("WScript.Shell")
shell.SendKeys('%')
win32gui.SetForegroundWindow(hwnd)
win32gui.MoveWindow(hwnd, 0, 0, 500, 500, True)

if hwnd == 0:
    print('ERROR: Can\'t find window')
    exit(1)

coords = win32gui.GetWindowRect(hwnd)

w = coords[2] - coords[0]
h = coords[3] - coords[1]

hwnd = win32gui.GetDesktopWindow()

wDC = win32gui.GetWindowDC(hwnd)
dcObj = win32ui.CreateDCFromHandle(wDC)
cDC = dcObj.CreateCompatibleDC()
dataBitMap = win32ui.CreateBitmap()
dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
cDC.SelectObject(dataBitMap)

num_iters = 10000
num_tries = 2
memory_size = 100 # Number of frames to keep in memory for backpropagation

keys = np.array(['none', 'left_arrow', 'right_arrow', 'enter', 'esc'])

imageData = np.zeros((memory_size,h,w,3), dtype=np.uint8)

#key_moves = np.zeros(memory_size, dtype=np.uint8)
#process_move.reset_globs()


# Run a number of tries
for j in range(num_tries):
    imageData = np.zeros((memory_size,h,w,3), dtype=np.uint8)
    key_moves = np.zeros(memory_size, dtype=np.uint8)
    process_move.reset_globs()
    
    press('enter')
    time.sleep(.01)
    release('enter')
    
    start_time = time.time()
    prevKey = 'enter'
    for i in range(num_iters):
        cDC.BitBlt((0, 0), (w, h), dcObj, (0, 0), win32con.SRCCOPY)
        
        # Get image, memory of images
        imageData[i % memory_size] = np.frombuffer(dataBitMap.GetBitmapBits(True), dtype=np.uint8).reshape((h, w, 4))[:,:,:-1][:,:,::-1]       

        # Function to process key given pixels
        key = process_move.get_move(imageData[i % memory_size], keys)
        
        # Memory of moves
        key_moves[i % memory_size] = np.where(keys==key)[0]
        if key == None:
            release(prevKey)
        elif key == 'esc':
            defeat = time.time()
            print('Run ' + str(j) + ' survived ' + str("%.2f" % (defeat - start_time)) + 's')
            break
        else:
            release(prevKey)
            press(key)
            prevKey = key
    
    release(prevKey)
    press('esc')
    time.sleep(.01)
    release('esc')
    print("--- %s fps ---" % ((i+1)/(time.time() - start_time))) 

# Free Resources
dcObj.DeleteDC()
cDC.DeleteDC()
win32gui.ReleaseDC(hwnd, wDC)
win32gui.DeleteObject(dataBitMap.GetHandle())
    
#%%
img = smp.toimage(imageData[58])       # Create a PIL image
img.show()                      # View in default viewer

