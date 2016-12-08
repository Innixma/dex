# By Nick Erickson, Benjamin Hillmann, Sivaraman Rajaganapathy
# Contains functions for image capture (may need to be moved to q-learning code for speed)
import globs as G
import win32gui, win32ui, win32con, win32api, win32com.client
import random
import numpy as np
import scipy.misc as smp
import time
import terminal_detection
           
#==============================================================================
# 
#==============================================================================
def press(*args):
    '''
    one press, one release.
    accepts as many arguments as you want. e.g. press('left_arrow', 'a','b').
    '''
    for i in args:
        if i != 'none':
            win32api.keybd_event(G.VK_CODE[i], 0, 0, 0)

#==============================================================================



#==============================================================================
#             
#==============================================================================
def release(*args):
    '''
    release depressed keys
    accepts as many arguments as you want.
    e.g. release('left_arrow', 'a','b').
    '''
    for i in args:
        if i != 'none':
            win32api.keybd_event(G.VK_CODE[i], 0, win32con.KEYEVENTF_KEYUP, 0)
#==============================================================================

#==============================================================================
# 
#==============================================================================

hwnd = win32gui.FindWindow(None, G.application)
shell = win32com.client.Dispatch("WScript.Shell")
shell.SendKeys('%')
win32gui.SetForegroundWindow(hwnd)
win32gui.MoveWindow(hwnd, G.x_offset, G.y_offset, G.x_size, G.y_size, True)

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

#'memory_size' marked for deletion
memory_size = 1 # Number of frames to keep in memory for backpropagation

#==============================================================================



#==============================================================================
# capture 1 screenshot
#==============================================================================
def captureIm():
  
    
    G.image = np.zeros((memory_size,h,w,3), dtype=np.uint8)
    cDC.BitBlt((G.x_offset, G.y_offset), (w, h), dcObj, (G.x_offset, G.y_offset), win32con.SRCCOPY)     
    G.image = np.frombuffer(dataBitMap.GetBitmapBits(True), dtype=np.uint8).reshape((h, w, 4))[:,:,:-1][:,:,::-1]       

    return G.image
#==============================================================================

#==============================================================================
# Free resources
#==============================================================================
def freeResources():

    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())
    return()
    
#==============================================================================
    


#==============================================================================
# Check for game termination (Replace with Nick's method)
#==============================================================================
def termination(data):
    terminal = 0
    if np.mean(data) >= 220:
        terminal = 1
    return(terminal)


#==============================================================================



#==============================================================================
# Game state function
#==============================================================================
def gameState(inKey):
    
    G.prevKey = G.curKey
    G.curKey = inKey
    release(G.prevKey)
    press(inKey)
    
    state = captureIm()
    
    #img = smp.toimage(state)
    #img.show()
    #time.sleep(4)
    
    optimal_move = terminal_detection.get_move(state, G.keys)
    if optimal_move == 'esc':
        release(inKey)
        press('enter')
        time.sleep(0.01)
        release('enter')
        terminal_detection.reset_globs()
        terminal = 1
    else:
        terminal = 0
    #terminal = termination(state)
    if terminal == 0:
        reward = 0.01
    else:
        reward = -1
    #print(optimal_move)
    return(state, reward, terminal)
#==============================================================================


'''

#%%
memory_size = 100



    
#imageData = np.zeros((memory_size,h,w,3), dtype=np.uint8)
st = time.time()
for i in range(0,memory_size,1):
    
    #imageData[i] = captureIm()
    print(gameState()[1])

et = time.time()
print(et-st)
print('images per second')
print(100/(et-st))    
        
#smp.toimage(imageData[0]) 


'''
