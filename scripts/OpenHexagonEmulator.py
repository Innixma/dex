# By Nick Erickson, Benjamin Hillmann, Sivaraman Rajaganapathy
# Contains functions for image capture (may need to be moved to q-learning code for speed)
import globs as G
import win32gui, win32ui, win32con, win32api, win32com.client
import random
import numpy as np
import scipy.misc as smp
import time
import terminal_detection          
import convert_to_polar
 
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

def configure():
    #time.sleep(0.5)
    #win32gui.SetForegroundWindow(hwnd)
    time.sleep(0.5)
    #img = captureIm()
    #img = img[31:G.y_size-10,10:G.x_size-10,:]
    #print(img.shape)
    #for pixel in img[0]:
    #    print(pixel)
    #img = smp.toimage(img)
    #img.show()

    G.x_offset = 10
    G.y_offset = 31
    
    G.x_size = G.x_size - G.x_offset - 10
    G.y_size = G.y_size - G.y_offset - 10
    
    # Zoom in:
        
    G.x_offset += G.x_zoom
    G.x_size -= G.x_zoom*2
    G.y_offset += G.y_zoom
    G.y_size -= G.y_zoom*2
    
    if G.y_size % 2 == 1:
        G.y_size += 1
    if G.x_size % 2 == 1:
        G.x_size += 1

    G.x_size_final = G.x_size
    G.y_size_final = G.y_size - 14
        
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

configure()

#rect = (0, 0, G.x_size, G.y_size)
#frame = win32ui.CreateFrame()
#frame.CreateWindow(None, "New Window", 0, rect)
        
#time.sleep(1)
#hwnd = frame.GetSafeHwnd()
hwnd = win32gui.GetDesktopWindow()

wDC = win32gui.GetWindowDC(hwnd)
dcObj = win32ui.CreateDCFromHandle(wDC)
cDC = dcObj.CreateCompatibleDC()
dataBitMap = win32ui.CreateBitmap()
dataBitMap.CreateCompatibleBitmap(dcObj, G.x_size, G.y_size)
cDC.SelectObject(dataBitMap)

#'memory_size' marked for deletion
memory_size = 1 # Number of frames to keep in memory for backpropagation

#==============================================================================



#==============================================================================
# capture 1 screenshot
#==============================================================================
def captureIm():
  
    
    #print(G.y_size, G.x_size)
    #G.image = np.zeros((memory_size,G.y_size,G.x_size,3), dtype=np.uint8)
    #print(G.image.shape)
    cDC.BitBlt((0, 0), (G.x_size, G.y_size), dcObj, (G.x_offset, G.y_offset), win32con.SRCCOPY)
    #print(np.frombuffer(dataBitMap.GetBitmapBits(True), dtype=np.uint8).shape)
    image = np.frombuffer(dataBitMap.GetBitmapBits(True), dtype=np.uint8).reshape((G.y_size, G.x_size, 4))[:,:,:-1][:,:,::-1]       
    #print(image.shape)
    
    # ----------------------------------------------------
    # Polar Conversion
    #image = convert_to_polar.reproject_image_into_polar(image)[0].reshape((G.y_size, G.x_size, 3))
    #image = image[14:][:]
    # ----------------------------------------------------
    
    #img = smp.toimage(state)
    #img.show()
    #smp.imsave('outfile.png', img)

    return image
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
    
    optimal_move = terminal_detection.get_move(state, G.keys)
    if optimal_move == 'esc':
        release(inKey)
        terminal = 1
    else:
        terminal = 0
    #terminal = termination(state)
    if terminal == 0:
        reward = G.REWARD_ALIVE
    else:
        reward = G.REWARD_TERMINAL
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
