# By Nick Erickson, Benjamin Hillmann, Sivaraman Rajaganapathy
# Contains functions for image capture (may need to be moved to q-learning code for speed)
import win32gui, win32ui, win32con, win32api, win32com.client
import random
import numpy as np
import scipy.misc as smp
import time

'''
#==============================================================================
# 
#==============================================================================
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

#'memory_size' marked for deletion
memory_size = 1 # Number of frames to keep in memory for backpropagation
#==============================================================================
'''

#==============================================================================
# Open Hexagon Emulator
#==============================================================================
class OpenHexagonEmulator:

    
    def __init__(self):
        pass
    
    def __enter__(self):
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
        
        #'memory_size' marked for deletion
        memory_size = 1 # Number of frames to keep in memory for backpropagation        
        return(self)
    
    #==============================================================================
    # capture 1 screenshot
    #==============================================================================
    def captureIm(self):
    

    
        imageData = np.zeros((memory_size,h,w,3), dtype=np.uint8)
        cDC.BitBlt((0, 0), (w, h), dcObj, (0, 0), win32con.SRCCOPY)     
        imageData = np.frombuffer(dataBitMap.GetBitmapBits(True), dtype=np.uint8).reshape((h, w, 4))[:,:,:-1][:,:,::-1]       
        
    

    
        return imageData
    #==============================================================================

    #==============================================================================
    # Free resources
    #==============================================================================
    def __exit__(self):

        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())
        
    #==============================================================================
    


memory_size = 100
imageData = np.zeros((memory_size,Hic_obj.h,Hic_obj.w,3), dtype=np.uint8)

with OpenHexagonEmulator() as hObj
hObj = OpenHexagonEmulator()
st = time.time()
for i in range(0,memory_size,1):
    
    imageData[i] = hObj.captureIm()

et = time.time()
print(et-st)
print('images per second')
print(100/(et-st))    
    




'''
memory_size = 100
imageData = np.zeros((memory_size,Hic_obj.h,Hic_obj.w,3), dtype=np.uint8)
#capIm = Hic_obj.captureIm()

st = time.time()
for i in range(0,memory_size,1):
    imageData[i] = captureIm()

et = time.time()
print(et-st)
print('images per second')
print(100/(et-st))
'''
'''

#==============================================================================
# capture 1 screenshot
#==============================================================================
def captureIm(self):

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
    
    #'memory_size' marked for deletion
    memory_size = 1 # Number of frames to keep in memory for backpropagation

    imageData = np.zeros((memory_size,h,w,3), dtype=np.uint8)
    cDC.BitBlt((0, 0), (w, h), dcObj, (0, 0), win32con.SRCCOPY)     
    imageData = np.frombuffer(dataBitMap.GetBitmapBits(True), dtype=np.uint8).reshape((h, w, 4))[:,:,:-1][:,:,::-1]       
    

    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())

    return imageData
#==============================================================================
'''
    