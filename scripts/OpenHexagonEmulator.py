# By Nick Erickson, Benjamin Hillmann, Sivaraman Rajaganapathy
# Contains functions for image capture
import globs as G
import win32gui, win32ui, win32con, win32api, win32com.client
import numpy as np
import time
import terminal_detection          
#import convert_to_polar
import scipy.misc as smp
import skimage as skimage
from skimage import color
import skimage.transform as transf

class HexagonEmulator:

    def __init__(self, screen_info, screen_id=-1, screen_number=0, rewards=[1,-1], mode='standard'): # window_size = x,y
        self.keys = np.array(['none', 'left_arrow', 'right_arrow'])
        self.scale = screen_info.scale
        self.action_dim = 3
        self.application = screen_info.app
        self.screen_id = screen_id
        self.screen_number = screen_number
        self.shell = win32com.client.Dispatch("WScript.Shell")
        if screen_id == -1:
            self.game_window = win32gui.FindWindow(None, self.application)
        else:
            self.game_window = self.screen_id
        
        #self.window_size = screen_info.size
        self.window_size = [0,0]
        #self.window_offset = [(self.window_size[0]+10)*self.screen_number,0] # [0, 0]
        #self.capture_size = [0,0]
        self.capture_size = np.copy(screen_info.size)
        self.window_offset = [(self.capture_size[0]+10)*self.screen_number,0] # [0, 0]
        self.capture_offset = [0,0]
        self.capture_zoom = screen_info.zoom
        self.reward_alive = rewards[0]
        self.reward_terminal = rewards[1]
        self.mode = mode
        self.alive = False

        
        
        self.configure()
        self.get_application_focus()
        
        self.hwnd = win32gui.GetDesktopWindow()
        self.wDC = win32gui.GetWindowDC(self.hwnd)
        self.dcObj = win32ui.CreateDCFromHandle(self.wDC)
        self.cDC = self.dcObj.CreateCompatibleDC()
        self.dataBitMap = win32ui.CreateBitmap()
        self.dataBitMap.CreateCompatibleBitmap(self.dcObj, self.capture_size[0], self.capture_size[1])
        self.cDC.SelectObject(self.dataBitMap)
        
        self.curKey = 'enter'
        self.prevKey = 'enter'
        
        self.state_dim = np.copy(self.capture_size)
        self.state_dim = [self.state_dim[1], self.state_dim[0]]
        
        if self.scale != 1:
            self.state_dim[0] = int(self.state_dim[0]/self.scale)
            self.state_dim[1] = int(self.state_dim[1]/self.scale)
            
        print(self.state_dim)
    #==============================================================================
    
    def start_game(self):
        self.get_focus_light()
        #time.sleep(0.05)
        self.press('enter')
        time.sleep(0.05)
        #self.get_focus_light()
        self.release('enter')
        self.alive = True
        
    def end_game(self):
        self.get_focus_light()
        self.release(self.curKey)
        time.sleep(0.2)
        #self.get_focus_light()
        self.press('esc')
        time.sleep(0.05)
        #self.get_focus_light()
        self.release('esc')
        self.alive = False
        time.sleep(0.03)
    
    def get_focus_light(self):
        #hwnd = win32gui.FindWindow(None, self.application)
        #shell = win32com.client.Dispatch("WScript.Shell")
        #self.shell.SendKeys('%')
        try:
            win32gui.SetForegroundWindow(self.game_window) 
        except Exception as e:
            print(str(e))

    def get_application_focus(self):
        #hwnd = win32gui.FindWindow(None, self.application)
        #shell = win32com.client.Dispatch("WScript.Shell")
        self.shell.SendKeys('%')
        win32gui.SetForegroundWindow(self.game_window)
        win32gui.MoveWindow(self.game_window, self.window_offset[0], self.window_offset[1], self.window_size[0], self.window_size[1], True)
        
        if self.game_window == 0:
            print('ERROR: Can\'t find window')
            exit(1)
            
        return
    
    def press(self, *args):
        '''
        one press
        accepts as many arguments as you want. e.g. press('left_arrow', 'a','b').
        '''
        for i in args:
            if i != 'none':
                win32api.keybd_event(G.VK_CODE[i], 0, 0, 0)
    
    def release(self, *args):
        '''
        release depressed keys
        accepts as many arguments as you want.
        e.g. release('left_arrow', 'a','b').
        '''
        for i in args:
            if i != 'none':
                win32api.keybd_event(G.VK_CODE[i], 0, win32con.KEYEVENTF_KEYUP, 0)
    
    def configure(self):
        #win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.5)
    
        #self.capture_offset[0] = 10
        #self.capture_offset[1] = 31
        self.capture_offset[0] = 9
        self.capture_offset[1] = 32
        
        self.window_offset = [10,10]

        self.window_size[0] = self.capture_size[0] + self.capture_offset[0] + 9
        self.window_size[1] = self.capture_size[1] + self.capture_offset[1] + 9

        print(self.capture_size)
        print(self.window_size)

        #self.window_size[0] += 21
        #self.window_size[1] += 41
        
        #self.capture_size[0] = self.window_size[0] - self.capture_offset[0] - 10
        #self.capture_size[1] = self.window_size[1] - self.capture_offset[1] - 10
        
        
        # Center on image
        self.capture_offset[0] += self.window_offset[0]
        self.capture_offset[1] += self.window_offset[1]

        # Zoom in:
        self.capture_offset[0] += self.capture_zoom[0]
        self.capture_offset[1] += self.capture_zoom[1]

        self.capture_size[0] -= 2*self.capture_zoom[0]
        self.capture_size[1] -= 2*self.capture_zoom[1]
    
        if self.capture_size[0] % 2 == 1:
            self.capture_size[0] += 1
        if self.capture_size[1] % 2 == 1:
            self.capture_size[1] += 1
            
        # Marked for deletion:
        #if G.image_mode == 'polar':
        #    self.capture_size[1] -= 14

    # Capture 1 screenshot
    def captureIm(self):
        self.cDC.BitBlt((0, 0), (self.capture_size[0], self.capture_size[1]), self.dcObj, (self.capture_offset[0], self.capture_offset[1]), win32con.SRCCOPY)
        image = np.frombuffer(self.dataBitMap.GetBitmapBits(True), dtype=np.uint8).reshape((self.capture_size[1], self.capture_size[0], 4))[:,:,:-1][:,:,::-1]       

        # ----------------------------------------------------
        # Polar Conversion
        #if self.mode == 'polar':
        #    image = convert_to_polar.reproject_image_into_polar(image)[0].reshape((self.capture_size[1], self.capture_size[0], 3))
        #    image = image[14:][:]
        # ----------------------------------------------------
        
        
        
        # ----
        # Testing
        #img = smp.toimage(image)
        #smp.imsave('outfile.png', img)
        #img.show()
        #exit()
        # ----
    
        return image
        
    # Converts image to grayscale, and forces image to proper dimensions
    def prepareImage(self, image):
        
        tmpImage = color.rgb2gray(image).astype('float16')

        # Following line commented out Feb 25 2017, due to potential issues caused.
        #tmpImage = skimage.exposure.rescale_intensity(tmpImage, out_range=(0, 255))

        if self.scale != 1:
            tmpImage = transf.downscale_local_mean(tmpImage, (self.scale,self.scale)) # Downsample
        
        tmpImage = tmpImage.reshape(tmpImage.shape[0], tmpImage.shape[1], 1) # Tensorflow
        return tmpImage
        
    # Free resources
    def freeResources(self):
    
        self.dcObj.DeleteDC()
        self.cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, self.wDC)
        win32gui.DeleteObject(self.dataBitMap.GetHandle())
        return()
        
    # Check for game termination (UNUSED)
    def termination(self, data):
        t = 0
        if np.mean(data) >= 220:
            t = 1
        return t
    
    # Game state function
    def step(self, inKey=0):
        if self.alive == False:
            print('invalid step call')
            return False

        inKey = self.keys[inKey]
        #self.get_focus_light()
        self.release(self.curKey)
        self.press(inKey)
        
        self.prevKey = self.curKey
        self.curKey = inKey
        
        s = self.captureIm()
        s = self.prepareImage(s)
        t = terminal_detection.check_terminal(s)
        
        if t:
            self.release(inKey)
            r = self.reward_terminal
            self.alive = False
            self.end_game()
        else:
            r = self.reward_alive
            
        return(s, r, t)
