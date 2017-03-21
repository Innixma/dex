# By Nick Erickson, Benjamin Hillmann, Sivaraman Rajaganapathy
# Contains functions for image capture
import globs as G
import win32gui, win32ui, win32con, win32api, win32com.client
import numpy as np
import time
import terminal_detection          
import convert_to_polar
import scipy.misc as smp
import skimage as skimage

class HexagonEmulator:

    def __init__(self, application, window_size=[140,140], capture_zoom=[0,0], rewards=[1,-1], mode='standard'): # window_size = x,y
        self.keys = np.array(['none', 'left_arrow', 'right_arrow'])
        self.action_dim = 3
        self.application = application
        self.shell = win32com.client.Dispatch("WScript.Shell")
        self.game_window = win32gui.FindWindow(None, self.application)
        self.window_size = window_size
        self.window_offset = [0,0]
        self.capture_size = [0,0]
        self.capture_offset = [0,0]
        self.capture_zoom = capture_zoom
        self.reward_alive = rewards[0]
        self.reward_terminal = rewards[1]
        self.mode = mode
        self.alive = False

        self.get_application_focus()
        
        self.configure()
        
        
        self.hwnd = win32gui.GetDesktopWindow()
        self.wDC = win32gui.GetWindowDC(self.hwnd)
        self.dcObj = win32ui.CreateDCFromHandle(self.wDC)
        self.cDC = self.dcObj.CreateCompatibleDC()
        self.dataBitMap = win32ui.CreateBitmap()
        self.dataBitMap.CreateCompatibleBitmap(self.dcObj, self.capture_size[0], self.capture_size[1])
        self.cDC.SelectObject(self.dataBitMap)
        
        self.curKey = 'enter'
        self.prevKey = 'enter'
        
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
    
        self.capture_offset[0] = 10
        self.capture_offset[1] = 31
        
        self.capture_size[0] = self.window_size[0] - self.capture_offset[0] - 10
        self.capture_size[1] = self.window_size[1] - self.capture_offset[1] - 10
        
        
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
      
        
        #print(G.y_size, G.x_size)
        #G.image = np.zeros((memory_size,G.y_size,G.x_size,3), dtype=np.uint8)
        #print(G.image.shape)
        self.cDC.BitBlt((0, 0), (self.capture_size[0], self.capture_size[1]), self.dcObj, (self.capture_offset[0], self.capture_offset[1]), win32con.SRCCOPY)
        #print(np.frombuffer(dataBitMap.GetBitmapBits(True), dtype=np.uint8).shape)
        image = np.frombuffer(self.dataBitMap.GetBitmapBits(True), dtype=np.uint8).reshape((self.capture_size[1], self.capture_size[0], 4))[:,:,:-1][:,:,::-1]       
        #print(image.shape)
        
        # ----------------------------------------------------
        # Polar Conversion
        if self.mode == 'polar':
            image = convert_to_polar.reproject_image_into_polar(image)[0].reshape((self.capture_size[1], self.capture_size[0], 3))
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
        
        tmpImage = skimage.color.rgb2gray(image)
        
        #global ZZ
        #ZZ += 1
        #if ZZ > 300:
        #    img = smp.toimage(tmpImage)
        #    smp.imsave('outfile' + str(ZZ % 36) + '.png', img)    
        
        # Following line commented out Feb 25 2017, due to potential issues caused.
        #tmpImage = skimage.exposure.rescale_intensity(tmpImage, out_range=(0, 255))
        
        # ----
        # NEW!!! Feb 28: Normalize pixels
        #print(np.mean(tmpImage))
        #tmpImage = tmpImage.astype('float32') / 128 - 1
        # ----
        tmpImage = tmpImage.astype('float16')
        #tmpImage = tmpImage.reshape(1, 1, tmpImage.shape[0], tmpImage.shape[1])
        tmpImage = tmpImage.reshape(1, tmpImage.shape[0], tmpImage.shape[1])
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
        terminal = 0
        if np.mean(data) >= 220:
            terminal = 1
        return(terminal)
    
    # Game state function
    def gameState(self, inKey=0):
        

        inKey = self.keys[inKey]
        #self.get_focus_light()
        self.release(self.curKey)
        self.press(inKey)
        
        self.prevKey = self.curKey
        self.curKey = inKey
        
        
        state = self.captureIm()
        
        # TODO: Replace this in future to check the prepared image state,
        # Much faster that way
        terminal = terminal_detection.check_terminal(state)
        
        state = self.prepareImage(state)

        if terminal:
            self.release(inKey)
            reward = self.reward_terminal
            self.alive = False
        else:
            reward = self.reward_alive
            
        return(state, reward, terminal)
