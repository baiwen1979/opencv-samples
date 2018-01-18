# coding=utf-8
import cv2
from managers import WindowManager, CaptureManager
import filters

# Begin of Class Cameo
class Cameo(object):
    def __init__ (self):
        self._windowManager = WindowManager('Cameo', self.onKeypress)
        self._captureManager = CaptureManager(
            cv2.VideoCapture(0), self._windowManager, True)
        
        # 使用的过滤器
        self._sharpenFilter = filters.FindEdgesFilter()
    
    def run(self):
        """ Run the main loop."""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            # TODO: Filter the frame
            filters.strokeEdges(frame, frame)
            self._sharpenFilter.apply(frame, frame)

            self._captureManager.exitFrame()
            self._windowManager.processEvents()
    
    def onKeypress(self, keycode):
        """ Handle a keypress
        space -> Take a screenshot
        tab -> Start/stop recording a screencast
        escape -> Quit
        """
        if keycode == 32: # space
            self._captureManager.writeImage('images/screenshot.png')
        elif keycode == 9: # tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('videos/screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 27: # escape
            self._windowManager.destroyWindow()
# End of Class Cameo

# main 
if __name__ == "__main__":
    Cameo().run()
    