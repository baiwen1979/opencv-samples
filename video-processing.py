import cv2
import numpy as np

fourcc_yuv = cv2.VideoWriter_fourcc('I', '4', '2', '0')
fourcc_mpg = cv2.VideoWriter_fourcc('P', 'I', 'M', '1')
fourcc_mp4 = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
fourcc_ogv = cv2.VideoWriter_fourcc('T', 'H', 'E', 'O')
fourcc_flv = cv2.VideoWriter_fourcc('F', 'L', 'V', '1')

def testVideoReadWrite():
    # Open the video file
    videoCapture = cv2.VideoCapture('videos/anim-logo.mov')
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    print "fps:", fps
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print "size:", size

    videoWriter = cv2.VideoWriter("videos/anim-logo.avi", 
        fourcc_yuv, fps, size)
    success, frame = videoCapture.read()

    while success:
        videoWriter.write(frame)
        success, frame = videoCapture.read()
    
def test_VideoCameraCap():
    # Open the camera
    cameraCapture = cv2.VideoCapture(0)
    fps = 30
    size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter('videos/my-video.avi', fourcc_yuv, fps, size)
    print "fps:", fps
    print "size:", size
    print "Capturing the video ..."

    success, frame = cameraCapture.read()
    numFramesRemaining = 10 * fps - 1 #only capture 10 seconds
    while success and numFramesRemaining > 0 :
        videoWriter.write(frame)
        success, frame = cameraCapture.read()
        numFramesRemaining -= 1
    cameraCapture.release()

if __name__ == '__main__':
    img = cv2.imread('images/car.jpg')
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

