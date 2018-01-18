import cv2

clicked = False

def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True

cameraCapture = cv2.VideoCapture(0)
cv2.namedWindow("Camera Window")
cv2.setMouseCallback("Camera Window", onMouse)

print 'Showing camera feed. Click window or press and key to stop'

success, frame = cameraCapture.read()
while success and cv2.waitKey(1) == -1 and not clicked:
    cv2.imshow("Camera Window", frame)
    success, frame = cameraCapture.read()

cv2.destroyWindow("Camera Window")
cameraCapture.release()
