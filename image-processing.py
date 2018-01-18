import numpy as np
import cv2
import os

# Create a 32 * 32 black image
img = np.zeros((32, 32), dtype = np.uint8)
# Fill the vertical gradient gray color
for i in range(0, img.shape[0]):
    color = i * 255 / img.shape[0]
    for j in range(0, img.shape[1]):
        img[i, j] = color
# Save the image as a PNG image file
cv2.imwrite('images/black.png', img)
print img.shape
# Color the 8bit gray color to BGR(RGB) color
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
print img.shape

# Read the original JPG image file
image = cv2.imread("images/car.jpg")
print image.item(300, 400, 2)
image.itemset((300, 400, 2), 255)
# Save it as PNG image file
cv2.imwrite('images/car.png', image)
image[:, :, 0] = 0
cv2.imwrite('images/car-yellow.png', image)
cropImage = image[100:500, 100:700]
cv2.imwrite('images/car-crop.png', cropImage)
print "shape : ", image.shape
print "size : ", image.size
print "dtype :", image.dtype
# Read the image file as grayscale image, and save it as gray image
grayImage = cv2.imread('images/car.png', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('images/gray-car.jpg', grayImage)

# Make an array of 120000 random bytes
randomByteArray = bytearray(os.urandom(300 * 400))
flatNumpyArray = np.array(randomByteArray)
# convert the array to make a 400*300 grayscale image.
grayImage = flatNumpyArray.reshape(300, 400)
cv2.imwrite('images/randomGray.png', grayImage)
# convert the array to make a 400*300 color image
bgrImage = flatNumpyArray.reshape(100, 400, 3)
cv2.imwrite('images/randomColor.png', bgrImage)


