import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

img = 'Images/NL.jpg'
image = cv2.imread(img)
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gray = cv2.GaussianBlur(image, (7, 7), 0)
edged = cv2.Canny(gray, 10,20)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

img1 = img[:, :, 0]

fig = plt.figure()

a = fig.add_subplot(2, 2, 1)
plt.axis("off")
imgplot = plt.imshow(img)
a.set_title('original')

a = fig.add_subplot(2, 2, 2)
plt.axis("off")
imgplot = plt.imshow(img1)
a.set_title('viridis')

a = fig.add_subplot(2, 2, 3)
plt.axis("off")
imgplot = plt.imshow(edged)
a.set_title('edged')

a = fig.add_subplot(2, 2, 4)
plt.axis("off")
imgplot = plt.imshow(img1, cmap="hot")
a.set_title('hot')

plt.show()