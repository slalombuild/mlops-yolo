import cv2 
import numpy as np
def threshold_otsu(
    image: np.ndarray, max_value: int = 255, inverse: bool = False
) -> np.ndarray:

    flags = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    flags += cv2.THRESH_OTSU
    _, img = cv2.threshold(image, 0, max_value, flags)
    return img

#Read the image 
img = cv2.imread("photos/court_detection.jpg")
#Create boundary coordinates 
lower = np.array([180, 180, 100])
upper = np.array([255, 255, 255])
#Create a mask using the boundary coordinates
mask = cv2.inRange(img, lower, upper)
#Use the mask to create a masked image
masked_img = cv2.bitwise_and(img, img, mask=mask)
gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
corners = cv2.cornerHarris(gray, 9, 3, 0.01)
zero_array = np.zeros_like(img)
corners = cv2.normalize(corners, zero_array, 255, 0, cv2.NORM_MINMAX).astype(np.uint8)
thresh = threshold_otsu(corners)
dilated = cv2.dilate(
        thresh,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
#Show the masked image
cv2.imshow("Masked Image", dilated)
cv2.waitKey(0)