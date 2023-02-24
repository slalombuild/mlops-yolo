import cv2
import numpy as np

for i in range(0, 1):
    # Read in the image containing the tennis ball
    img = cv2.imread(f"videos/clips/australian_open/australian_open_1/frame_270.jpg")

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the range of HSV values that correspond to the color of a tennis ball
    lower_range = np.array([20, 100, 100])  # lower range for hue, saturation, and value
    upper_range = np.array([50, 255, 255])  # upper range for hue, saturation, and value

    # Create a binary mask of the pixels that fall within the specified HSV range
    mask = cv2.inRange(hsv, lower_range, upper_range)

    # Apply the mask to the original image to extract the tennis ball
    result = cv2.bitwise_and(img, img, mask=mask)

    # Convert to gray scale
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise and enhance edges
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect circles using HoughCircles function
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, 1, 50, param1=10, param2=3, minRadius=1, maxRadius=4
    )

    # Draw detected circles on the original image
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for x, y, r in circles:
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
            cv2.putText(
                img,
                "tennis ball",
                (x, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    # Display the output image
    cv2.imshow("Tennis ball detection", img)
    cv2.waitKey(0)
