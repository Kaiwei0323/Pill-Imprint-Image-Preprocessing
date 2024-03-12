import cv2
import numpy as np
import time

"""
def blackout_outside_pill(img):
    print("MSER Started")
    t0 = time.perf_counter()
    mser = cv2.MSER_create()

    # Convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_copy = img.copy()

    # detect regions in gray scale image
    regions, bboxes = mser.detectRegions(gray)

    # Find the largest region based on area
    largest_region_index = np.argmax([cv2.contourArea(p) for p in regions])
    largest_region = regions[largest_region_index]

    # Create a mask for the entire image
    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    cv2.drawContours(mask, [largest_region], -1, (255, 255, 255), thickness=cv2.FILLED)

    # Black out the area outside the largest region
    img_outside_pill = cv2.bitwise_and(img, img, mask=mask)

    print("MSER Ended")
    mser_execution_time = time.perf_counter() - t0
    print("MSER EXECUTION TIME: ", mser_execution_time)

    return img_outside_pill
"""
def blackout_outside_pill(img):
    print("MSER Started")
    t0 = time.perf_counter()

    # Convert to LAB color space
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_img)

    # Apply adaptive thresholding to the L channel
    _, thresholded_l = cv2.threshold(l_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img_copy = img.copy()

    # detect regions in thresholded image
    mser = cv2.MSER_create()
    regions, bboxes = mser.detectRegions(thresholded_l)

    # Find the largest region based on area
    largest_region_index = np.argmax([cv2.contourArea(p) for p in regions])
    largest_region = regions[largest_region_index]

    # Create a mask for the entire image
    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    cv2.drawContours(mask, [largest_region], -1, (255, 255, 255), thickness=cv2.FILLED)

    # Black out the area outside the largest region
    img_outside_pill = cv2.bitwise_and(img, img, mask=mask)

    print("MSER Ended")
    mser_execution_time = time.perf_counter() - t0
    print("MSER EXECUTION TIME: ", mser_execution_time)

    return img_outside_pill

if __name__ == '__main__':
    file_location = input("File Name: ")
    img = cv2.imread(file_location)

    if img is not None:
        img_outside_pill = blackout_outside_pill(img)
        cv2.imshow('img_outside_pill', img_outside_pill)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Error: Unable to read the image at {file_location}")