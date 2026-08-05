# load an image as grid of numbers
# write a function that handles 3 functionalites:
# cropping, flipping and blurring

import cv2 as cv

def imageFilter(image: str):
    loadedImg = cv.imread(cv.samples.findFile(image))

    # define image properties
    y=0
    x=0
    w=1200
    h=1200

    # crop the image using numpy slicing
    cropped_img = loadedImg[y:y+h, x:x+w].copy()
    cv.imshow("cropped:", cropped_img)
    cv.waitKey(0)

    # flip the image
    print("Flipping image...")
    filpped_img = cv.flip(loadedImg, 1) # 0 flips vertically, 1 flips horizontally
    cv.imshow("flipped", filpped_img)
    cv.waitKey(0)

    # blur the image
    blurred_img = cv.blur(loadedImg, (10, 10))
    cv.imshow("blurred", blurred_img)
    cv.waitKey(0)

imageFilter("/Users/mac/Downloads/WhatsApp Image 2026-07-23 at 18.48.45.jpeg")
