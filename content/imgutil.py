import numpy as np
import cv2
import arg


def rotate_noclip(img, angle):

    #get image dimensions to compute the center
    h, w = img.shape[:2]
    cx, cy = w//2, h//2

    #get rotation matrix (-angle for clockwise rotation
    Mat = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)

    #compute the sine and cosine which are the rotation/angular components of rotation matrix
    cos = np.abs(Mat[0, 0])
    sin = np.abs(Mat[0, 1])

    #compute new image bounds
    bW = int((h*sin) + (w*cos))
    bH = int((h*cos) + (w*sin))

    #modify rotation matrix
    Mat[0, 2] += bW/2 - cx
    Mat[1, 2] += bH/2 - cy

    #apply the modification (apply new rotation matrix to the image)
    return cv2.warpAffine(img, Mat, (bW, bH))


if len(sys.argv) > 1:
    img = cv2.imread(argv[1])
    prefix = argv[1][-3:]
    for arg in sys.argv[2:-1]:
        if arg in ["--r90", "--rotate", "--rotate90"]:
            rotate_noclip(img, 90)
        elif arg in ["--r180", "--rotate180"]:
            rotate_noclip(img, 180)
        elif arg in ["--r270", "-rotate270", "--r-90", "rotate-90"]:
            rotate_noclip(img, 270)
        elif arg in ["--grey", "--g", "--gray"]:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif arg in ["--RGB", "--color", "c"]:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite("new_img.jpg", img)

