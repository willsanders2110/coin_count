import cv2
import imutils
import numpy as np

lower_thresh = 150
upper_thresh = 200

silver_coin_boundary = ()
copper_coin_boundary = ()
bi_metallic_coin_boundary = ()

camera = cv2.VideoCapture(0)

while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    colour_frame = frame.copy()
    # if the frame could not be grabbed, then we have reached the end of the video
    if not grabbed:
        break

    # Do work in here
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (7, 7), 0)
    # thresh = cv2.threshold(grey, lower_thresh, upper_thresh, cv2.THRESH_BINARY)[1]

    # circles = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, 1, 50)
    #
    # if circles is not None:
    #     circles = np.round(circles[0, :]).astype("int")
    #
    #     for (x, y, r) in circles:
    #         # cv2.circle(frame, (x, y), r, (0, 0, 255, 4))
    #         # cv2.rectangle(frame, (x-5, y-5), (x+5, y+5), (0, 0, 255), 1)
    #         cv2.putText(frame, "circle", (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    edged = cv2.Canny(grey, 75, 200)
    # TODO: Have all coins separated using erosion and dilation
    # # perform edge detection, then perform a erosion + dilation to
    # # open gaps in between object edges
    # edged = cv2.erode(grey, None, iterations=2)
    # edged = cv2.dilate(edged, None, iterations=1)

    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    number_of_coins = 0
    number_of_circular_coins = 0
    for (i, c) in enumerate(contours):


        # compute the convex hull of the contour, then use the area of the
        # original contour and the area of the convex hull to compute the
        # solidity
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        hullArea = cv2.contourArea(hull)

        ((x, y), radius) = cv2.minEnclosingCircle(c)

        mask = np.zeros(frame.shape[:2], dtype="uint8")
        cv2.circle(mask, (int(x), int(y)), int(radius)+10, 255, -1)

        masked = cv2.bitwise_and(grey, grey, mask=mask)
        masked_colour = cv2.bitwise_and(colour_frame, colour_frame, mask=mask)

        # cv2.imshow('mask', masked_colour)
        # # Checking Circles
        # circles = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, 2, 100)
        #
        # if circles is not None:
        #     circles = np.round(circles[0, :]).astype("int")
        #     number_of_circular_coins += 1
        #     for (x, y, r) in circles:
        #         cv2.putText(frame, "circle", (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # cv2.waitKey(0)

        # peri = cv2.arcLength(c, True)
        # approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        # print(len(approx))
        # cv2.putText(frame, "{}".format(int(radius)), (int(x)-10, int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        # (0, 255, 0), 2)

        try:
            solidity = area / float(hullArea)
        except ZeroDivisionError:
            solidity = 0

        # if the solidity is high, then we are examining an `O`
        if solidity > 0.9 and radius > 10:
            number_of_coins += 1

        # for contour in contours:
        cv2.drawContours(frame, [c], 0, (0, 255, 0), 2)

    cv2.putText(frame, "No. Coins: {}".format(number_of_coins), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                4)
    # cv2.putText(frame, "Circular. Coins: {}".format(number_of_circular_coins), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #             (0, 0, 255), 4)

    # show the frame and record if the user presses a key
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the loop
    if key == ord("q"):
        break

    # cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
