import cv2
import numpy as np
import os

# Load the pre-trained model
model = cv2.dnn.readNetFromCaffe('/home/jasper/dev_ws/src/turtlebot_sim/commander_move_base/scripts/deploy.prototxt', '/home/jasper/dev_ws/src/turtlebot_sim/commander_move_base/scripts/mobilenet_iter_73000.caffemodel')

# Specify the directory containing reference images
reference_images_directory = '/home/jasper/Pictures/black_mug'

# Read all reference images from the directory
reference_images = []
for filename in os.listdir(reference_images_directory):
    if filename.endswith('.jpeg'):
        path = os.path.join(reference_images_directory, filename)
        print(path)
        print("gqewgd")
        reference_images.append(cv2.imread(path))

# Create a ORB object
orb = cv2.ORB_create(1000,1.2)

# Create a FLANN matcher
# flann = cv2.FlannBasedMatcher_create()

#FLANN_INDEX_KDTREE = 1
#index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#search_params = dict(checks=50)

#flann = cv2.FlannBasedMatcher(index_params, search_params)

brute_force = cv2.BFMatcher()

# Initialize variables
tracking = False
roi = None

# Capture video from the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors
    kp_frame = orb.detect(gray_frame, None)
    kp_frame, des_frame = orb.compute(gray_frame, kp_frame)

    # If tracking, update the tracker
    if tracking:
        # Match descriptors using FLANN
        for reference_image in reference_images:
            gray_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
            kp_ref = orb.detect(gray_ref, None)
            kp_ref, des_ref = orb.compute(gray_ref, kp_ref)

            # matches = flann.knnMatch(des_ref, des_frame, k=2)
            matches = brute_force.match(des_ref,des_frame)

            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            # If enough good matches are found, start tracking
            if len(good_matches) > 10:
                tracking = True
                break

        if tracking:
            # Draw bounding box and label on the frame
            (startX, startY, endX, endY) = (int(roi[0]), int(roi[1]), int(roi[0] + roi[2]), int(roi[1] + roi[3]))
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, 'Tracked Object', (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    else:
        # Perform object detection
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        model.setInput(blob)
        detections = model.forward()

        # Loop over the detections and draw bounding boxes
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:  # You can adjust the confidence threshold
                print(confidence)
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")

                # Extract the region of interest (ROI) from the frame;
                roi = frame[startY:endY, startX:endX]

                # Check if the object is found in any of the reference images
                for j, reference_image in enumerate(reference_images):
                    # Detect keypoints and compute descriptors for the reference image
                    kp_ref = orb.detect(cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY), None)
                    kp_ref, des_ref = orb.compute(cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY), kp_ref)
                    
                    # Match descriptors using FLANN
                    #matches = flann.knnMatch(des_ref, des_frame, k=2)
                    matches = brute_force.knnMatch(des_ref,des_frame,k=2)
                    

                    # Apply ratio test
                    good_matches = []
                    for m, n in matches:
                       if m.distance < 0.75 * n.distance:
                          good_matches.append(m)

                    # If enough good matches are found, start tracking
                    #if len(good_matches) > 10:
                     #   tracking = True
                      #  break
                    print(len(good_matches))
                    if len(good_matches) > 15:
                        # Draw bounding box and label on the frame for detection
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                        cv2.putText(frame, f'Object #{i}', (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        #results = cv2.drawMatchesKnn(des_ref, kp_ref, des_frame, kp_frame, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the frame
    cv2.imshow('Object Detection and Tracking', frame)
    #cv2.imshow(results)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
