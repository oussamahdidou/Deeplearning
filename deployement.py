import cv2
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model("best_model.h5")


labels = ["no-plastic", "plastic"]

video_capture = cv2.VideoCapture(
    "Observation of an underwater drone (V6) by another drone. Credit- OIST-NTT Communications.mp4"
)

while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    
    resized_frame = cv2.resize(
        frame, (224, 224)
    )  
    resized_frame = np.expand_dims(resized_frame, axis=0)  

    
    predictions = model.predict(resized_frame)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = labels[predicted_class_index]
    probability = predictions[0][predicted_class_index]
    text = f"{predicted_class_label}: {probability:.2f}"
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # Display prediction and probability on the frame
    # if probability >= 0.6:
    #     text = f"{predicted_class_label}: {probability:.2f}"
    #     cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # else:
    #     text = "I don't know"
    #     cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # if predicted_class_label == "plastic":
    #     # Convert frame to grayscale
    #     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     # Apply binary thresholding to extract plastic object
    #     ret, thresh = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)
    #     # Find contours
    #     contours, _ = cv2.findContours(
    #         thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    #     )
    #     # Draw bounding box around each contour
    #     for contour in contours:
    #         x, y, w, h = cv2.boundingRect(contour)
    #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
