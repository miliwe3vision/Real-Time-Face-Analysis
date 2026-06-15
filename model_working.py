import cv2
import tensorflow as tf
import numpy as np

# ==========================
# Load Emotion Model
# ==========================
model_pred = tf.keras.models.load_model(
    "CKmodel.h5",
    compile=False
)

model_pred.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================
# Load Face Detector
# ==========================
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# ==========================
# Labels
# ==========================
exp = ['Angry', 'Happy', 'Sad', 'Surprise']

MODEL_MEAN_VALUES = (
    78.4263377603,
    87.7689143744,
    114.895847746
)

age_list = [
    '(0-2)', '(4-6)', '(8-12)', '(15-20)',
    '(25-32)', '(38-43)', '(48-53)', '(60-100)'
]

gender_list = ['Male', 'Female']

# ==========================
# Load Age & Gender Models
# ==========================
age_net = cv2.dnn.readNetFromCaffe(
    "deploy_age.prototxt",
    "age_net.caffemodel"
)

gender_net = cv2.dnn.readNetFromCaffe(
    "deploy_gender.prototxt",
    "gender_net.caffemodel"
)


def detect_face(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 255, 255),
            2
        )

        # ======================
        # Emotion Detection
        # ======================
        face_roi = frame[y:y+h, x:x+w]

        try:
            emotion_img = cv2.resize(face_roi, (48, 48))
            emotion_img = emotion_img.astype("float32")
            emotion_img = emotion_img.reshape(
                1, 48, 48, 3
            )

            pred = model_pred.predict(
                emotion_img,
                verbose=0
            )

            emotion = exp[np.argmax(pred)]

            cv2.putText(
                frame,
                emotion,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        except:
            pass

        # ======================
        # Age + Gender Detection
        # ======================
        try:
            blob = cv2.dnn.blobFromImage(
                cv2.resize(face_roi, (227, 227)),
                1.0,
                (227, 227),
                MODEL_MEAN_VALUES,
                swapRB=False
            )

            # Age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]

            # Gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[
                gender_preds[0].argmax()
            ]

            cv2.putText(
                frame,
                f"Age: {age}",
                (x, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Gender: {gender}",
                (x, y + h + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        except:
            pass

    return frame


# ==========================
# Webcam
# ==========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

writer = cv2.VideoWriter(
    "video_out.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    15,
    (width, height)
)

print("Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = detect_face(frame)

    writer.write(frame)

    cv2.imshow(
        "Emotion Age Gender Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()