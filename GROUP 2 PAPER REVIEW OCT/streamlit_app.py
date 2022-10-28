import streamlit as st
from PIL import Image, ImageColor
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import pathlib
import joblib

model = joblib.load(open('model.clf', 'rb'))


RTC_CONFIGURATION = RTCConfiguration(
{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Set page configs. Get emoji names from WebFx
st.set_page_config(page_title="Real-time Face Detection", page_icon="image.png", layout="centered")

# -------------Header Section------------------------------------------------

title = '<p style="text-align: center;font-size: 40px;font-weight: 550; "> Real-time Face Detection</p>'
st.markdown(title, unsafe_allow_html=True)

st.markdown(
    "Face Recognition using *Dlib* *and* *OpenCV*.")

supported_modes = "<html> " \
                  "<body><div> <b>Supported Face Detection Modes (Change modes from sidebar menu)</b>" \
                  "<ul><li>Webcam Video Realtime</li></ul>" \
                  "</div></body></html>"
st.markdown(supported_modes, unsafe_allow_html=True)

st.warning("NOTE : Click the arrow icon at Top-Left to open Sidebar menu. ")

# -------------Sidebar Section------------------------------------------------

detection_mode = None
# bounding box thickness
bbox_thickness = 3
# bounding box color
bbox_color = (0, 255, 0)

with st.sidebar:
    st.image("image.png", width=260)

    title = '<p style="font-size: 25px;font-weight: 550;">Face Detection Settings</p>'
    st.markdown(title, unsafe_allow_html=True)

    # choose the mode for detection
    mode = st.radio("Choose Face Detection Mode", ('', 'Webcam Real-time'), index=1)
    if mode == "Webcam Image Capture":
        detection_mode = mode
    elif mode == 'Webcam Real-time':
        detection_mode = mode

    # Get bbox color and convert from hex to rgb
    bbox_color = ImageColor.getcolor(str(st.color_picker(label="Bounding Box Color", value="#00FF00")), "RGB")

    # ste bbox thickness
    bbox_thickness = st.slider("Bounding Box Thickness", min_value=1, max_value=30,
                               help="Sets the thickness of bounding boxes",
                               value=bbox_thickness)

    st.info("NOTE : Finetune the above paramters."
            " This is for users satisfaction")

    # line break
    st.markdown(" ")

    # About the programmer
    st.markdown("## Made by *Joshua Sofowora(Group Lead), Oluwatimilehin Folarin, Benard Zephaniah, Olabisi Oluwale Anthony, Favour James, Khaya Biyela, Marini Phahlamohlaka* \U0001F609")
    st.markdown("[*Github Repo*](https://github.com/chosenking1/facialAuthModel)")


# -------------Webcam Real-time Section------------------------------------------------


if detection_mode == "Webcam Real-time":

    # load face detection model
    cascPath = pathlib.Path(cv2.__file__).parent.absolute()/"data/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(str(cascPath))

    st.warning("NOTE : In order to use this mode, you need to give webcam access. "
               "After clicking 'Start' , it takes about 10-20 seconds to ready the webcam.")

    spinner_message = "Wait a sec, getting some things done..."

    with st.spinner(spinner_message):
        class VideoProcessor:
            def recv(self, frame):
                img = frame.to_ndarray(format = 'bgr24')
                faces = face_cascade.detectMultiScale(image=img, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

                for (x, y, w, h) in faces:
                    # Draw rectangle over face
                    cv2.rectangle(img = img, pt1 = (x, y), pt2 = (x + w, y + h), color = (0, 255, 0), thickness = 2)
                    
                    # Do preprocessing based on model
                    # face_crop = img[y:y + h, x:x + w]
                    # face_crop = cv2.resize(face_crop, (224, 224))
                    # face_crop = img_to_array(face_crop)
                    # face_crop = face_crop / 255
                    # face_crop = np.expand_dims(face_crop, axis = 0)
                    
                    import pickle
                    import face_recognition


                    def training(img_path, knn_clf=None, model_path=None, threshold=0.6): # 6 needs 40+ accuracy, 4 needs 60+ accuracy
                        if knn_clf is None and model_path is None:
                            raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")
                        # Load a trained KNN model (if one was passed in)
                        if knn_clf is None:
                            with open(model_path, 'rb') as f:
                                knn_clf = pickle.load(f)
                        # Load image file and find face locations
                        img = img_path
                        face_box = face_recognition.face_locations(img)
                        # If no faces are found in the image, return an empty result.
                        if len(face_box) == 0:
                            return []
                        # Find encodings for faces in the test iamge
                        faces_encodings = face_recognition.face_encodings(img, known_face_locations=face_box)
                        # Use the KNN model to find the best matches for the test face
                        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=2)
                        matches = [closest_distances[0][i][0] <= threshold for i in range(len(face_box))]
                        # Predict classes and remove classifications that aren't within the threshold
                        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings),face_box,matches
                        )]


                              
                    # Flip the image (optional)
                    frame=cv2.flip(img,1) # 0 = horizontal ,1 = vertical , -1 = both
                    frame_copy = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                    frame_copy=cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                    predictions = training(frame_copy, model_path="model.clf") # add path here
                    font = cv2.FONT_HERSHEY_DUPLEX
                    for name, (top, right, bottom, left) in predictions:
                        top *= 4 #scale back the frame since it was scaled to 1/4 in size
                        right *= 4
                        bottom *= 4
                        left *= 4
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
                        cv2.putText(img, name, (left-10,top-6), font, 0.8, (255, 255, 255), 1)

                
                return av.VideoFrame.from_ndarray(img, format = 'bgr24')


        webrtc_streamer(key = 'example',
                        rtc_configuration = RTC_CONFIGURATION,
                        video_processor_factory = VideoProcessor,
                        media_stream_constraints = {
                            'video': True,
                            'audio': False
                            }
                        )

# -------------Hide Streamlit Watermark------------------------------------------------
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)