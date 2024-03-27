import onnxruntime as rt
import cv2
import numpy as np
import time
import service.main as s


def emotions_detector(img_array):
    time_init = time.time()
    IM_SIZE = 256
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    test_image = cv2.resize(img_array, (IM_SIZE, IM_SIZE))
    im = np.float32(test_image)
    im = np.expand_dims(im, axis=0)  # Add the batch dimension

    onnx_pred = s.m_q.run(['dense_3'], {"input": im})  # Dense name is important, changes with the NN version
    time_elapsed = time.time() - time_init
    emotion = ""
    if np.argmax(onnx_pred, axis=-1)[0][0] == 0:
        emotion = "angry"
    elif np.argmax(onnx_pred, axis=-1)[0][0] == 1:
        emotion = "happy"
    elif np.argmax(onnx_pred, axis=-1)[0][0] == 2:
        emotion = "sad"
    else:
        emotion = "wtf"
    return {
        "emotion": emotion,
        "time_elapsed": str(time_elapsed),

    }



