#frontend lib
import cv2
import streamlit as st
import numpy as np
from PIL import Image
import requests

from streamlit_webrtc import webrtc_streamer
import mediapipe as mp
import av
#main lib

#import demo as dm

# Path for exported data, numpy arrays
# DATA_PATH = os.path.join('MP_Data') 
# log_dir = os.path.join('weight')
# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 30

colors = [(245,117,16), (117,245,16), (16,117,245)]

sequence = []
sentence = []
predictions = []
threshold = 0.5

def web_cam():
    # 创建一个空的用于显示视频的图像容器
    video_container = st.empty()
    cap = cv2.VideoCapture(0)

    # 设置视频编码器和视频参数
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # video_writer = cv2.VideoWriter("output.mp4", fourcc, 20.0, (640, 480))



    # 循环读取摄像头视频帧并显示在Streamlit网页上
    while True:
        # 读取摄像头视频帧
        ret, frame = cap.read()

        # 显示视频帧
        video_container.image(frame, channels="BGR")

        # 发送视频帧到后端 
        # response = requests.post(url, files={"frame": frame})
        # data_received = response.json() 
        # st.write(f'data_received: {data_received[0]}')


        # 按下'q'键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 关闭摄像头和销毁窗口
    cap.release()
    cv2.destroyAllWindows()


#------------------Keypoints using MP Holistic------------------
# mp_holistic = mp.solutions.holistic # Holistic model
# mp_drawing = mp.solutions.drawing_utils # Drawing utilities


mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    # print(result.multi_hand_landmarks)
    
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(img_rgb, handLms, mpHands.HAND_CONNECTIONS)
            
    return av.VideoFrame.from_ndarray(img_rgb, format="rgb24")




def main():
    # 设置应用程序的主题
    # st.set_page_config(page_title="ML_Web_App", page_icon=":smiley:", 
    #                    layout="wide", initial_sidebar_state="expanded", theme="dark")
    
    st.title("Action Detection Web App")
    st.subheader("This app allows you to play with hand gesture using machine learning!")
    st.text("ML01 Machine Learning, UTSEUS, Shanghai University")

    #sidebar experiment
    # Using object notation
    add_selectbox = st.sidebar.selectbox(
        "Would you like to spend time learning ML?",
        ("Absolutely yes", "Possibly", "Never")
    )
    # Using "with" notation
    with st.sidebar:
        add_radio = st.radio(
            "How do you like our project?",
            ("Not much", "Just so-so", "Great", "Perfect")
        )
        
    #Using tabs
    tab1, tab2, tab3, tab4 = st.tabs(["hello", "thanks", "i love you", "i hate you"])
    with tab1:
        st.header("hello")
        st.image("example/hello.png", width=500)

    with tab2:
        st.header("thanks")
        st.image("example/thanks.png", width=500)

    with tab3:
        st.header("i love you")
        st.image("example/iloveyou.png", width=500)

    with tab4:
        st.header("i hate you")
        st.image("example/ihateyou.png", width=500) 



    webrtc_streamer(key="gesture recognition", video_frame_callback=video_frame_callback)  
    button_clicked = st.checkbox("Open My Web Camera", key="primary_button")
    # 检查按钮是否被点击 
    if button_clicked:
         web_cam()

if __name__ == '__main__':
    main()



#大致思路及task：
#cv2中调用库 打开web cam视频，  已完成。
#前端显示 反馈用户
#设计一下比较简洁美观的web UI  using streamlit