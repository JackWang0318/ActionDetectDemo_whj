import cv2
import streamlit as st
import numpy as np
from PIL import Image


def web_cam():
    # 创建一个空的用于显示视频的图像容器
    video_container = st.empty()
    cap = cv2.VideoCapture(0)
    # 循环读取摄像头视频帧并显示在Streamlit网页上
    while True:
        # 读取摄像头视频帧
        ret, frame = cap.read()

        # 显示视频帧
        video_container.image(frame, channels="BGR")

        # 按下'q'键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 关闭摄像头和销毁窗口
    cap.release()
    cv2.destroyAllWindows()



def main_loop():

    web_cam()



if __name__ == '__main__':

    st.title("Action Detection Demo App")
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
    tab1, tab2, tab3 = st.tabs(["hello", "thanks", "i love you"])
    with tab1:
        st.header("hello")
        #st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

    with tab2:
        st.header("thanks")
        #st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

    with tab3:
        st.header("i love you")
        #st.image("https://static.streamlit.io/examples/owl.jpg", width=200)


    # 创建一个互动按钮
    #button_clicked = st.button("Open My Web Camera")  # checkbox 也可以   *按钮模板可以改进
    button_clicked = st.checkbox("Open My Web Camera", key="primary_button")

    # 检查按钮是否被点击 
    if button_clicked:
        main_loop()




#大致思路及task：
#cv2中调用库 打开web cam视频，将用户手势视频实时传输到后端
#后端服务器进行predict  再将手势对应的内容传回前端
#前端显示 反馈用户
#设计一下比较简洁美观的web UI  using streamlit