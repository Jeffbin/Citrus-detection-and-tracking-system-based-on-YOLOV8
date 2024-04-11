import time

import cv2
from ultralytics import YOLO
from collections import defaultdict
import streamlit as st
import tempfile

model = YOLO(r'D:\yolo\yolov5-master\runs\detect\train\weights\best.pt')

# track_history用于保存目标ID，以及它在各帧的目标位置坐标，这些坐标是按先后顺序存储的
track_history = defaultdict(lambda: [])
count = 0


def box_label(image, box, label='', color=(0, 0, 128), txt_color=(0, 0, 255)):
    # 得到目标矩形框的左上角和右下角坐标
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    # 绘制矩形框
    cv2.rectangle(image, p1, p2, color, thickness=1, lineType=cv2.LINE_AA)
    if label:
        # 得到要书写的文本的宽和长，用于给文本绘制背景色
        w, h = cv2.getTextSize(label, 0, fontScale=2 / 3, thickness=1)[0]
        # 确保显示的文本不会超出图片范围
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # 填充颜色
        # 书写文本
        cv2.putText(image,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    2 / 3,
                    txt_color,
                    thickness=1,
                    lineType=cv2.LINE_AA)


def get_cap(cap):
    # 视频帧循环
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #
    # fourcc = cv2.VideoWriter.fourcc(*'XVID')
    # videoWriter = cv2.VideoWriter("F:/store_video/counting.avi", fourcc, fps, size)
    # 在Streamlit页面上创建图像的占位符
    img_placeholder = st.empty()

    # 在Streamlit页面上创建计数器的占位符
    count_placeholder = st.empty()
    while cap.isOpened():
        # 读取一帧图像
        success, frame = cap.read()

        if success:
            # 在帧上运行YOLOv8跟踪，persist为True表示保留跟踪信息，conf为0.3表示只检测置信值大于0.3的目标
            t1=time.time()
            results = model.track(frame, conf=0.3, persist=True)
            # 得到该帧的各个目标的ID
            track_ids = results[0].boxes.id.int().cpu().tolist()
            # 遍历该帧的所有目标
            for track_id, box in zip(track_ids, results[0].boxes.data):

                if box[-1] == 0:  # 目标为橙子
                    # 绘制该目标的矩形框
                    box_label(frame, box, '#' + str(track_id) + ' orange', (0, 255, 255))
                    # 得到该目标矩形框的中心点坐标(x, y)
                    x1, y1, x2, y2 = box[:4]
                    x = (x1 + x2) / 2
                    y = (y1 + y2) / 2
                    # 提取出该ID的以前所有帧的目标坐标，当该ID是第一次出现时，则创建该ID的字典
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # 追加当前目标ID的坐标
            global count
            count = max(track_ids)
            cv2.putText(frame, 'Orange_counting:  ' + str(max(track_ids)), (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # cv2.imshow("YOLOv8 Tracking", frame)  # 显示标记好的当前帧图像
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 更新Streamlit页面上的图像
            img_placeholder.image(frame, channels="BGR", use_column_width=True)

            # 更新Streamlit页面上的计数器
            count_placeholder.metric("柑橘数目", count)
            t2=time.time()
            fps=1/(t2-t1)
            print('fps:{}'.format(fps))
            #videoWriter.write(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):  # 'q'按下时，终止运行

                #videoWriter.release()
                break

        else:  # 视频播放结束时退出循环

            break


def main():
    # markdown

    # 设置网页标题
    st.title('基于深度学习的柑橘计数系统')

    # 展示一级标题
    st.header('上传视频')

    video_file = st.file_uploader(' ', type=['mp4', 'avi'])

    if video_file is not None:
        st.video(video_file)
        st.markdown(r"""
        ##       原始视频


        """)

    if st.button('点击处理'):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        with st.spinner("处理中，请等待"):
            t3=time.time()
            get_cap(cap)
            t4 = time.time()
            all_time = t4 - t3
            print('处理时间{}'.format(all_time))
        st.video('F:/store_video/counting.mp4')

        col1, col2, col3 = st.columns(3)

        col1.metric("柑橘数目", count)

        st.markdown(r"""
                ##       处理成功


                """)

        cap.release()

        cv2.destroyAllWindows()
    # st.video(r"D:\yolo\yolov5-master\data\MOT16\test.mp4")

    # 释放视频捕捉对象，并关闭显示窗口


if __name__ == '__main__':
    main()
