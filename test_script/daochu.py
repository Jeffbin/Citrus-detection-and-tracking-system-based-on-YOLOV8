import tempfile
import time

import cv2
from ultralytics import YOLO
from collections import defaultdict
import streamlit as st

model = YOLO(r'D:\yolo\yolov5-master\runs\detect\train\weights\best.pt')
list_fps = []
def average(list):
    sum = 0
    for i in list:
        sum += i
    return sum / len(list)
def box_label(image, box, label='', color=(255, 0, 128), txt_color=(50, 0, 255)):
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


def process_video_frame(frame, results, track_history):

    track_ids = results[0].boxes.id.int().cpu().tolist()

    for track_id, box in zip(track_ids, results[0].boxes.data):
        if box[-1] == 0:
            box_label(frame, box, '#' + str(track_id) + ' orange', (0, 255, 255))

            x1, y1, x2, y2 = box[:4]
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2

            track = track_history[track_id]
            track.append((float(x), float(y)))


def visualize_processed_frame(frame, img_placeholder):
    #frame = cv2.resize(frame, (640, 720))
    img_placeholder.image(frame, channels="BGR", use_column_width=True)
    cv2.imshow("YOLOv8 Tracking", frame)

def main():
    st.title('基于深度学习的柑橘检测与跟踪系统')
    video_file = st.file_uploader(' ', type=['mp4', 'avi'])
    count_test=0
    if video_file is not None:
        st.video(video_file)
        st.markdown(r"""
        ##       原始视频


        """)

    if st.button('点击处理'):
        T = time.time()
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)

        track_history = defaultdict(lambda: [])
        count_placeholder = st.empty()
        img_placeholder = st.empty()

        while cap.isOpened():
            success, frame = cap.read()
            start_time = time.time()
            count_test+=1

            if success:
                results = model.track(frame, conf=0.3, persist=True)
                if results[0].boxes.id is None:
                    continue
                process_video_frame(frame, results, track_history)


                count=max(track_history)

                count_placeholder.metric("柑橘数目", count)
                visualize_processed_frame(frame, img_placeholder)
                fps=1/(time.time() - start_time)
                list_fps.append(fps)
                print("fps：{}".format(fps))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            else:
                break

        cap.release()
        print("总时长：{}".format(time.time() - T))
        print("平均fps：{}".format(average(list_fps)))
        st.markdown(r"""
                        ##       处理成功


                        """)

        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
