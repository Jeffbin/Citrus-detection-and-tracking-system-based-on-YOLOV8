import tempfile
import time
from streamlit.runtime.scriptrunner import add_script_run_ctx
import threading
from queue import Queue
import cv2
from ultralytics import YOLO
from collections import defaultdict
import streamlit as st

model = YOLO(r'D:\yolo\yolov5-master\runs\detect\train\weights\best.pt')
list_fps = []
q = Queue()
track_history = defaultdict(lambda: [])
success = False
T_count=0

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


def process_video_frame(count_placeholder, img_placeholder):
    time.sleep(0.1)
    print("process_video_frame线程启动")
    while not q.empty() or success:
        frame = q.get()
        results = model.track(frame, conf=0.3, persist=True)
        if results[0].boxes.id is None:
            continue

        track_ids = results[0].boxes.id.int().cpu().tolist()


        for track_id, box in zip(track_ids, results[0].boxes.data):
            if box[-1] == 0:
                box_label(frame, box, '#' + str(track_id) + ' orange', (0, 255, 255))

                x1, y1, x2, y2 = box[:4]
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2

                track = track_history[track_id]
                track.append((float(x), float(y)))

        count = max(track_history)


        count_placeholder.metric("柑橘数目", count)
        img_placeholder.image(frame, channels="BGR", use_column_width=True)


def get_frame(cap):
    print("get_frame线程启动")
    global success, T_count
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            T_count+=1
            q.put(frame)
        else:
            break


def main():
    st.title('基于深度学习的柑橘检测与跟踪系统')
    video_file = st.file_uploader(' ', type=['mp4', 'avi'])
    if video_file is not None:
        st.video(video_file)
        st.markdown(r"""
        ##       原始视频


        """)

    if st.button('点击处理'):
        start_time = time.time()
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)

        count_placeholder = st.empty()
        img_placeholder = st.empty()
        with st.spinner("处理中，请等待"):
            show_threading = threading.Thread(target=get_frame, args=(cap,))
            process_threading1 = threading.Thread(target=process_video_frame,
                                                  args=(count_placeholder, img_placeholder,))

            add_script_run_ctx(show_threading)
            add_script_run_ctx(process_threading1)

            show_threading.start()
            process_threading1.start()

            show_threading.join()
            process_threading1.join()

            st.markdown(r"""
                            ##       处理成功
    
    
                            """)
            print("处理时间:{}".format(time.time() - start_time))
            print(T_count)
            print("平均帧率:{}".format(T_count/(time.time() - start_time)))


        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
