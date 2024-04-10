import tempfile
import threading
import time
from collections import defaultdict
from queue import Queue
import cv2
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
from ultralytics import YOLO

model = YOLO(r'D:\yolo\yolov5-master\runs\detect\train\weights\best.pt')

track_history = defaultdict(lambda: [])
count = 0
q = Queue(maxsize=0)
pq=Queue(maxsize=0)

success = False

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

def process_frame():
    print("处理线程启动")
    global count
    global success
    frame=q.get()
    while success:
        results = model.track(frame, conf=0.3, persist=True)
        track_ids = results[0].boxes.id.int().cpu().tolist()
        for track_id, box in zip(track_ids, results[0].boxes.data):
            if box[-1] == 0:  # 目标为橙子
                box_label(frame, box, '#' + str(track_id) + ' orange', (0, 255, 255))
                x1, y1, x2, y2 = box[:4]
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
                track = track_history[track_id]
                track.append((float(x), float(y)))
        count = max(track_ids)
        cv2.putText(frame, 'Orange_counting:  ' + str(max(track_ids)), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        pq.put(frame)


def show_cap(cap):
    print("展示线程启动")

    global success
    img_placeholder = st.empty()
    count_placeholder = st.empty()
    while cap.isOpened():
        start_T = time.time()
        success, frame1 = cap.read()
        q.put(frame1)
        if success or (not pq.empty()):
            #start_time = time.time()
            frame_processed = pq.get()
            img_placeholder.image(frame_processed, channels="BGR", use_column_width=True)
            count_placeholder.metric("柑橘数目", count)

            fps = 1 / (time.time() - start_T)
            print('fps:{}'.format(fps))
            print('一幅图处理时间:{}'.format(time.time() - start_T))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
def main():

    st.title('基于深度学习的柑橘计数系统')

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
            t3 = time.time()
            show_threading = threading.Thread(target=show_cap, args=(cap,))
            process_threading = threading.Thread(target=process_frame)
            add_script_run_ctx(show_threading)
            add_script_run_ctx(process_threading)
            show_threading.start()
            process_threading.start()
            show_threading.join()
            process_threading.join()
            t4 = time.time()
            all_time = t4 - t3
            print('总处理时间{}'.format(all_time))
            cap.release()

            cv2.destroyAllWindows()

        st.markdown(r"""
                ##       处理成功


                """)

    # 释放视频捕捉对象，并关闭显示窗口


if __name__ == '__main__':
    main()
