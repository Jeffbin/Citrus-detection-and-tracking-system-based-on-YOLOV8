# 工程文件说明

该工程构建了一个Streamlit应用，实现视频内柑橘的实时检测与跟踪。流程如下：用户上传视频文件后，程序启动双线程并发处理。一号线程读取视频帧送入队列，另一线程从队列取出帧，利用YOLOv8模型检测柑橘并追踪，标记框体及ID于帧上，同时在Web界面实时显示处理视频与柑橘计数。处理完毕，在后端展示处理时间和平均帧率，反映系统性能。整个应用集成视频处理与深度学习技术，直观高效地完成水果监测任务

- ”finally_run_worker_代码.py “为工程文件
- “orange_video.mp4”为本次测试所需视频

- “系统演示视频.mp4”为本系统测试视频
- ”model“文件夹包含YOLOv8训练出来的柑橘模型best.pt及训练结果
- “orange_data“为柑橘数据集

## 1.环境配置

```shell
pip install -r requirements.txt
```



## 2.运行系统

```shell
streamlit run .\test_script\finally_run_worker.py
```



## 3.选择测试视频

![image-20240612210352794](orange_video.mp4)







## 4.点击处理，查看视频处理结果及柑橘数量

![image-20240612210513557](C:\Users\HQR\AppData\Roaming\Typora\typora-user-images\image-20240612210513557.png) 



