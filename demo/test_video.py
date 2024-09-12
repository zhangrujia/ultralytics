from ultralytics import YOLO
import cv2

# 加载预训练的YOLOv8模型（例如yolov8n，yolov8s等）
model = YOLO('yolov8s.pt')

# 读取视频
video_path = './demo/jijian4.mp4'  # 替换为你的视频路径
cap = cv2.VideoCapture(video_path)

# 获取视频信息
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 输出视频文件
out = cv2.VideoWriter('./output/jijian4.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用YOLOv8模型进行预测
    results = model(frame)

    # 可视化预测结果
    annotated_frame = results[0].plot()  # 绘制预测结果

    # 显示/保存每帧
    out.write(annotated_frame)

    cv2.imshow('Fencing Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
