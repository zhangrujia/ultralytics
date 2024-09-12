from ultralytics import YOLO
 
# 读取模型，这里传入训练好的模型
model = YOLO('yolov8s.pt')
 
# 模型预测，save=True 的时候表示直接保存yolov8的预测结果
metrics = model.predict(['./demo/jijian.jpg'], save=True)
# 如果想自定义的处理预测结果可以这么操作，遍历每个预测结果分别的去处理
for m in metrics:
    # 获取每个boxes的结果
    box = m.boxes
    # 获取box的位置，
    xywh = box.xywh
    # 获取预测的类别
    cls = box.cls
    
    print(box, xywh, cls)