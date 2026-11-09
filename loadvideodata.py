import cv2

# 视频流地址（目前无本地视频流数据注入，保留读取能力用于后续接入）
path = "https://www.bilibili.com/video/BV18M411k7yb/?share_source=copy_web&vd_source=499591ddb720e884ca4a28d83dd4a674"
cap = cv2.VideoCapture(path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
