# 导入工具包
import numpy as np
import argparse
import imutils
import cv2

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# 正确答案
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}


def order_points(pts):
    """对4个坐标点进行排序：左上、右上、右下、左下"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下

    return rect


def four_point_transform(image, pts):
    """执行透视变换"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算新图像的宽度和高度
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 定义目标点
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 计算变换矩阵并执行变换
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def sort_contours(cnts, method="left-to-right"):
    """对轮廓进行排序"""
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes


def cv_show(name, img):
    """显示图像并等待按键关闭"""
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 1. 图像预处理
image = cv2.imread(args["image"])
if image is None:
    print(f"[ERROR] 无法加载图像: {args['image']}")
    exit(1)

contours_img = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv_show('blurred', blurred)

# 边缘检测
edged = cv2.Canny(blurred, 75, 200)
cv_show('edged', edged)
print(f"边缘像素数量: {np.count_nonzero(edged)}")

# 2. 查找答题卡轮廓
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print(f"检测到轮廓数量: {len(cnts)}")

if len(cnts) == 0:
    print("[ERROR] 未检测到任何轮廓")
    exit(1)

# 按面积排序并找到答题卡轮廓
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
docCnt = None

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        docCnt = approx
        break

if docCnt is None:
    print("[ERROR] 未找到四边形轮廓")
    exit(1)

# 显示轮廓
cv2.drawContours(contours_img, [docCnt], -1, (0, 255, 0), 3)
cv_show('contours_img', contours_img)

# 3. 透视变换
warped = four_point_transform(gray, docCnt.reshape(4, 2))
cv_show('warped', warped)

# 4. 阈值处理
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
print(f"阈值图像统计 - 最小值: {thresh.min()}, 最大值: {thresh.max()}, 非零像素: {np.count_nonzero(thresh)}")
cv_show('thresh', thresh)

# 5. 查找选项轮廓
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print(f"阈值轮廓数量: {len(cnts)}")

if len(cnts) == 0:
    print("[ERROR] 未检测到任何选项轮廓")
    exit(1)

# 筛选出可能是选项的轮廓
questionCnts = []

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    # 根据实际调整这些参数
    if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:
        questionCnts.append(c)

if len(questionCnts) == 0:
    print("[ERROR] 未找到符合条件的选项轮廓")
    exit(1)

# 绘制所有检测到的选项轮廓
thresh_Contours = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
for c in questionCnts:
    cv2.drawContours(thresh_Contours, [c], -1, (0, 0, 255), 2)
cv_show('thresh_Contours', thresh_Contours)

# 6. 处理答案
questionCnts = sort_contours(questionCnts, method="top-to-bottom")[0]
correct = 0

# 每排有5个选项
for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
    cnts = sort_contours(questionCnts[i:i + 5])[0]
    bubbled = None

    for (j, c) in enumerate(cnts):
        # 创建掩膜
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        # 计算非零像素
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)

        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)

    # 检查答案
    color = (0, 0, 255)
    k = ANSWER_KEY.get(q, -1)

    if k == bubbled[1]:
        color = (0, 255, 0)
        correct += 1

    # 绘制结果
    cv2.drawContours(warped, [cnts[k]], -1, color, 3)

# 7. 显示结果
score = (correct / 5.0) * 100
print(f"[INFO] 得分: {score:.2f}%")

cv2.putText(warped, f"{score:.2f}%", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# 显示原始图像和处理后的图像
cv2.imshow("Original", image)
cv2.imshow("Exam", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()