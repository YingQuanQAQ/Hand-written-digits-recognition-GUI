import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf  # 用于加载训练好的模型
import cv2

# 创建画板类
class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别")

        # 画布
        self.canvas = tk.Canvas(root, width=280, height=280, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=2)
        self.canvas.bind("<B1-Motion>", self.paint)

        # 画布内容存储
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)

        # 按钮
        self.clear_button = tk.Button(root, text="清除", command=self.clear_canvas)
        self.clear_button.grid(row=1, column=0)
        self.predict_button = tk.Button(root, text="识别", command=self.predict_digit)
        self.predict_button.grid(row=1, column=1)

        # 识别结果标签
        self.result_label = tk.Label(root, text="识别结果: ")
        self.result_label.grid(row=2, column=0, columnspan=2)

        # 加载训练好的神经网络模型
        self.model = tf.keras.models.load_model("venv/mnist_cnn.h5")  # 需要先训练并保存

    def paint(self, event):
        """ 绘制用户手写输入 """
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)
        self.draw.ellipse([x1, y1, x2, y2], fill="black")

    def clear_canvas(self):
        """ 清除画布 """
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)

    def predict_digit(self):
        """ 处理画布内容并进行预测 """
        img = self.image.resize((28, 28)).convert("L")  # 转为灰度
        img_array = np.array(img)

        # 颜色反转（你的画板可能是黑底白字）
        img_array = 255 - img_array

        # 二值化处理，让数字更清晰
        _, img_array = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY)

        # 归一化 & 形状调整
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # 预测
        prediction = self.model.predict(img_array)
        digit = np.argmax(prediction)
        self.result_label.config(text=f"识别结果: {digit}")


# 启动应用
root = tk.Tk()
app = DrawingApp(root)
root.mainloop()
