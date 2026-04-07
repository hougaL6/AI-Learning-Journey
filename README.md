🚀 Day 1: 从零通关 AI 核心理论与 PyTorch 实战
📝 学习目标：建立 AI 项目大框架，掌握 Python 数据处理基础，打通 PyTorch 最小训练闭环。
💡 核心心法：先学概念，不追公式；拒绝死记硬背，用大白话理解一切！

🧠 一、 AI 内功心法（核心基础概念）
1. 三大概念的“套娃”关系
人工智能 (AI) > 机器学习 (ML) > 深度学习 (DL)

ML (机器学习)：不写死规则，给机器喂数据让它自己找规律。

DL (深度学习)：ML 中最前沿的分支，模仿人脑神经网络，擅长处理图像、语音等极其复杂的数据。

2. 机器怎么上课？
监督学习：带“参考答案”的学习（如：看照片认猫狗）。

无监督学习：没参考答案，机器自己找隐藏规律（如：把一筐无名水果按颜色聚类）。

3. 监督学习的两大任务
分类 (Classification)：做选择题，预测离散类别（如：垃圾邮件 / 正常邮件）。

回归 (Regression)：做填空题，预测连续数值（如：预测明天的气温是 25.3 度）。

4. 考试卷子怎么分？
训练集 (Train)：平时作业。机器每天看这些数据更新大脑。

验证集 (Validation)：期中模考。用来评估学习状态，调整学习方法（调参）。

测试集 (Test)：最终高考。数据极其保密，只在最后考一次，评估最终能力。

5. 学生的两类坏毛病
欠拟合 (Underfitting)：学渣。平时作业和考试都不及格，规律没学明白。

过拟合 (Overfitting)：死记硬背的书呆子。平时作业 100 分，一到大考遇到没见过的新题就抓瞎（泛化能力极差）。

6. 怎么给 AI 警察打分？（四大评估指标）
准确率 (Accuracy)：整体猜对的比例。（⚠️ 坑：数据极端不平衡时容易骗人，比如镇上全猜好人）。

精确率 (Precision)：“宁缺毋滥”。你抓回来的人里，到底有几个是真小偷？（侧重：不冤枉好人）。

召回率 (Recall)：“宁可错杀不放过”。镇上真正的那些小偷，你找出来了几个？（侧重：不漏网）。

F1 Score：精确率和召回率的综合平衡得分。

🛠️ 二、 Python 实操与“数据三剑客”
1. Python 核心语法铁律
缩进即灵魂：遇到冒号 :（如 for, if, def），下一行必须缩进 4 个空格，代表从属关系。

字典 (Dict)：person = {"姓名": "张三", "年龄": 20}（⚠️ 键值对之间必须加英文逗号）。

2. NumPy（极速计算大脑）
核心特性：强迫症，必须装同类型数据；支持矩阵极速批量运算。

布尔筛选（极其常用）：

code
Python
import numpy as np
scores = np.array([78, 92, 85, 60])
good_scores = scores[scores >= 85] # 瞬间挑出及格的分数
3. Pandas（超级 Excel 表格）
核心特性：专门处理带有行列名称的结构化数据（CSV）。

常用连招：

code
Python
import pandas as pd
df = pd.read_csv("data.csv") # 读取
df.to_csv("new.csv", index=False) # 保存
avg = df["分数"].mean() # 算平均值
best = df[df["分数"] == df["分数"].max()] # 找最高分
df_sorted = df.sort_values(by="分数", ascending=False) # 降序排队
4. Matplotlib（神笔马良）
避坑指南：必须先 plt.savefig()（拍照保存），再 plt.show()（交卷展示），否则保存的是白纸！

5. OpenCV（视觉预处理）
电脑眼里的彩色图片是 高 × 宽 × 3（BGR通道） 的三维数字矩阵。

预处理流水线：

code
Python
import cv2
img = cv2.imread("test.jpg") # 读图
resized = cv2.resize(img, (256, 256)) # 缩放：迎合 AI 的固定输入强迫症
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转灰度：给 AI 减负
edges = cv2.Canny(gray, 100, 200) # 边缘检测：提取灵魂线稿
🚀 三、 PyTorch 深度学习练功房
1. 核心概念翻译
Tensor (张量)：带了 GPU 显卡加速外挂、且能自动追踪数学运算轨迹的超级多维数组。

Shape (维度套娃)：[[1.0], [2.0]]。外层括号代表“有几个样本(Batch)”，内层代表“每个样本有几个特征”。

self (本厂专属)：Python 面向对象编程的核心。代表“当前被实例化的具体对象”。在 __init__ 里加 self. 等于把机器焊死在自己的工厂里。

2. 神经网络搭建模板 (OOP 八股文)
code
Python
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        # 采购机器（定义网络层）
        self.fc1 = nn.Linear(28*28, 128) # 线性层（质检员）
        self.relu = nn.ReLU()            # 激活函数（转弯滤镜，打破线性规律）
        self.fc2 = nn.Linear(128, 10)    # 最终输出 10 个选项
        
    def forward(self, x):
        # 开启流水线
        x = x.view(-1, 28*28) # 把图片压扁成长条 (-1代表让电脑自动计算批次数量)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
3. 【终极奥义】标准训练闭环五连鞭 + 测试结界
code
Python
import torch
import torch.optim as optim

# 【准备教练与红笔】
optimizer = optim.SGD(model.parameters(), lr=0.01) # 纠错教练（lr 是迈步的力度）
criterion = nn.CrossEntropyLoss() # 批改红笔（算误差）

# ================= 🏋️ 平时魔鬼训练 (Train) =================
model.train() # 开启训练模式
for images, labels in train_loader:
    outputs = model(images)           # 1. 蒙答案 (前向传播)
    loss = criterion(outputs, labels) # 2. 算离谱度 (计算 Loss)
    
    optimizer.zero_grad()             # 3. 擦去旧草稿 (清空旧梯度)
    loss.backward()                   # 4. 寻找病根 (反向传播，算梯度)
    optimizer.step()                  # 5. 动手改错 (更新模型参数)

# ================= 📝 周末期末考试 (Eval) =================
model.eval() # 开启考试模式 (冻结学习机制)
with torch.no_grad(): # 没收草稿纸结界：绝对不允许算梯度，极大节省算力和内存！
    for images, labels in test_loader:
        outputs = model(images)
        # 获取 10 个选项中最高分所在的“序号”(即预测的数字)
        _, predicted = torch.max(outputs.data, 1) 
        
        # 将张量脱壳成纯数字并累加
        correct += (predicted == labels).sum().item()
🎯 总结与碎碎念
代码不用背！代码不用背！代码不用背！ 重要的是脑子里要有数据流动的“加工厂”画面。

_ 在 Python 里是垃圾桶变量，用来接住不用看的数据。

遇到报错不慌张，第一时间 print(x.shape) 查一查维度对不对！
