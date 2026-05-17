# CNN 手写数字识别系统

基于 PyTorch 和 Gradio 构建的手写数字识别 Web 应用。

## 功能特性

- **手写识别**：在画板上直接书写数字进行识别
- **图片上传**：支持上传手写数字图片进行识别
- **实时预测**：显示 Top-3 预测结果及置信度
- **模型准确率**：99.1%

## 快速开始

### 本地运行

```bash
pip install -r requirements.txt
python app.py
```

打开浏览器访问 `http://localhost:7860`

## 技术栈

- **框架**: PyTorch 2.0+
- **界面**: Gradio
- **模型**: CNN（436,586 参数）

## 项目结构

```
project/
├── app.py              # Web 应用入口
├── model.pth           # 训练好的模型权重
├── requirements.txt    # 依赖列表
└── README.md           # 项目说明
```

## 许可证

MIT License
