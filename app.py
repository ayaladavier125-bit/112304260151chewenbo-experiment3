import os
import torch
import torch.nn as nn
import gradio as gr
import numpy as np
from PIL import Image

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

model = CNN()
model.load_state_dict(torch.load('cnn_best_model.pth', map_location='cpu'))
model.eval()

def process_image(image):
    if image is None:
        return None
    
    if isinstance(image, dict):
        if 'composite' in image:
            image = image['composite']
        elif 'layers' in image and len(image['layers']) > 0:
            image = image['layers'][0]
        else:
            return None
    
    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 4:
            img = Image.fromarray(image.astype('uint8'), 'RGBA')
        elif image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            img = Image.fromarray(image.astype('uint8'), 'L')
        else:
            img = Image.fromarray(image.astype('uint8'))
    else:
        return None
    
    img_gray = img.convert('L')
    img_resized = img_gray.resize((28, 28), Image.LANCZOS)
    img_array = np.array(img_resized)
    
    if np.all(img_array == 255):
        return None
    
    img_inverted = 255 - img_array
    img_normalized = img_inverted.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_normalized).view(1, 1, 28, 28)
    
    return img_tensor

def predict_digit(image):
    img_tensor = process_image(image)
    
    if img_tensor is None:
        return {"请在画板上书写数字或上传图片": 1.0}
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
    
    probs = probabilities[0].numpy()
    result = {str(i): float(probs[i]) for i in range(10)}
    
    return result

with gr.Blocks(title="CNN 手写数字识别") as demo:
    gr.Markdown("# 🖋️ CNN 手写数字识别系统")
    gr.Markdown("选择以下任意一种方式输入数字进行识别：")
    
    with gr.Tabs():
        with gr.TabItem("✏️ 手写识别"):
            gr.Markdown("在画板上书写数字，点击识别按钮进行预测")
            canvas = gr.Sketchpad()
            gr.Button("🔍 识别手写").click(predict_digit, inputs=canvas, outputs=gr.Label(num_top_classes=3))
        
        with gr.TabItem("📷 图片上传"):
            gr.Markdown("上传手写数字图片进行识别")
            image_upload = gr.Image(sources=["upload"], type="numpy")
            gr.Button("🔍 识别图片").click(predict_digit, inputs=image_upload, outputs=gr.Label(num_top_classes=3))

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))