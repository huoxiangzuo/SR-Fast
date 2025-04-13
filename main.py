import os
import time
import torch
import cv2
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from torchvision.transforms import ToTensor
from PIL import Image


# 模拟一个简单的超分辨率模型
class SimpleSRModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x):
        return torch.nn.functional.interpolate(self.conv(x), scale_factor=2, mode='bicubic')


class SRProcessor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 初始化模型
        self.model = SimpleSRModel().to(self.device)

        # 加载权重 (这里模拟加载过程)
        if os.path.exists("models/weights.pth"):
            self.model.load_state_dict(torch.load("models/weights.pth"))

        self.model.eval()
        self.transform = ToTensor()

        # 预热
        with torch.no_grad():
            dummy = torch.rand(1, 3, 256, 256).to(self.device)
            _ = self.model(dummy)
        print("Model warmed up")

    def process_image(self, img_path):
        try:
            # 读取图片
            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            # 预处理
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            # 推理
            with torch.no_grad():
                output = self.model(img_tensor)

            # 后处理
            output = output.squeeze().cpu().clamp(0, 1)
            output_img = Image.fromarray((output.permute(1, 2, 0).numpy() * 255).astype('uint8'))

            return output_img.resize((w * 2, h * 2))  # 模拟2倍超分

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return None


class ImageHandler(FileSystemEventHandler):
    def __init__(self, processor):
        self.processor = processor

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"\nProcessing: {os.path.basename(event.src_path)}")
            start_time = time.time()

            # 等待文件完全写入
            while not self._file_ready(event.src_path):
                time.sleep(0.1)

            result = self.processor.process_image(event.src_path)

            if result:
                # 保存结果
                filename = os.path.basename(event.src_path)
                output_path = os.path.join("output", f"sr_{filename}")
                result.save(output_path)

                # 移动原始文件
                processed_path = os.path.join("processed", filename)
                os.rename(event.src_path, processed_path)

                print(f"Done in {time.time() - start_time:.2f}s | Original: {filename}")

    def _file_ready(self, filepath):
        """检查文件是否完全写入"""
        try:
            with open(filepath, 'rb') as f:
                f.seek(0, 2)
                size1 = f.tell()
                time.sleep(0.1)
                f.seek(0, 2)
                size2 = f.tell()
                return size1 == size2 and size1 > 0
        except IOError:
            return False


if __name__ == "__main__":
    # 创建目录
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 初始化处理器
    processor = SRProcessor()

    # 设置监控
    event_handler = ImageHandler(processor)
    observer = Observer()
    observer.schedule(event_handler, path="input", recursive=False)

    print("=" * 50)
    print(f"Monitoring directory: {os.path.abspath('input')}")
    print("Drop images into the input folder to process")
    print("=" * 50)

    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()