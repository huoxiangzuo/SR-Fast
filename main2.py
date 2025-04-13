import os
import time
import torch
import cv2
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from torchvision.transforms import ToTensor
from PIL import Image


# 替换 SimpleSRModel 相关内容
from basicsr.archs.femasr_arch import FeMaSRNet
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url

class SRProcessor:
    def __init__(self, scale=2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.scale = scale
        self.model = FeMaSRNet(codebook_params=[[32, 1024, 512]], LQ_stage=True, scale_factor=self.scale).to(self.device)

        # 下载或加载模型权重
        weight_path = "models/FeMaSR_SRX2_model_g.pth"
        if not os.path.exists(weight_path):
            print("Downloading pretrained weights...")
            url = 'https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_SRX2_model_g.pth'
            weight_path = load_file_from_url(url, model_dir="models")

        self.model.load_state_dict(torch.load(weight_path)['params'], strict=False)
        self.model.eval()

        # 预热
        with torch.no_grad():
            dummy = torch.rand(1, 3, 64, 64).to(self.device)
            _ = self.model.test(dummy)
        print("FeMaSR model ready.")

    def process_image(self, img_path):
        try:
            import cv2
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img_tensor = img2tensor(img).to(self.device) / 255.
            img_tensor = img_tensor.unsqueeze(0)

            h, w = img_tensor.shape[2:]
            max_size = 600 * 600

            with torch.no_grad():
                if h * w < max_size:
                    output = self.model.test(img_tensor)
                else:
                    output = self.model.test_tile(img_tensor)

            output_img = tensor2img(output)

            from PIL import Image
            return Image.fromarray(output_img.astype('uint8'))

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