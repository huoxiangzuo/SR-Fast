import os
import time
import torch
import cv2
import threading
from queue import Queue
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PIL import Image
import numpy as np
from basicsr.archs.femasr_arch import FeMaSRNet
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url


class SRProcessor:
    def __init__(self, scale=2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.amp_enabled = True if torch.cuda.is_available() else False
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)

        self.scale = scale
        self.model = FeMaSRNet(
            codebook_params=[[32, 1024, 512]],
            LQ_stage=True,
            scale_factor=self.scale
        ).to(self.device)

        weight_path = "models/FeMaSR_SRX2_model_g.pth"
        if not os.path.exists(weight_path):
            print("Downloading pretrained weights...")
            os.makedirs("models", exist_ok=True)
            url = 'https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_SRX2_model_g.pth'
            weight_path = load_file_from_url(url, model_dir="models")

        state_dict = torch.load(weight_path, map_location=self.device)['params']
        if self.amp_enabled:
            state_dict = {k: v.half() if v.is_floating_point() else v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        self._warmup_model()
        print(f"FeMaSR model ready. AMP enabled: {self.amp_enabled}")

    def _warmup_model(self):
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.amp_enabled):
            dummy = torch.rand(1, 3, 128, 128).to(self.device)
            if self.amp_enabled:
                dummy = dummy.half()
            _ = self.model.test(dummy)

    def preprocess_images(self, img_paths):
        """读取图像并 resize 成统一尺寸，返回 batch tensor"""
        tensors = []
        filenames = []

        sizes = []
        pil_imgs = []

        # 预读取所有图像并记录尺寸
        for path in img_paths:
            try:
                img = Image.open(path).convert('RGB')
                pil_imgs.append(img)
                filenames.append(os.path.basename(path))
                sizes.append(img.size)  # (W, H)
            except Exception as e:
                print(f"Error loading {path}: {e}")

        if not pil_imgs:
            return torch.tensor([]), []

        # 统一 resize 成最大尺寸
        max_w = max([w for w, h in sizes])
        max_h = max([h for w, h in sizes])
        target_size = (max_w, max_h)

        for img in pil_imgs:
            resized = img.resize(target_size, Image.BICUBIC)
            img_bgr = cv2.cvtColor(np.array(resized), cv2.COLOR_RGB2BGR)
            img_tensor = img2tensor(img_bgr, bgr2rgb=True, float32=True) / 255.
            if self.amp_enabled:
                img_tensor = img_tensor.half()
            tensors.append(img_tensor)

        return torch.stack(tensors).to(self.device), filenames

    def batch_process_images(self, img_paths):
        try:
            imgs_tensor, filenames = self.preprocess_images(img_paths)
            if imgs_tensor.ndim != 4 or imgs_tensor.size(0) == 0:
                return []

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.amp_enabled):
                output = self.model.test(imgs_tensor)

            output = output.float()
            results = []
            for i in range(output.shape[0]):
                out_img = tensor2img(output[i].unsqueeze(0), rgb2bgr=False)
                pil_img = Image.fromarray(out_img.astype('uint8'))
                results.append((filenames[i], pil_img))
            return results
        except Exception as e:
            print(f"Batch processing error: {e}")
            return []


class ImageHandler(FileSystemEventHandler):
    def __init__(self, queue):
        self.queue = queue

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            while not self._file_ready(event.src_path):
                time.sleep(0.1)
            print(f"Queued: {os.path.basename(event.src_path)}")
            self.queue.put(event.src_path)

    def _file_ready(self, filepath):
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


def batch_processor(processor, queue, batch_size=4, interval=1.0):
    while True:
        time.sleep(interval)
        batch = []
        while not queue.empty() and len(batch) < batch_size:
            filepath = queue.get()
            if os.path.exists(filepath):
                batch.append(filepath)

        if batch:
            print(f"\n[Batch] Processing {len(batch)} images...")
            start_time = time.time()
            results = processor.batch_process_images(batch)

            for filename, result_img in results:
                result_img.save(os.path.join("output", f"sr_{filename}"))
                os.rename(os.path.join("input", filename), os.path.join("processed", filename))

            print(f"[Batch] Done in {time.time() - start_time:.2f}s.")


if __name__ == "__main__":
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    processor = SRProcessor()
    image_queue = Queue()

    event_handler = ImageHandler(image_queue)
    observer = Observer()
    observer.schedule(event_handler, path="input", recursive=False)
    observer.start()

    threading.Thread(target=batch_processor, args=(processor, image_queue), daemon=True).start()

    print("=" * 50)
    print(f"Monitoring directory: {os.path.abspath('input')}")
    print("Drop images into the input folder to process")
    print("=" * 50)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
