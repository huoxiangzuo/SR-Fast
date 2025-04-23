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
            for size in [(64, 64), (128, 128), (256, 256)]:
                dummy = torch.rand(1, 3, *size).to(self.device)
                if self.amp_enabled:
                    dummy = dummy.half()
                _ = self.model.test(dummy)
                _ = self.model.test_tile(dummy)

    def process_image(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img_tensor = img2tensor(img, bgr2rgb=True, float32=True).to(self.device) / 255.
            if self.amp_enabled:
                img_tensor = img_tensor.half()
            img_tensor = img_tensor.unsqueeze(0)

            h, w = img_tensor.shape[2:]
            max_size = 1200 * 1200

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.amp_enabled):
                if h * w < max_size:
                    output = self.model.test(img_tensor)
                else:
                    output = self.model.test_tile(img_tensor)

            output = output.float()
            output_img = tensor2img(output, rgb2bgr=False)
            return Image.fromarray(output_img.astype('uint8'))

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return None


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
            print(f"\nProcessing batch of {len(batch)} images...")
            start_time = time.time()
            for img_path in batch:
                result = processor.process_image(img_path)
                if result:
                    filename = os.path.basename(img_path)
                    output_path = os.path.join("output", f"sr_{filename}")
                    result.save(output_path)
                    os.rename(img_path, os.path.join("processed", filename))
            print(f"Batch done in {time.time() - start_time:.2f}s.")


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
