import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import torch
import os 
import numpy as np
from PIL import Image, ImageTk
import cv2
from model import HTDNet
from threading import Thread
import queue
import time

class FPSCounter:
    def __init__(self, avg_frames=30):
        self.avg_frames = avg_frames
        self.times = []
        
    def start(self):
        self.times = []
        self.start_time = time.time()
        
    def update(self):
        self.times.append(time.time())
        if len(self.times) > self.avg_frames:
            self.times.pop(0)
        
        if len(self.times) > 1:
            fps = (len(self.times) - 1) / (self.times[-1] - self.times[0])
            return fps
        return 0

class VideoProcessor:
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2
        self.stop_processing = False
        self.processing_queue = queue.Queue(maxsize=30)
        self.output_queue = queue.Queue(maxsize=30)
        self.processing_thread = None
        self.fps_counter = FPSCounter()
        self.current_capture = None

    def process_frame(self, frame):
        # Chuyển đổi frame sang định dạng phù hợp
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        haze = np.array(frame_pil) / 255
        haze_tensor = torch.Tensor(haze.transpose(2, 0, 1)[np.newaxis, :, :, :])

        # Xử lý frame bằng cả 2 model
        with torch.no_grad():
            _, _, _, out_frame1 = self.model1(haze_tensor)
            _, _, _, out_frame2 = self.model2(haze_tensor)
        
        # Chuyển đổi kết quả về numpy array
        processed_frame1 = out_frame1[0].cpu().numpy().transpose(1, 2, 0)
        processed_frame2 = out_frame2[0].cpu().numpy().transpose(1, 2, 0)
        
        # Kết hợp kết quả từ 2 model (có thể điều chỉnh trọng số)
        combined_frame = (processed_frame1 * 0.9 + processed_frame2 * 0.1)
        combined_frame = (combined_frame * 255).clip(0, 255).astype(np.uint8)
        
        return cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)

    def processing_worker(self):
        while not self.stop_processing:
            try:
                frame = self.processing_queue.get(timeout=1)
                processed_frame = self.process_frame(frame)
                self.output_queue.put((frame, processed_frame))
                self.processing_queue.task_done()
            except queue.Empty:
                continue

    def process_video(self, video_path=None, is_camera=False):
        # Ensure previous processing is stopped and cleaned up
        self.cleanup()
        
        # Start new capture
        self.current_capture = cv2.VideoCapture(0 if is_camera else video_path)
        if not self.current_capture.isOpened():
            messagebox.showerror("Error", "Could not open video source")
            return

        self.stop_processing = False
        
        # Start processing thread
        self.processing_thread = Thread(target=self.processing_worker)
        self.processing_thread.start()

        self.fps_counter.start()

        try:
            while self.current_capture and self.current_capture.isOpened() and not self.stop_processing:
                ret, frame = self.current_capture.read()
                if not ret:
                    break

                try:
                    self.processing_queue.put(frame, timeout=1)
                except queue.Full:
                    continue

                try:
                    original_frame, processed_frame = self.output_queue.get(timeout=1)
                    
                    combined_frame = np.hstack((original_frame, processed_frame))
                    
                    fps = self.fps_counter.update()
                    cv2.putText(combined_frame, f"FPS: {fps:.1f}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.putText(combined_frame, "Press 'a' to stop", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    cv2.imshow('Video Processing (Combined Method)', combined_frame)

                except queue.Empty:
                    continue

                key = cv2.waitKey(1) & 0xFF
                if key == ord('a'):
                    print("Stopping video processing...")
                    break

        finally:
            self.cleanup()

    def cleanup(self):
        self.stop_processing = True
        
        # Clear queues
        while not self.processing_queue.empty():
            try:
                self.processing_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break

        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join()
            self.processing_thread = None

        # Release capture device
        if self.current_capture:
            self.current_capture.release()
            self.current_capture = None

        cv2.destroyAllWindows()

class ImageProcessor:
    def __init__(self, model):
        self.model = model

    def process_image(self, image_path):
        haze = np.array(Image.open(image_path)) / 255
        haze_tensor = torch.Tensor(haze.transpose(2, 0, 1)[np.newaxis, :, :, :])
        with torch.no_grad():
            _, _, _, out = self.model(haze_tensor)
        return out[0].cpu().numpy().transpose(1, 2, 0)

def load_model(checkpoint_dir, model_name='Fmodel.tar'):
    if os.path.exists(checkpoint_dir + model_name):
        Fmodel_info = torch.load(checkpoint_dir + model_name, map_location=torch.device('cpu'))
        print('==> Loading existing model:', checkpoint_dir + model_name)

        FNet = HTDNet()
        
        state_dict = Fmodel_info['state_dict']
        new_state_dict = {}
        for key in state_dict.keys():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = state_dict[key]
        
        FNet.load_state_dict(new_state_dict)
        FNet.eval()
        return FNet
    else:
        raise FileNotFoundError("Model checkpoint not found.")

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ Thống Khử Sương Mù")
        
        # Load models
        model_checkpoint_dir = "./checkpoint/"
        self.model_method1 = load_model(model_checkpoint_dir, 'Fmodel.tar')
        self.model_method2 = load_model(model_checkpoint_dir, 'Fmodel_non_haze.tar')
        
        # Initialize processors
        self.image_processor1 = ImageProcessor(self.model_method1)
        self.image_processor2 = ImageProcessor(self.model_method2)
        self.video_processor = VideoProcessor(self.model_method1, self.model_method2)
        
        self.setup_ui()

    def setup_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(pady=10, expand=True)

        self.image_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.image_frame, text='Xử lý ảnh')

        self.video_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.video_frame, text='Xử lý video')

        self.setup_image_tab()
        self.setup_video_tab()

    def setup_image_tab(self):
        display_frame = tk.Frame(self.image_frame)
        display_frame.pack(pady=20)

        self.original_label = tk.Label(display_frame, text="Ảnh Gốc")
        self.original_label.grid(row=0, column=0, padx=10)
        
        self.processed_label_method1 = tk.Label(display_frame, text="Phương Pháp 1")
        self.processed_label_method1.grid(row=0, column=1, padx=10)
        
        self.processed_label_method2 = tk.Label(display_frame, text="Phương Pháp 2")
        self.processed_label_method2.grid(row=0, column=2, padx=10)

        control_frame = tk.Frame(self.image_frame)
        control_frame.pack(pady=10)

        tk.Button(control_frame, text="Xử lý ảnh (Phương pháp 1)",
                 command=lambda: self.process_image(1)).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Xử lý ảnh (Phương pháp 2)",
                 command=lambda: self.process_image(2)).pack(side=tk.LEFT, padx=5)

    def setup_video_tab(self):
        control_frame = tk.Frame(self.video_frame)
        control_frame.pack(pady=20)

        tk.Button(control_frame, text="Xử lý Video File",
                 command=lambda: self.select_video_for_processing(False)
                 ).pack(pady=5)
        
        tk.Button(control_frame, text="Xử lý Camera",
                 command=lambda: self.select_video_for_processing(True)
                 ).pack(pady=5)

    def process_image(self, method):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return

        original_image = Image.open(file_path)
        original_image.thumbnail((400, 400))
        original_photo = ImageTk.PhotoImage(original_image)
        self.original_label.configure(image=original_photo)
        self.original_label.image = original_photo

        processor = self.image_processor1 if method == 1 else self.image_processor2
        processed_array = processor.process_image(file_path)
        processed_array = (processed_array * 255).astype(np.uint8)
        processed_image = Image.fromarray(processed_array)
        processed_image.thumbnail((400, 400))
        processed_photo = ImageTk.PhotoImage(processed_image)
        
        label = self.processed_label_method1 if method == 1 else self.processed_label_method2
        label.configure(image=processed_photo)
        label.image = processed_photo

    def select_video_for_processing(self, is_camera=False):
        if is_camera:
            self.video_processor.process_video(is_camera=True)
        else:
            video_path = filedialog.askopenfilename(
                filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov")]
            )
            if video_path:
                self.video_processor.process_video(video_path)

    def on_closing(self):
        self.video_processor.cleanup()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()