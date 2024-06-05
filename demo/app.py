import sys
sys.path.append('../')


import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
import torch
from PIL import Image, ImageTk

# from utils.util import load_is_model
from canvas import CanvasImage
from controller import InteractiveController
from wrappers import BoundedNumericalEntry, FocusHorizontalScale, FocusCheckButton, \
    FocusButton, FocusLabelFrame
from models.Unet_id import UNet2D_Test

class ImageEditorApp:
    def __init__(self, root, model, device):
        self.root = root
        self.root.title("Image Editor")

        # 图片变量
        self.image = None
        self.tk_image = None
        self.file_name = None

        self.controller = InteractiveController(model, device, 
                                                update_image_callback=self._update_image)
        
        self._init_state()

        # 创建界面元素
        self.create_widgets()


    def _init_state(self):
        self.state = {
            'predictor_params': {
                'net_clicks_limit': tk.IntVar(value=8)
            },
            'prob_thresh': tk.DoubleVar(value=0.5),
            'lbfgs_max_iters': tk.IntVar(value=20),

            'alpha_blend': tk.DoubleVar(value=0.5),
            'click_radius': tk.IntVar(value=3),
        }

    def create_widgets(self):
        # 选择加载图片按钮
        open_button = tk.Button(self.root, text="选择加载图片", command=self.open_image)
        open_button.pack(pady=10)

        # 保存图片按钮
        save_button = tk.Button(self.root, text="保存分割掩码", command=self.save_mask)
        save_button.pack(pady=10)

        undo_button = tk.Button(self.root, text="撤销点击", command=self.controller.undo_click)
        undo_button.pack(pady=10)

        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(0, weight=1)
        self.canvas_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=5, pady=5)

        # 图片显示区域
        self.canvas = tk.Canvas(self.canvas_frame, width=600, height=600)
        # self.canvas.pack(padx=10, pady=10)
        self.canvas.grid(row=0, column=0, sticky='nswe', padx=5, pady=5)
        self.image_on_canvas = None

    def open_image(self):
        file_path = filedialog.askopenfilename(title="选择图片文件", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
        self.file_name = file_path.split("/")[-1]
        if file_path:
            # 打开图片并显示在界面上
            # self.image = Image.open(file_path).resize((400, 400))
            # self.tk_image = ImageTk.PhotoImage(self.image)
            # self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (512, 512))
            self.controller.set_image(image)

    def save_mask(self):
        
        mask = self.controller.result_mask
        if mask is None:
            return

        filename = filedialog.asksaveasfilename(parent=self.root, 
                                                initialfile=self.file_name.split('.')[0]+"_mask."+self.file_name.split('.')[1],
                                                filetypes=[
            ("PNG image", "*.png"),
            ("BMP image", "*.bmp"),
            ("All files", "*.*"),
        ], title="Save the current mask as...")

        if len(filename) > 0:
            if mask.max() < 256:
                mask = mask.astype(np.uint8)
                mask *= 255 // mask.max()
            mask = cv2.resize(mask, (3000, 3000))
            cv2.imwrite(filename, mask)

    def _reset_last_object(self):
        self.state['alpha_blend'].set(0.5)
        self.state['prob_thresh'].set(0.5)
        self.controller.reset_last_object()

    def _update_image(self, reset_canvas=False):
        image = self.controller.get_visualization(alpha_blend=self.state['alpha_blend'].get(),
                                                  click_radius=self.state['click_radius'].get())
        if self.image_on_canvas is None:
            self.image_on_canvas = CanvasImage(self.canvas_frame, self.canvas)
            self.image_on_canvas.register_click_callback(self._click_callback)

        # self._set_click_dependent_widgets_state()
        if image is not None:
            self.image_on_canvas.reload_image(Image.fromarray(image), reset_canvas)

    def _click_callback(self, is_positive, x, y):
        self.canvas.focus_set()

        
        self.controller.add_click(x, y, is_positive)

    def _check_entry(self, widget):
        all_checked = True
        if widget.winfo_children is not None:
            for w in widget.winfo_children():
                all_checked = all_checked and self._check_entry(w)

        if getattr(widget, "_check_bounds", None) is not None:
            all_checked = all_checked and widget._check_bounds(widget.get(), '-1')

        return all_checked

if __name__ == "__main__":
    check_point = "..\\pretrained_model\\120_epochs.pth"
    # model = load_is_model(check_point, "cude:6", cpu_dist_maps=True)
    model = UNet2D_Test(s_num_classes=1)
    model.load_state_dict(torch.load(check_point))
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    model.to(device)
    root = tk.Tk()
    app = ImageEditorApp(root, model, device=device)
    root.mainloop()