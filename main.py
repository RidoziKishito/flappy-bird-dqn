import os
import sys
import ctypes
from tkinter import (
    Tk, Frame, Canvas, Label, Button, Text, Scrollbar, Toplevel, Entry, StringVar,
    LEFT, RIGHT, BOTH, END, Y, NW, CENTER, Radiobutton
)
from tkinter.font import Font
from PIL import Image, ImageTk

# -------------------------
# Config / Assets
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

BG_MENU = os.path.join(ASSETS_DIR, "bg_menu.png")
TITLE_IMG = os.path.join(ASSETS_DIR, "title.png")
BTN_TRAIN = os.path.join(ASSETS_DIR, "train_new.png")
BTN_LOAD_AND_TRAIN = os.path.join(ASSETS_DIR, "load_and_train.png")
BTN_PLAY_RENDER = os.path.join(ASSETS_DIR, "play_render.png")
BTN_PLAY_NO_RENDER = os.path.join(ASSETS_DIR, "play_no_render.png")

WINDOW_W = 1400
WINDOW_H = 760

# DPI fix Windows
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# -------------------------
# Import train/play functions
# -------------------------
from train import train_loop
from play import play_model, play_model_no_render

# -------------------------
# Helpers
# -------------------------
def load_image_pil(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    im = Image.open(path).convert("RGBA")
    bbox = im.getbbox()
    if bbox:
        im = im.crop(bbox)
    return im

# -------------------------
# Console redirector
# -------------------------
class ConsoleRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, s):
        if not s:
            return
        try:
            self.text_widget.configure(state="normal")
            self.text_widget.insert(END, s)
            self.text_widget.see(END)
            self.text_widget.configure(state="disabled")
            self.text_widget.update_idletasks()
        except Exception:
            pass

    def flush(self):
        pass

# -------------------------
# Main GUI
# -------------------------
class FlappyGUI:
    def __init__(self, root):
        self.root = root
        root.title("Flappy DQN")
        root.geometry(f"{WINDOW_W}x{WINDOW_H}")
        root.resizable(False, False)

        self._images = {}
        self.difficulty_var = StringVar(value="normal")

        # Layout: left 60%, right 40%
        left_w = int(WINDOW_W * 0.60)
        right_w = WINDOW_W - left_w
        self.left_frame = Frame(root, width=left_w, height=WINDOW_H)
        self.left_frame.pack(side=LEFT, fill=BOTH)
        self.right_frame = Frame(root, width=right_w, height=WINDOW_H, bg="#0f1214")
        self.right_frame.pack(side=RIGHT, fill=BOTH)

        # Font setup
        try:
            self.pixel_font_small = Font(root=root, family="Fixedsys", size=12)
            self.pixel_font_med = Font(root=root, family="Fixedsys", size=16)
            self.diff_font = Font(root=root, family="Fixedsys", size=18)
        except Exception:
            self.pixel_font_small = ("TkDefaultFont", 12)
            self.pixel_font_med = ("TkDefaultFont", 16)
            self.diff_font = ("TkDefaultFont", 18)

        # Build UI panels
        self._build_left_panel(left_w)
        self._build_right_panel()

        # Redirect stdout/stderr -> console
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = ConsoleRedirector(self.console_text)
        sys.stderr = ConsoleRedirector(self.console_text)

        # Close handler
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_left_panel(self, left_w):
        left_h = WINDOW_H
        self.left_canvas = Canvas(self.left_frame, width=left_w, height=left_h, highlightthickness=0)
        self.left_canvas.pack(fill=BOTH, expand=True)

        # Background
        try:
            bg = load_image_pil(BG_MENU)
            bw, bh = bg.size
            ratio = max(left_w / bw, left_h / bh)
            new_w, new_h = int(bw * ratio), int(bh * ratio)
            bg_resized = bg.resize((new_w, new_h), Image.LANCZOS)
            self._images["bg"] = ImageTk.PhotoImage(bg_resized)
            offset_x = (left_w - new_w) // 2
            self.left_canvas.create_image(offset_x, 0, anchor=NW, image=self._images["bg"])
        except Exception as e:
            print("[UI] bg load failed:", e)
            self.left_canvas.configure(bg="#dfeeff")

        # Title
        try:
            title = load_image_pil(TITLE_IMG)
            scale = 0.35
            tw, th = title.size
            new_tw, new_th = max(1, int(tw * scale)), max(1, int(th * scale))
            title_resized = title.resize((new_tw, new_th), Image.LANCZOS)
            self._images["title"] = ImageTk.PhotoImage(title_resized)
            x = max(20, (left_w - new_tw) // 2)
            y = 30
            self.left_canvas.create_image(x, y, anchor=NW, image=self._images["title"])
        except Exception as e:
            print("[UI] title load failed:", e)
            self.left_canvas.create_text(left_w // 2, 60, text="FlappyBird", font=("Fixedsys", 36), anchor=CENTER)

        # Buttons
        btn_paths = [
            (BTN_TRAIN, self._on_train),
            (BTN_LOAD_AND_TRAIN, self._on_load_and_train),
            (BTN_PLAY_RENDER, self._on_play_render),
            (BTN_PLAY_NO_RENDER, self._on_play_no_render),
        ]
        max_btn_w = int(left_w * 0.45)
        start_y = int(left_h * 0.32) + 18
        spacing = 26
        for idx, (p, cmd) in enumerate(btn_paths):
            try:
                im = load_image_pil(p)
                scale = 1.9
                new_w = max(1, int(im.width * scale))
                new_h = max(1, int(im.height * scale))
                if new_w > max_btn_w:
                    ratio = max_btn_w / new_w
                    new_w = max(1, int(new_w * ratio))
                    new_h = max(1, int(new_h * ratio))
                im_resized = im.resize((new_w, new_h), Image.LANCZOS)
                photo = ImageTk.PhotoImage(im_resized)
                key = f"btn_{idx}"
                self._images[key] = photo
                btn = Button(self.left_canvas, image=photo, bd=0, relief="flat", command=cmd)
                btn_w, btn_h = photo.width(), photo.height()
                x = max(40, (left_w - btn_w) // 2)
                y = start_y + idx * (btn_h + spacing)
                self.left_canvas.create_window(x, y, anchor=NW, window=btn, width=btn_w, height=btn_h)
            except Exception as e:
                print(f"[UI] button load failed {p}: {e}")
                fallback = Button(self.left_canvas, text=f"BTN {idx}", command=cmd)
                x = max(40, (left_w - 160) // 2)
                y = start_y + idx * (50 + spacing)
                self.left_canvas.create_window(x, y, anchor=NW, window=fallback)

    def _build_right_panel(self):
        padx, pady = 14, 12
        
        # Difficulty label
        Label(self.right_frame, text="Difficulty", bg="#0f1214", fg="#e6f7ee", font=self.diff_font).pack(anchor="nw", padx=padx, pady=(pady, 6))

        # Radiobuttons
        radios_frame = Frame(self.right_frame, bg="#0f1214")
        radios_frame.pack(anchor="nw", padx=padx)
        for val in ("easy", "normal", "hard"):
            rb = Radiobutton(radios_frame, text=val.capitalize(), variable=self.difficulty_var, value=val,
                             indicatoron=1, bg="#0f1214", fg="#e7f9f0", selectcolor="#0f1214",
                             activebackground="#0f1214", activeforeground="#ffffff", highlightthickness=0,
                             anchor="w", padx=8, pady=6, font=self.pixel_font_small)
            rb.pack(side=LEFT, padx=6)

        # Console area
        Label(self.right_frame, text="Console Output", bg="#0f1214", fg="#7adfc9", font=self.pixel_font_small).pack(anchor="nw", padx=padx, pady=(12, 0))
        console_frame = Frame(self.right_frame, bg="#0f1214")
        console_frame.pack(fill=BOTH, expand=True, padx=padx, pady=8)

        console_font = ("Consolas", 12) if os.name == "nt" else ("Courier", 12)

        self.console_text = Text(console_frame, bg="#071013", fg="#7adfc9", wrap="word", state="disabled", bd=0, font=console_font)
        self.console_text.pack(side=LEFT, fill=BOTH, expand=True)

        sb = Scrollbar(console_frame, command=self.console_text.yview)
        sb.pack(side=RIGHT, fill=Y)
        self.console_text.configure(yscrollcommand=sb.set)

        # Clear console button
        clear_btn = Button(self.right_frame, text="Clear Console", command=self._clear_console, bg="#20262a", fg="#fff")
        clear_btn.pack(anchor="se", padx=padx, pady=(4, 12))

    # -------------------------
    # Actions
    # -------------------------
    def _clear_console(self):
        self.console_text.configure(state="normal")
        self.console_text.delete("1.0", END)
        self.console_text.configure(state="disabled")

    def _get_diff(self):
        return self.difficulty_var.get()

    def _on_train(self):
        # Choice 1: train_loop(num_episodes=EPI_NUMS, render=False, resume=False, difficulty=input_dif)
        try:
            from config import EPI_NUMS
            epis = int(EPI_NUMS)
        except Exception:
            epis = 100
        diff = self._get_diff()
        train_loop(num_episodes=epis, render=False, resume=False, difficulty=diff)

    def _on_load_and_train(self):
        # Choice 2: train_loop(num_episodes=EPI_NUMS, render=False, resume=True)
        try:
            from config import EPI_NUMS
            epis = int(EPI_NUMS)
        except Exception:
            epis = 100
        diff = self._get_diff()
        train_loop(num_episodes=epis, render=False, resume=True, difficulty=diff)

    def _on_play_render(self):
        # Choice 3: play_model(render=True, dif=input_dif)
        diff = self._get_diff()
        play_model(render=True, dif=diff)

    def _on_play_no_render(self):
        # Choice 4: play_model_no_render(target_score=500, num_episodes=5, dif=input_dif)
        # Popup to get parameters
        popup = Toplevel(self.root)
        popup.title("Play (no render)")
        popup.geometry("360x200")
        popup.resizable(False, False)
        
        Label(popup, text="Target score:", font=("Fixedsys", 11)).pack(pady=(15, 5))
        target_entry = Entry(popup, width=20, font=("Fixedsys", 11))
        target_entry.pack(pady=(0, 10))
        target_entry.insert(0, "500")

        Label(popup, text="Number of episodes:", font=("Fixedsys", 11)).pack(pady=(5, 5))
        num_entry = Entry(popup, width=20, font=("Fixedsys", 11))
        num_entry.pack(pady=(0, 15))
        num_entry.insert(0, "5")

        def start():
            tval = target_entry.get().strip()
            nval = num_entry.get().strip()
            try:
                target = int(tval)
            except Exception:
                print("[UI] Invalid target_score; using 500")
                target = 500
            try:
                num = int(nval)
            except Exception:
                print("[UI] Invalid num_episodes; using 5")
                num = 5
            popup.destroy()
            diff = self._get_diff()
            play_model_no_render(target_score=target, num_episodes=num, dif=diff)

        Button(popup, text="Run", command=start, width=12, bg="#20262a", fg="#fff", font=("Arial", 11)).pack(pady=10)

    def _on_close(self):
        # Restore stdout/stderr
        try:
            sys.stdout = self._orig_stdout
            sys.stderr = self._orig_stderr
        except Exception:
            pass
        self.root.destroy()

# -------------------------
# Entrypoint
# -------------------------
def main():
    root = Tk()
    app = FlappyGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()