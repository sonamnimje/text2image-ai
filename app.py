import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch, uuid, os, shutil, subprocess, sys
import threading # Import the threading module

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

# === Config ===
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
model_path = "./realisticVisionV60B1_v51HyperVAE.safetensors"
 # Removed invalid pipeline instantiation
device      = "cuda" if torch.cuda.is_available() else "cpu"

# === Dummy safety checker ===
def dummy_checker(images, **kwargs):
    return images, [False] * len(images)

# === Load txt-2-img pipeline once ===
print("Loading model …")
try:
    pipe = StableDiffusionPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None
    ).to(device)
    pipe.safety_checker = dummy_checker
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    messagebox.showerror("Model Load Error", f"Failed to load the model: {e}\nPlease ensure 'realisticVisionV60B1_v51HyperVAE.safetensors' is in the same directory as the script and all dependencies are installed.")
    sys.exit(1)

img2img_pipe        = None
uploaded_image_path = None
last_output_path    = None

# === GUI ===
app = tk.Tk()
app.title("AI Image Generator")
app.geometry("700x750")
app.minsize(width=600, height=650)

# --- Dark Theme Setup ---
style = ttk.Style(app)
style.theme_use('clam')

DARK_BG = "#2e2e2e"
LIGHT_FG = "#e0e0e0"
ACCENT_GREEN = "#4CAF50"
ACCENT_BLUE = "#2196F3"
DARK_BUTTON_BG = "#555555"
IMAGE_PANEL_BG = "#424242"

app.configure(bg=DARK_BG)
style.configure('.', background=DARK_BG, foreground=LIGHT_FG, font=("Arial", 10))
style.configure('TLabel', background=DARK_BG, foreground=LIGHT_FG)
style.configure('TEntry', fieldbackground="#424242", foreground=LIGHT_FG, insertcolor=LIGHT_FG)
style.configure('TButton', background=DARK_BUTTON_BG, foreground=LIGHT_FG,
                font=("Arial", 10, "bold"), borderwidth=1, relief="raised")
style.map('TButton', background=[('active', '#777777')], foreground=[('active', 'white')])

style.configure('Generate.TButton', background=ACCENT_GREEN, foreground='white',
                font=("Arial", 12, "bold"), borderwidth=0)
style.map('Generate.TButton', background=[('active', '#66BB6A')])

style.configure('Download.TButton', background=DARK_BUTTON_BG, foreground='white',
                font=("Arial", 11), borderwidth=0)
style.map('Download.TButton', background=[('active', '#6a6a6a')])

style.configure('Copy.TButton', background=DARK_BUTTON_BG, foreground='white',
                font=("Arial", 10), borderwidth=0)
style.map('Copy.TButton', background=[('active', '#6a6a6a')])

style.configure('ImagePanel.TLabel', background=IMAGE_PANEL_BG, foreground="#999999",
                relief="solid", borderwidth=1, anchor="center", font=("Arial", 12))


prompt_var        = tk.StringVar()
progress_var      = tk.StringVar()
output_path_var   = tk.StringVar(value="Not saved yet.")

output_panel   = None # Will be initialized later in layout section

# --- Button references for disabling/enabling ---
generate_button = None
browse_button = None
download_button = None
copy_path_button = None

# ---------- helpers ----------
def browse_image():
    global uploaded_image_path
    file_path = filedialog.askopenfilename(
        title="Choose base image",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.webp")]
    )
    if file_path:
        uploaded_image_path = file_path
        messagebox.showinfo("Image Uploaded", f"Base image selected:\n{os.path.basename(file_path)}")

def _enable_buttons():
    """Re-enables all relevant buttons."""
    generate_button.config(state=tk.NORMAL)
    browse_button.config(state=tk.NORMAL)
    download_button.config(state=tk.NORMAL)
    copy_path_button.config(state=tk.NORMAL)

def _update_gui_after_generation(result_img_pil, full_output_path, error):
    """
    Called from the main thread after the worker thread finishes.
    Updates the GUI safely.
    """
    _enable_buttons() # Always re-enable buttons first

    if error:
        progress_var.set("")
        messagebox.showerror("Error", f"An error occurred during generation: {str(error)}")
        print(f"Error during image generation: {error}")
    else:
        # Resize PIL image and create PhotoImage in the main thread
        outTk = ImageTk.PhotoImage(result_img_pil.resize((256, 256)))
        output_panel.config(image=outTk, text="")
        output_panel.image = outTk # Keep a reference

        progress_var.set("Generation complete ✅")
        output_path_var.set(full_output_path)

    # Force a redraw of the entire window after final updates
    app.update_idletasks()


def _generate_image_worker():
    """
    This function runs in a separate thread and performs the heavy lifting.
    It should NOT directly update Tkinter widgets.
    """
    global img2img_pipe, uploaded_image_path, last_output_path

    try:
        prompt = prompt_var.get().strip() # Access variable safely (read-only here)

        if uploaded_image_path:            # ========== img2img ==========
            if img2img_pipe is None:
                print("Loading img2img pipeline…")
                img2img_pipe = StableDiffusionImg2ImgPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    safety_checker=None
                ).to(device)
                img2img_pipe.safety_checker = dummy_checker
                img2img_pipe.enable_attention_slicing()
                print("img2img pipeline loaded.")

            init_img = (Image.open(uploaded_image_path)
                              .convert("RGB")
                              .resize((512, 512)))
            result_img = img2img_pipe(
                prompt          = prompt,
                image           = init_img,
                strength        = 0.75,
                guidance_scale  = 7.5
            ).images[0]

        else:                              # ========== txt2img ==========
            result_img = pipe(
                prompt             = prompt,
                height             = 512,
                width              = 512,
                num_inference_steps= 50,
                guidance_scale     = 7.5
            ).images[0]

        # ---------- save ----------
        filename         = f"{uuid.uuid4().hex}.png"
        full_output_path = os.path.abspath(os.path.join(OUTPUT_DIR, filename))
        result_img.save(full_output_path)
        last_output_path = full_output_path # Update global variable

        # Schedule the GUI update to happen on the main thread
        app.after(0, _update_gui_after_generation, result_img, full_output_path, None)

    except Exception as e:
        # Schedule error handling to happen on the main thread
        app.after(0, _update_gui_after_generation, None, None, e)

def generate_image():
    """
    This function is called by the button click. It prepares the UI
    and starts the generation in a new thread.
    """
    prompt = prompt_var.get().strip()
    if not prompt:
        messagebox.showerror("Missing prompt", "Please enter a prompt.")
        return

    progress_var.set("Generating image… please wait ⏳")
    output_path_var.set("Not saved yet.")

    # Disable buttons to prevent re-clicks while busy
    generate_button.config(state=tk.DISABLED)
    browse_button.config(state=tk.DISABLED)
    download_button.config(state=tk.DISABLED)
    copy_path_button.config(state=tk.DISABLED)

    app.update_idletasks() # Ensure status updates immediately

    # Start the worker thread
    # daemon=True means the thread will automatically exit when the main program exits
    threading.Thread(target=_generate_image_worker, daemon=True).start()


def download_image():
    if not last_output_path or not os.path.exists(last_output_path):
        messagebox.showerror("No image", "Please generate an image first or the last generated image could not be found.")
        return
    save_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG Image", "*.png"), ("All files", "*.*")]
    )
    if save_path:
        try:
            shutil.copy(last_output_path, save_path)
            messagebox.showinfo("Saved", f"Image saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save image: {e}")

def copy_path_to_clipboard():
    current_path = output_path_var.get()
    if current_path and current_path != "Not saved yet.":
        app.clipboard_clear()
        app.clipboard_append(current_path)
        messagebox.showinfo("Path copied", "Output path copied to clipboard!")
    else:
        messagebox.showinfo("No Path", "No output path available to copy yet.")


# ---------- layout ----------
ttk.Label(app, text="Enter Prompt:", font=("Arial", 11)).pack(pady=(20,5))
ttk.Entry(app, textvariable=prompt_var, width=50, font=("Arial", 11)).pack(padx=20)

# Store button references globally
browse_button = ttk.Button(app, text="Upload Base Image (optional)", command=browse_image)
browse_button.pack(pady=10)

generate_button = ttk.Button(app, text="Generate Image", command=generate_image, style='Generate.TButton')
generate_button.pack(pady=5)

ttk.Label(app, textvariable=progress_var, foreground=ACCENT_BLUE, font=("Arial", 11, "italic")).pack(pady=10)
ttk.Label(app, textvariable=output_path_var, wraplength=600, font=("Arial", 9)).pack(pady=(0,5))

copy_path_button = ttk.Button(app, text="Copy output path", command=copy_path_to_clipboard, style='Copy.TButton')
copy_path_button.pack(pady=(0,10))

# --- output image panel ---
output_panel   = ttk.Label(app, text="Generated Image",
                           style='ImagePanel.TLabel',
                           compound="image"
                           )
output_panel.pack(pady=10, padx=20, expand=True, fill='both')


download_button = ttk.Button(app, text="Download Image", command=download_image, style='Download.TButton')
download_button.pack(pady=15)

# Start the Tkinter event loop
app.mainloop()