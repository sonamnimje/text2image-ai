# Text-to-Image Generation Using Python AI

Generate high-quality images from text prompts using the latest Realistic Vision v6.0 (HyperVAE) model, fully offline with Python. No API keys or internet required after setup. GPU acceleration supported.


---

## 🚀 Features
- Text-to-Image generation (txt2img)
- Image-to-Image generation (img2img, optional input image)
- Based on Stable Diffusion Pipeline
- Tkinter GUI (modern dark theme)
- Transparent PNG output (512x512)
- Output folder with preview and download
- Unfiltered: no safety checker (full creative control)

---

## 📁 Project Structure
```
├── app.py                # Main Tkinter GUI application
├── requirements.txt      # Python dependencies
├── runtime.txt           # Python version info
├── output/               # Generated images
├── static/frames/        # (Optional) UI frames/assets
├── README.md             # Project documentation
```

---

## 🖥️ System Requirements
- **OS:** Windows / Linux / macOS
- **Python:** 3.10+ (see `runtime.txt`)
- **RAM:** 8GB minimum (16GB+ recommended)
- **CUDA GPU:** Optional, but recommended for best performance

---

## ⚡ Installation

### 1. Clone the Repository
```sh
https://github.com/sonamnimje/text2image-ai.git
cd text2image-ai
```

### 2. Create Virtual Environment (Recommended)
```sh
python -m venv venv
```
Activate it:
- **Windows:**
  ```sh
  venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```sh
  source venv/bin/activate
  ```

### 3. Install Requirements
```sh
pip install -r requirements.txt
```

### 4. Download Model File
- Model: `realisticVisionV60B1_v51HyperVAE.safetensors`
- Source: [Civitai](https://civitai.com/models/4201/realistic-vision-v60)
- Place the file in the same directory as `app.py`

---

## 🧑‍💻 Usage

### Run the Tkinter GUI
```sh
python app.py
```
- Enter a text prompt and click "Generate Image".
- (Optional) Upload a base image for img2img.
- Download or copy the output path after generation.

### Output
- Generated images are saved in the `output/` folder as PNG files.

---

## ⚠️ Important Note
This app uses an unfiltered AI model. NSFW, biased, or inappropriate results are possible depending on the prompt. For educational and personal use only. Use responsibly.

---

## 🙏 Credits
Developed by Sonam Nimje💫

---
### Enjoy exploring AI creativity—Happy Prompting!
