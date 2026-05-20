"""
Barchan Dune Segmentation — Flask Backend
==========================================
يشغّل هذا السيرفر الموديلز الأربعة محلياً ويرسل النتائج للـ HTML

متطلبات التشغيل (في Terminal داخل VS Code):
    pip install flask flask-cors torch torchvision segmentation-models-pytorch pillow numpy
    python app.py

بعدين افتح index.html في المتصفح — كل شيء يشتغل تلقائياً.
"""

import io
import base64
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import segmentation_models_pytorch as smp
from torchvision import transforms

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ─────────────────────────────────────────
#  إعدادات — غيّر المسارات حقت الـ .pth فقط
# ─────────────────────────────────────────
MODEL_PATHS = {
    "unet":    "/Users/monirah/Downloads/protoype22/Models/unet_CLAHE_pretrained_best.pth",       # ← اسم الملف الحقيقي
    "unetpp":  "/Users/monirah/Downloads/protoype22/Models/unetplusplus_no preprocessing_pretrained_best.pth",     # ← اسم الملف الحقيقي
    "deepv3":  "/Users/monirah/Downloads/protoype22/Models/deeplabv3_CLAHE_pretrained_best.pth",  # ← اسم الملف الحقيقي
    "deepv3p": "/Users/monirah/Downloads/protoype22/Models/deeplabv3plus_no preprocessing_pretrained_best.pth", # ← اسم الملف الحقيقي
}

# نفس إعدادات التدريب من نوتبوكاتكم
IMAGE_SIZE  = 256
IMAGE_MEAN  = (0.485, 0.456, 0.406)
IMAGE_STD   = (0.229, 0.224, 0.225)
THRESHOLD   = 0.5
ENCODER     = "resnet34"

# ─────────────────────────────────────────
#  تحميل الموديلز عند بدء السيرفر
# ─────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

def build_model(key):
    """يبني هيكل الموديل — نفس الكود من نوتبوكاتكم"""
    kwargs = dict(encoder_name=ENCODER, encoder_weights=None, in_channels=3, classes=1, activation=None)
    if key == "unet":    return smp.Unet(**kwargs)
    if key == "unetpp":  return smp.UnetPlusPlus(**kwargs)
    if key == "deepv3":  return smp.DeepLabV3(**kwargs)
    if key == "deepv3p": return smp.DeepLabV3Plus(**kwargs)

def load_models():
    models = {}
    for key, path in MODEL_PATHS.items():
        pth = Path(path)
        if not pth.exists():
            print(f"⚠️  ملف الموديل ما لقيته: {path}  — تأكد من المسار")
            continue
        model = build_model(key)
        checkpoint = torch.load(pth, map_location=device)
        # نوتبوكاتكم تحفظ بالصيغة: {"model_state_dict": ..., ...}
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state)
        model.eval().to(device)
        models[key] = model
        print(f"✅  {key} loaded from {path}")
    return models

models = load_models()

# ─────────────────────────────────────────
#  Preprocessing — نفس val_test_transform من نوتبوكاتكم
# ─────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
])

def image_to_tensor(pil_img):
    """PIL RGB Image → (1,3,256,256) tensor"""
    return preprocess(pil_img).unsqueeze(0).to(device)

def mask_to_base64(mask_np):
    """numpy array (256,256) float → base64 PNG string"""
    mask_uint8 = (mask_np * 255).astype(np.uint8)
    pil = Image.fromarray(mask_uint8, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ─────────────────────────────────────────
#  Flask App
# ─────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
CORS(app)  # يسمح للـ HTML يتكلم مع السيرفر

@app.route('/desert-background.png')
def desert_background():
    return send_from_directory(BASE_DIR, 'desert-background.png')

@app.route('/barchan-transition.png')
def barchan_transition():
    return send_from_directory(BASE_DIR, 'barchan-transition.png')

@app.route('/newbackground.JPG')
def new_background():
    return send_from_directory(BASE_DIR, 'newbackground.JPG')

@app.route('/withoutWords.PNG')
def without_words_logo_PNG():
    return send_from_directory(BASE_DIR, 'withoutWords.PNG')

@app.route('/project-logo.png')
def project_logo():
    return send_from_directory(BASE_DIR, 'project-logo.png')

@app.route('/withoutWords.svg')
def without_words_logo_svg():
    return send_from_directory(BASE_DIR, 'withoutWords.svg')
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "barchan_final.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "loaded": list(models.keys()), "device": device})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files["image"]
    pil_img  = Image.open(img_file).convert("RGB")
    tensor   = image_to_tensor(pil_img)

    results = {}

    with torch.no_grad():
        for key, model in models.items():
            logits = model(tensor)                              # (1,1,256,256)
            probs  = torch.sigmoid(logits).squeeze().cpu().numpy()  # (256,256) float 0-1
            mask   = (probs >= THRESHOLD).astype(np.float32)   # binary 0/1

            total_pixels = mask.size

            # ── أرقام حقيقية محسوبة من الصورة ──
            confidence    = float(np.mean(probs))               # متوسط ثقة الموديل
            dune_coverage = float(np.sum(mask) / total_pixels)  # نسبة مساحة الـ dune
            mean_activation = float(np.mean(probs[mask == 1])) if np.sum(mask) > 0 else 0.0  # متوسط activation داخل الـ mask

            # Boundary sharpness — كم الحواف واضحة
            from scipy.ndimage import sobel
            edge = np.hypot(sobel(mask, axis=0), sobel(mask, axis=1))
            sharpness = float(np.mean(edge[edge > 0])) if np.any(edge > 0) else 0.0

            results[key] = {
                "mask":            mask_to_base64(mask),
                "confidence":      round(confidence * 100, 2),        # %
                "dune_coverage":   round(dune_coverage * 100, 2),     # %
                "mean_activation": round(mean_activation, 4),
                "sharpness":       round(sharpness, 4),
            }

    return jsonify(results)

if __name__ == "__main__":
    print("\n🚀  افتحي المتصفح على: http://localhost:5000")
    print("    ما تحتاجين Live Server — السيرفر يخدم الموقع مباشرة\n")
    app.run(port=5000, debug=False)