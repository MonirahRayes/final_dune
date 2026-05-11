# Barchan Dune Segmentation — إعداد المشروع

## هيكل الفولدر المطلوب

```
my-project/
├── index.html               ← الملف barchan_final.html (غيّر اسمه)
├── app.py                   ← السيرفر
├── models/
│   ├── unet_pretrained_best.pth
│   ├── unetpp_pretrained_best.pth
│   ├── deeplabv3_pretrained_best.pth
│   └── deeplabv3plus_pretrained_best.pth
```

---

## الخطوات

### 1. حمّل الموديلز من Google Drive
انزّل الملفات الأربعة هذي وحطها في فولدر `models/`:
- `unet_pretrained_best.pth`
- `unetplusplus_pretrained_best.pth`  ← غيّر اسمه لـ `unetpp_pretrained_best.pth`
- `deeplabv3_pretrained_best.pth`
- `deeplabv3plus_pretrained_best.pth`

### 2. تثبيت المكتبات (مرة وحدة فقط)
افتح Terminal في VS Code وشغّل:
```
pip install flask flask-cors torch torchvision segmentation-models-pytorch pillow numpy
```

### 3. شغّل السيرفر
```
python app.py
```
المفروض تشوف:
```
✅  unet loaded
✅  unetpp loaded
✅  deepv3 loaded
✅  deepv3p loaded
🚀  السيرفر يشتغل على http://localhost:5000
```

### 4. افتح الموقع
افتح `index.html` في المتصفح — ارفع صورة واضغط **Run All Models**

---

## ملاحظات مهمة
- السيرفر لازم يكون شغّال طول ما تستخدم الموقع
- إذا أغلقت الـ Terminal — السيرفر يوقف
- يشتغل على الجهاز المحلي فقط — للنشر على الإنترنت راجع خيارات الـ Deploy
