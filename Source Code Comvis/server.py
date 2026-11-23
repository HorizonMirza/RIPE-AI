import os
import base64
import traceback
import numpy as np
import cv2
import joblib
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ─── Konfigurasi Path Model (Gunakan Absolute Path agar aman) ─────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "fruit_svm_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "fruit_scaler.pkl")

model  = None
scaler = None
classes = None

def load_model():
    global model, scaler, classes
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] Model tidak ditemukan di {MODEL_PATH}")
        return False
    try:
        model   = joblib.load(MODEL_PATH)
        scaler  = joblib.load(SCALER_PATH)
        # Ambil list kelas langsung dari atribut bawaan model Scikit-Learn
        classes = model.classes_.tolist() 
        print(f"[OK] Model loaded: {MODEL_PATH}")
        print(f"[OK] Classes found: {classes}")
        return True
    except Exception as e:
        print(f"[ERROR] Gagal load model: {e}")
        return False

model_loaded = load_model()


# ─── Helper: Mapping Label Kaggle ke Format Frontend ─────────────────────────
def map_kaggle_label(raw_label):
    mapping = {
        'RipeBanana':      ('banana',     'ripe'),
        'RottenBanana':    ('banana',     'rotten'),
        'UnripeBanana':    ('banana',     'unripe'),
        'RipeStrawberry':  ('strawberry', 'ripe'),
        'RottenStrawberry':('strawberry', 'rotten'),
        'UnripeStrawberry':('strawberry', 'unripe'),
        'RipeOrange':      ('orange',     'ripe'),
        'RottenOrange':    ('orange',     'rotten'),
        'UnripeOrange':    ('orange',     'unripe'),
    }
    ft, rs = mapping.get(raw_label, (raw_label, 'unknown'))
    return f"{ft}_{rs}", ft, rs


# ─── Feature extraction — exact copy dari notebook (GrabCut version) ─────────
def extract_features_from_array(img_array, size=(128, 128)):
    img_resized = cv2.resize(img_array, size)
    gray    = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # ── Segmentation: GrabCut (matches notebook) ──────────────────────────────
    h, w = blurred.shape[:2]
    margin = int(min(h, w) * 0.085)
    rect = (margin, margin, w - margin * 2, h - margin * 2)
    mask      = np.zeros(blurred.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_resized, mask, rect, bgd_model, fgd_model,
                iterCount=20, mode=cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')

    # ── Shape features ────────────────────────────────────────────────────────
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    aspect_ratio, extent = 0, 0
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(c)
        aspect_ratio = float(bw) / bh if bh != 0 else 0
        area      = cv2.contourArea(c)
        rect_area = bw * bh
        extent    = float(area) / rect_area if rect_area != 0 else 0

    fp = mask2 > 0

    # ── HSV color features (6) ────────────────────────────────────────────────
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    hsv_feats = [
        np.mean(h_ch[fp]) if fp.any() else 0,
        np.mean(s_ch[fp]) if fp.any() else 0,
        np.mean(v_ch[fp]) if fp.any() else 0,
        np.std(h_ch[fp])  if fp.any() else 0,
        np.std(s_ch[fp])  if fp.any() else 0,
        np.std(v_ch[fp])  if fp.any() else 0,
    ]

    # ── LAB color features (5) ────────────────────────────────────────────────
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    lab_feats = [
        np.mean(l_ch[fp]) if fp.any() else 0,
        np.mean(a_ch[fp]) if fp.any() else 0,
        np.mean(b_ch[fp]) if fp.any() else 0,
        np.std(a_ch[fp])  if fp.any() else 0,
        np.std(b_ch[fp])  if fp.any() else 0,
    ]

    # ── Hue histogram 18 bins (18) ────────────────────────────────────────────
    # NOTE: uses mask2 (uint8 0/255) as the cv2.calcHist mask — matches notebook
    h_hist = cv2.calcHist([h_ch], [0], mask2, [18], [0, 180])
    h_hist = cv2.normalize(h_hist, h_hist).flatten().tolist()

    # ── GLCM texture (6) ──────────────────────────────────────────────────────
    # Crop to bounding box of mask, inpaint background pixels, then quantise.
    # This exactly replicates the notebook's GLCM pipeline.
    x, y, bw, bh = cv2.boundingRect(mask2)
    if bw > 0 and bh > 0:
        gray_crop    = gray[y:y + bh, x:x + bw]
        mask_crop    = mask2[y:y + bh, x:x + bw]
        masked_gray_raw = np.where(mask_crop > 0, gray_crop, 0).astype(np.uint8)
        inv_mask_crop   = cv2.bitwise_not(mask_crop)
        if np.count_nonzero(inv_mask_crop) > 0:
            inpainted = cv2.inpaint(masked_gray_raw, inv_mask_crop,
                                    inpaintRadius=1, flags=cv2.INPAINT_TELEA)
            masked_gray = inpainted if inpainted is not None else masked_gray_raw
        else:
            masked_gray = masked_gray_raw
    else:
        # GrabCut returned empty mask — fall back to full grayscale
        masked_gray = gray

    masked_gray_q = (masked_gray // 8).astype(np.uint8)
    valid_pixels  = masked_gray_q[masked_gray_q > 0]
    if valid_pixels.size < 100:
        # Fallback: use full unmasked grayscale
        glcm_input = (gray // 8).astype(np.uint8)
    else:
        glcm_input = masked_gray_q

    glcm = graycomatrix(glcm_input, distances=[1, 3, 5],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=32, symmetric=True, normed=True)
    glcm_feats = [
        graycoprops(glcm, 'contrast').mean(),
        graycoprops(glcm, 'correlation').mean(),
        graycoprops(glcm, 'energy').mean(),
        graycoprops(glcm, 'homogeneity').mean(),
        graycoprops(glcm, 'dissimilarity').mean(),
        graycoprops(glcm, 'ASM').mean(),
    ]

    # ── LBP texture 10 bins (10) ──────────────────────────────────────────────
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_pixels = lbp[fp] if fp.any() else lbp.ravel()
    lbp_hist, _ = np.histogram(lbp_pixels, bins=10, range=(0, 10), density=True)

    features = hsv_feats + lab_feats + h_hist + glcm_feats + lbp_hist.tolist() + [aspect_ratio, extent]

    # raw dict for frontend display
    raw = {
        'h_mean': hsv_feats[0], 's_mean': hsv_feats[1], 'v_mean': hsv_feats[2],
        'h_std':  hsv_feats[3], 's_std':  hsv_feats[4], 'v_std':  hsv_feats[5],
        'contrast': glcm_feats[0], 'correlation': glcm_feats[1],
        'energy': glcm_feats[2],   'homogeneity': glcm_feats[3],
        'aspect_ratio': aspect_ratio, 'extent': extent,
    }
    return features, raw


# ─── CORS headers helper ──────────────────────────────────────────────────────
def add_cors(response):
    response.headers['Access-Control-Allow-Origin']  = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.after_request
def after_request(response):
    return add_cors(response)

@app.route('/predict', methods=['OPTIONS'])
@app.route('/health',  methods=['OPTIONS'])
@app.route('/classes', methods=['OPTIONS'])
def options():
    resp = jsonify({})
    return add_cors(resp)


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    # Frontend sekarang akan menerima mapped classes
    mapped_classes = [map_kaggle_label(c)[0] for c in classes] if classes else []
    return jsonify({
        'status':       'ok',
        'model_loaded': model_loaded,
        'classes':      mapped_classes, 
        'model_path':   MODEL_PATH,
    })


@app.route('/classes', methods=['GET'])
def get_classes():
    mapped_classes = [map_kaggle_label(c)[0] for c in classes] if classes else []
    return jsonify({'classes': mapped_classes})


@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded or model is None:
        return jsonify({'error': 'Model belum diload.'}), 503

    try:
        if request.is_json:
            data = request.get_json()
            image_b64 = data.get('image', '')
            if ',' in image_b64:
                image_b64 = image_b64.split(',', 1)[1]
            img_bytes = base64.b64decode(image_b64)
            nparr     = np.frombuffer(img_bytes, np.uint8)
            img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            return jsonify({'error': 'Input tidak valid.'}), 400

        if img_array is None:
            return jsonify({'error': 'Gagal decode gambar.'}), 400

        # Ekstraksi & Scaling
        features, raw = extract_features_from_array(img_array)
        features_scaled = scaler.transform([features])

        # Prediksi dari model
        prediction_raw = model.predict(features_scaled)[0]
        
        # Format ke frontend
        frontend_prediction, fruit_type, ripeness_stage = map_kaggle_label(prediction_raw)

        probs = {}
        confidence = None
        
        if hasattr(model, 'predict_proba'):
            # Jika training dengan probability=True
            prob_array = model.predict_proba(features_scaled)[0]
            for cls_raw, p in zip(classes, prob_array):
                front_cls, _, _ = map_kaggle_label(cls_raw)
                probs[front_cls] = float(p)
            confidence = float(max(prob_array))
        else:
            # Fallback jika model SVC tidak dilatih dengan probability=True
            dec = model.decision_function(features_scaled)[0]
            dec_shifted = dec - dec.min()
            total = dec_shifted.sum()
            for cls_raw, sc in zip(classes, dec_shifted):
                front_cls, _, _ = map_kaggle_label(cls_raw)
                probs[front_cls] = float(sc / total) if total > 0 else 0.0
            confidence = float(probs.get(frontend_prediction, 0.0))

        return jsonify({
            'prediction':     frontend_prediction,
            'fruit_type':     fruit_type,
            'ripeness_stage': ripeness_stage,
            'confidence':     confidence,
            'probabilities':  probs,
            'features':       raw,
            'source':         'python_svm_model',
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("RIPE.AI — Flask API Server")
    print("=" * 60)
    print(f"Model loaded: {model_loaded}")
    app.run(host='0.0.0.0', port=5000, debug=False)