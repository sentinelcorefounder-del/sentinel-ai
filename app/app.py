from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import os
import uuid
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

device = torch.device("cpu")

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

FUNDUS_GATE_MODEL_PATH = os.path.join(BASE_DIR, "fundus_gate_model_v1.pth")
DR_MODEL_PATH = os.path.join(BASE_DIR, "model.pth")

IMG_SIZE = 224
FUNDUS_ACCEPT_THRESHOLD = 0.80
FUNDUS_REJECT_THRESHOLD = 0.20

# =========================
# LOAD MODELS
# =========================
fundus_model = models.resnet18(weights=None)
fundus_model.fc = nn.Linear(fundus_model.fc.in_features, 2)
fundus_model.load_state_dict(torch.load(FUNDUS_GATE_MODEL_PATH, map_location=device))
fundus_model.to(device)
fundus_model.eval()

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
model.load_state_dict(torch.load(DR_MODEL_PATH, map_location=device))
model.to(device)
model.eval()

target_layer = model.features[-1]

# =========================
# TRANSFORMS
# =========================
fundus_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# FUNDUS GATE
# =========================
def predict_fundus_gate(img):
    img_tensor = fundus_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = fundus_model(img_tensor)
        probs = F.softmax(output, dim=1)

    fundus_prob = probs[0][0].item()

    if fundus_prob >= FUNDUS_ACCEPT_THRESHOLD:
        decision = "accept_fundus"
    elif fundus_prob <= FUNDUS_REJECT_THRESHOLD:
        decision = "reject_non_fundus"
    else:
        decision = "uncertain"

    return fundus_prob, decision

# =========================
# PREPROCESS (SAFE)
# =========================
def preprocess_retina(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    h, w, _ = img.shape
    min_dim = min(h, w)

    start_x = w // 2 - min_dim // 2
    start_y = h // 2 - min_dim // 2

    img = img[start_y:start_y+min_dim, start_x:start_x+min_dim]
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# =========================
# ANALYZE ROUTE
# =========================
@app.route("/analyze", methods=["POST"])
def analyze():
    print("🔥 ANALYZE ROUTE HIT")

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    
    # 🔥 FILE SIZE CHECK (5MB limit)
    file.seek(0, os.SEEK_END)
    file_length = file.tell()
    file.seek(0)
    
    if file_length > 5 * 1024 * 1024:
        return jsonify({"error": "File too large (max 5MB)"}), 400

    try:
        upload_filename = f"upload_{uuid.uuid4().hex}.jpg"
        upload_path = os.path.join(STATIC_DIR, upload_filename)
        file.save(upload_path)

        img = Image.open(upload_path).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image"}), 400

    # =========================
    # FUNDUS GATE
    # =========================
    fundus_prob, decision = predict_fundus_gate(img)

    if decision == "reject_non_fundus":
        return jsonify({
            "prediction": "Invalid image",
            "message": "Please upload a retinal fundus image",
            "confidence": fundus_prob,
            "heatmap_url": None,
            "processed_image_url": None
        })

    if decision == "uncertain":
        return jsonify({
            "prediction": "Uncertain image",
            "message": "Image unclear, please re-upload",
            "confidence": fundus_prob,
            "heatmap_url": None,
            "processed_image_url": None
        })

    # =========================
    # PREPROCESS
    # =========================
    img_processed = preprocess_retina(img)

    # SAVE PROCESSED IMAGE
    processed_filename = f"processed_{uuid.uuid4().hex}.jpg"
    processed_path = os.path.join(STATIC_DIR, processed_filename)

    cv2.imwrite(processed_path, cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR))

    # =========================
    # MODEL
    # =========================
    img_tensor = transform(img_processed).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0].cpu().numpy()

    confidence = float(np.max(probs))
    severity = int(np.argmax(probs))

    # =========================
    # 🔥 REFERABLE LOGIC (FIXED)
    # =========================
    referable_score = probs[2] + probs[3] + probs[4]

    print("PROBS:", probs)
    print("REFERABLE SCORE:", referable_score)

    if referable_score > 0.35:
        prediction = "Referable DR"
    else:
        prediction = "No Referable DR"

    # =========================
    # GRAD-CAM
    # =========================
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=img_tensor)[0]

    img_np = np.array(img_processed) / 255.0
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    heatmap_filename = f"heatmap_{uuid.uuid4().hex}.jpg"
    heatmap_path = os.path.join(STATIC_DIR, heatmap_filename)

    cv2.imwrite(heatmap_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

    # =========================
    # LABELS
    # =========================
    severity_labels = [
        "No DR",
        "Mild",
        "Moderate",
        "Severe",
        "Proliferative DR"
    ]

    response = {
        "prediction": prediction,
        "confidence": confidence,
        "severity": severity,
        "severity_label": severity_labels[severity],
        "heatmap_url": f"/static/{heatmap_filename}",
        "processed_image_url": f"/static/{processed_filename}",
        "disclaimer": "This tool provides AI-assisted screening for diabetic retinopathy only and does not replace professional clinical diagnosis. The model is not trained to detect other retinal or ocular conditions (e.g. retinal detachment, glaucoma, macular degeneration). All images should be reviewed by a qualified eye care professional before clinical decisions are made."
    }

    print("🔥 RESPONSE:", response)

    return jsonify(response)

# =========================
@app.route("/")
def index():
    return "Backend running..."

# =========================
if __name__ == "__main__":
    if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
