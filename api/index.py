from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from PIL import Image
import io
import torch
from transformers import pipeline

app = FastAPI(title="TomatoAI - Leaf Disease Detection API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ─── Load the real ML model (ViT fine-tuned on PlantVillage, 99.67% accuracy) ───
print("Loading ViT model for tomato leaf disease classification...")
classifier = pipeline(
    "image-classification",
    model="wellCh4n/tomato-leaf-disease-classification-vit",
    device="cpu",
    top_k=10,
)
print("Model loaded successfully!")

# ─── Label mapping from model labels to our display labels ───
LABEL_MAP = {
    "A healthy tomato leaf": "Healthy",
    "A tomato leaf with Bacterial Spot": "Bacterial Spot",
    "A tomato leaf with Early Blight": "Early Blight",
    "A tomato leaf with Late Blight": "Late Blight",
    "A tomato leaf with Leaf Mold": "Leaf Mold",
    "A tomato leaf with Septoria Leaf Spot": "Septoria Leaf Spot",
    "A tomato leaf with Spider Mites Two-spotted Spider Mite": "Spider Mites",
    "A tomato leaf with Target Spot": "Target Spot",
    "A tomato leaf with Tomato Mosaic Virus": "Mosaic Virus",
    "A tomato leaf with Tomato Yellow Leaf Curl Virus": "Yellow Leaf Curl Virus",
}

CLASS_LABELS = [
    'Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold',
    'Septoria Leaf Spot', 'Spider Mites', 'Target Spot',
    'Yellow Leaf Curl Virus', 'Mosaic Virus', 'Healthy',
]

DISEASE_INFO = {
    'Bacterial Spot': {'severity': 'High', 'description': 'Caused by Xanthomonas vesicatoria. Small dark spots with yellow halos scattered across the leaf surface, eventually causing defoliation.', 'treatment': 'Apply copper-based bactericides. Remove infected leaves. Avoid overhead watering. Use disease-free seeds.'},
    'Early Blight': {'severity': 'Medium', 'description': 'Caused by Alternaria solani. Concentric ring patterns (target spots) on older leaves, progressing upward through the plant.', 'treatment': 'Apply fungicides (chlorothalonil/mancozeb). Practice crop rotation. Remove infected debris. Ensure adequate spacing.'},
    'Late Blight': {'severity': 'Critical', 'description': 'Caused by Phytophthora infestans. Large dark water-soaked lesions on leaves and stems, often with white mold underneath.', 'treatment': 'Apply fungicides immediately (metalaxyl). Remove and destroy infected plants. Avoid overhead irrigation.'},
    'Leaf Mold': {'severity': 'Medium', 'description': 'Caused by Passalora fulva. Pale greenish-yellow spots on upper leaf surfaces with olive-green velvety growth underneath.', 'treatment': 'Improve air circulation, reduce humidity. Apply fungicides if severe. Remove affected leaves.'},
    'Septoria Leaf Spot': {'severity': 'Medium', 'description': 'Caused by Septoria lycopersici. Small circular spots with dark borders and gray centers on lower leaves.', 'treatment': 'Apply fungicides (chlorothalonil). Remove infected leaves. Mulch around plants. Practice crop rotation.'},
    'Spider Mites': {'severity': 'Medium', 'description': 'Tetranychus urticae causes tiny yellow/white speckles. Heavy infestations produce fine webbing and leaf bronzing.', 'treatment': 'Spray insecticidal soap or neem oil. Increase humidity. Introduce predatory mites.'},
    'Target Spot': {'severity': 'Medium', 'description': 'Caused by Corynespora cassiicola. Brown spots with concentric rings and yellow halos appear on leaves.', 'treatment': 'Apply fungicides (azoxystrobin). Improve air circulation. Remove crop debris. Practice crop rotation.'},
    'Yellow Leaf Curl Virus': {'severity': 'Critical', 'description': 'TYLCV transmitted by whiteflies. Leaves curl upward, become yellow, plants are severely stunted with reduced fruit.', 'treatment': 'Control whiteflies with insecticides/sticky traps. Remove infected plants. Use TYLCV-resistant varieties.'},
    'Mosaic Virus': {'severity': 'High', 'description': 'Tomato mosaic virus causes mottled light/dark green mosaic patterns on leaves. Leaves may be distorted.', 'treatment': 'Remove and destroy infected plants. Disinfect tools. Use resistant varieties. Avoid tobacco near plants.'},
    'Healthy': {'severity': 'None', 'description': 'The leaf appears healthy with no visible signs of disease, pest damage, or nutrient deficiency.', 'treatment': 'Continue regular monitoring, proper watering, and balanced fertilization for best results.'},
}


# ─── Leaf Validation using OpenCV ──────────────────────────────────

def validate_leaf_image(image_bytes: bytes):
    """
    Validate that the uploaded image is actually a plant/tomato leaf.
    Rejects human photos, objects, blank images, etc.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return False, "Could not process the image. Please upload a valid image file."

    img = cv2.resize(img_bgr, (256, 256))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    total = 256 * 256

    # 1. Check for skin tones (reject human photos)
    # Skin in HSV: Hue 0-25, Sat 20-180, Val 50-255
    skin_lower1 = np.array([0, 20, 50])
    skin_upper1 = np.array([25, 180, 255])
    skin_lower2 = np.array([160, 20, 50])
    skin_upper2 = np.array([180, 180, 255])
    skin_mask1 = cv2.inRange(hsv, skin_lower1, skin_upper1)
    skin_mask2 = cv2.inRange(hsv, skin_lower2, skin_upper2)
    skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
    skin_ratio = float(np.sum(skin_mask > 0) / total)

    # 2. Check for plant-like colors (green, yellow-green, brown-green)
    green_mask = cv2.inRange(hsv, (25, 30, 30), (90, 255, 255))
    green_ratio = float(np.sum(green_mask > 0) / total)

    # 3. Check for white/blank images
    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
    white_ratio = float(np.sum(white_mask > 0) / total)

    # 4. Color variance (reject uniform/solid color images)
    color_var = float(np.std(gray))

    # 5. Edge density (reject very flat images)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = float(np.sum(edges > 0) / total) * 100

    # ── Decision rules ──
    # Too much skin tone → likely a person/face
    if skin_ratio > 0.35 and green_ratio < 0.15:
        return False, "This appears to be a photo of a person, not a plant leaf. Please upload a close-up photo of a tomato leaf."

    # Too much skin without enough green
    if skin_ratio > 0.25 and green_ratio < skin_ratio * 0.5:
        return False, "This doesn't appear to be a leaf image. The image contains too many non-plant colors. Please upload a photo of a tomato leaf."

    # Too white/blank
    if white_ratio > 0.6:
        return False, "This appears to be a blank or white image. Please upload a close-up photo of a tomato leaf."

    # Not enough plant-like color
    if green_ratio < 0.05 and color_var < 30:
        return False, "No plant-like colors detected in the image. Please upload a photo of a tomato leaf."

    # Too uniform (solid color)
    if color_var < 12:
        return False, "The image color is too uniform. Please upload a real photo of a tomato leaf."

    # Very low green + low edge = likely not a plant
    if green_ratio < 0.08 and edge_ratio < 1.5:
        return False, "This doesn't appear to be a leaf image. Please upload a clear, close-up photo of a tomato leaf."

    return True, "OK"


# ─── API Endpoints ─────────────────────────────────────────────────

@app.get("/api/health")
@app.get("/health")
def health_check():
    return {
        "status": "online",
        "model": "ViT (google/vit-base-patch16-224) fine-tuned on PlantVillage",
        "accuracy": "99.67%",
        "classes": 10,
        "version": "3.0.0",
    }


@app.post("/api/predict")
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # 1. Validate it's a real image
    try:
        pil_image = Image.open(io.BytesIO(contents))
        pil_image.verify()
        # Re-open after verify (verify consumes the stream)
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return {"success": False, "message": "Invalid or corrupted image file. Please upload a valid JPG or PNG.", "is_leaf": False}

    # 2. Validate it's a leaf (not a person, object, etc.)
    is_valid, reason = validate_leaf_image(contents)
    if not is_valid:
        return {"success": False, "message": reason, "is_leaf": False}

    # 3. Run the real ML model
    try:
        results = classifier(pil_image)
    except Exception as e:
        return {"success": False, "message": f"Model inference failed: {str(e)}", "is_leaf": False}

    # 4. Parse results
    top_result = results[0]
    raw_label = top_result["label"]
    confidence = top_result["score"] * 100

    # Map model label to our display label
    predicted_class = LABEL_MAP.get(raw_label, raw_label)
    info = DISEASE_INFO.get(predicted_class, {})

    # Build probability dict for all classes
    prob_dict = {}
    for r in results:
        display_label = LABEL_MAP.get(r["label"], r["label"])
        prob_dict[display_label] = round(r["score"], 6)

    # Ensure all classes are present
    for label in CLASS_LABELS:
        if label not in prob_dict:
            prob_dict[label] = 0.0

    return {
        "success": True,
        "diagnosis": predicted_class,
        "confidence": round(confidence, 2),
        "probabilities": prob_dict,
        "severity": info.get("severity", "Unknown"),
        "description": info.get("description", ""),
        "treatment": info.get("treatment", ""),
        "is_leaf": True,
    }
