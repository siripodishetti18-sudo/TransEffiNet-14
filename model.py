import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from datetime import datetime

CLASSES = [
    'Atelectasis','Cardiomegaly','Effusion','Infiltration',
    'Mass','Nodule','Pneumonia','Pneumothorax','Consolidation',
    'Edema','Emphysema','Fibrosis','Pleural_Thickening','Hernia'
]

REGION_HINTS = {
    "Atelectasis":        "Lower lobes / basal regions",
    "Cardiomegaly":       "Cardiac silhouette / mediastinum",
    "Effusion":           "Costophrenic angles / pleural space",
    "Infiltration":       "Bilateral lung fields",
    "Mass":               "Focal lung parenchyma",
    "Nodule":             "Peripheral lung zones",
    "Pneumonia":          "Consolidation zones / air bronchograms",
    "Pneumothorax":       "Pleural apex / lung margins",
    "Consolidation":      "Lobar / segmental regions",
    "Edema":              "Perihilar / bilateral fields",
    "Emphysema":          "Upper lobes / hyperinflation zones",
    "Fibrosis":           "Basal subpleural regions",
    "Pleural_Thickening": "Pleural margins / costophrenic angles",
    "Hernia":             "Diaphragm / retrocardiac region"
}

IMG_SIZE  = 224
THRESHOLD = 0.35
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ── Model Architecture ──
class HybridEfficientNetTransformer(nn.Module):
    def __init__(self, num_classes=14, dropout=0.3):
        super().__init__()
        self.cnn = timm.create_model(
            "efficientnet_b4", pretrained=False, features_only=True)
        with torch.no_grad():
            dummy     = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
            feat_out  = self.cnn(dummy)[-1]
            actual_ch = feat_out.shape[1]
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(actual_ch, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.GELU()
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=1024,
            dropout=dropout, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.dropout     = nn.Dropout(dropout)
        self.classifier  = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.cnn(x)[-1]
        features = self.channel_reduce(features)
        B, C, H, W = features.shape
        tokens   = features.flatten(2).permute(0, 2, 1)
        tokens   = self.transformer(tokens)
        out      = tokens.mean(dim=1)
        out      = self.dropout(out)
        return self.classifier(out)


def load_model(ckpt_path="best_hybrid_model.pth"):
    model = HybridEfficientNetTransformer(num_classes=14).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    clean = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(clean)
    model.eval()
    print(f"✓ Model loaded on {device}")
    return model


# ── Grad-CAM ──
class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.activations = None
        self.gradients   = None
        # Pick safe non-SE conv layer at ~65% depth
        valid_convs = [
            (n, m) for n, m in model.cnn.named_modules()
            if isinstance(m, nn.Conv2d)
            and ".se."        not in n
            and "conv_reduce" not in n
            and "conv_expand" not in n
        ]
        _, target = valid_convs[int(len(valid_convs) * 0.65)]
        self._fh = target.register_forward_hook(self._save_act)
        self._bh = target.register_full_backward_hook(self._save_grad)

    def _save_act(self, m, i, o):
        self.activations = o.detach().clone()

    def _save_grad(self, m, gi, go):
        self.gradients = go[0].detach().clone()

    def generate(self, tensor, class_idx):
        self.model.zero_grad(set_to_none=True)
        out = self.model(tensor)
        out[0, class_idx].backward()
        w   = torch.clamp(self.gradients, min=0).mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.activations).sum(dim=1)).squeeze()
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, out

    def remove(self):
        self._fh.remove()
        self._bh.remove()


# ── Preprocess uploaded PIL image ──
def preprocess(pil_image):
    img_gray  = np.array(pil_image.convert("L"))
    img_gray  = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)
    img_rgb   = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
    tensor    = val_transform(Image.fromarray(img_rgb)).unsqueeze(0).to(device)
    return img_gray, tensor


# ── Severity helpers ──
def get_severity(prob):
    if prob >= 0.75:        return "SEVERE",   "🔴"
    elif prob >= 0.50:      return "MODERATE",  "🟠"
    elif prob >= THRESHOLD: return "MILD",      "🟡"
    else:                   return "NORMAL",    "🟢"


# ── Build text severity report ──
def build_report(probs):
    now      = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    detected = []
    for i, disease in enumerate(CLASSES):
        sev, icon = get_severity(probs[i])
        if sev != "NORMAL":
            detected.append((disease, probs[i], sev, icon))
    detected = sorted(detected, key=lambda x: x[1], reverse=True)

    if not detected:
        overall = "✅  NORMAL — No significant findings detected"
    elif detected[0][2] == "SEVERE":
        overall = "🔴  HIGH RISK — Immediate radiologist review recommended"
    elif detected[0][2] == "MODERATE":
        overall = "🟠  MODERATE RISK — Follow-up recommended"
    else:
        overall = "🟡  LOW RISK — Monitor and recheck if symptoms persist"

    lines = []
    lines.append("=" * 58)
    lines.append("     AI CHEST X-RAY SEVERITY REPORT")
    lines.append(f"     Generated : {now}")
    lines.append(f"     Model     : EfficientNet-B4 + Transformer")
    lines.append(f"     Dataset   : NIH ChestX-ray14 (112,000+ images)")
    lines.append("=" * 58)
    lines.append(f"\n  OVERALL ASSESSMENT:\n  {overall}\n")

    if detected:
        lines.append("  DETECTED CONDITIONS:")
        lines.append(f"  {'Disease':<22} {'Prob':>6}   {'Severity':<10}  Region")
        lines.append("  " + "-" * 55)
        for disease, prob, sev, icon in detected:
            lines.append(
                f"  {icon} {disease:<20} {prob:>5.3f}   {sev:<10}  {REGION_HINTS[disease]}"
            )
        lines.append("")
    else:
        lines.append("  No pathological findings above threshold.\n")

    lines.append("  ALL DISEASE SCORES:")
    lines.append(f"  {'Disease':<22} {'Prob':>6}   Status")
    lines.append("  " + "-" * 40)
    for i, disease in enumerate(CLASSES):
        sev, icon = get_severity(probs[i])
        lines.append(f"  {icon} {disease:<20} {probs[i]:>5.3f}   {sev}")

    lines.append("\n" + "=" * 58)
    lines.append("  ⚠  AI-assisted tool. NOT for clinical diagnosis.")
    lines.append("     Must be reviewed by a licensed radiologist.")
    lines.append("=" * 58)
    return "\n".join(lines)


# ── Main inference: returns everything needed by app.py ──
def run_inference(pil_image, model):
    img_gray, tensor = preprocess(pil_image)

    # Forward pass for probabilities
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits).cpu().numpy()[0]

    # Grad-CAM on top predicted disease
    top_idx = int(np.argmax(probs))
    gc      = GradCAM(model)
    cam, _  = gc.generate(tensor, top_idx)
    gc.remove()

    # Upsample + smooth
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    cam = cv2.bilateralFilter(cam.astype(np.float32), 9, 0.1, 5)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam[cam < np.percentile(cam, 60)] = 0

    # Build overlay
    heatmap     = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    xray_rgb    = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    overlay     = cv2.addWeighted(xray_rgb, 0.5, heatmap_rgb, 0.5, 0)

    report = build_report(probs)

    return (
        Image.fromarray(img_gray),           # original enhanced
        Image.fromarray(overlay),            # grad-cam overlay
        CLASSES[top_idx],                    # top disease name
        probs,                               # all 14 probabilities
        report                               # text report
    )
