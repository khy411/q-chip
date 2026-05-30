import onnxruntime as ort
import numpy as np
import cv2, glob, os, random

MODEL = "qchip_fp32.onnx"
VAL_DIR = "val_images"
OUT_DIR = "detections"
NUM_IMGS = 8
CONF_THRESH = 0.25
IMG_SIZE = 640

CLASSES = ['open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole']
COLORS = [
    (255, 80, 80),
    (80, 200, 255),
    (255, 200, 50),
    (120, 255, 120),
    (200, 100, 255),
    (255, 160, 50),
]

os.makedirs(OUT_DIR, exist_ok=True)

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name

images = glob.glob(os.path.join(VAL_DIR, "*.jpg"))
random.seed(99)
random.shuffle(images)

saved = 0
for img_path in images:
    if saved >= 5:
        break

    orig = cv2.imread(img_path)
    h, w = orig.shape[:2]

    inp = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = inp.transpose(2, 0, 1)[np.newaxis, :]

    outputs = sess.run(None, {input_name: inp})
    preds = outputs[0][0]

    drawn = orig.copy()
    detected = False

    for pred in preds:
        x1, y1, x2, y2, conf, cls_id = pred
        if conf < CONF_THRESH:
            continue
        detected = True
        cls_id = int(cls_id)

        # Scale boxes back to original image dimensions
        x1 = int(x1 / IMG_SIZE * w)
        y1 = int(y1 / IMG_SIZE * h)
        x2 = int(x2 / IMG_SIZE * w)
        y2 = int(y2 / IMG_SIZE * h)

        color = COLORS[cls_id % len(COLORS)]
        label = f"{CLASSES[cls_id]} {conf:.2f}"

        cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(drawn, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(drawn, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    if detected:
        out_path = os.path.join(OUT_DIR, os.path.basename(img_path))
        cv2.imwrite(out_path, drawn)
        saved += 1
        print(f"saved: {out_path}")

print(f"Done. {saved} Images saved to {OUT_DIR}/")