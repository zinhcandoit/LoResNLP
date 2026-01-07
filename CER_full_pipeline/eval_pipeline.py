import os, time
import torch
import Levenshtein
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from paddleocr import PaddleOCR

# Use paddle-gpu 2.6.1 to get the best performance in Vietnamese OCR
PROJECT_DIR = r"C:\Users\VINH\OneDrive - VNU-HCMUS\Attachments\Desktop\P_OCR\OCR_Lab-20260106T152002Z-3-001\OCR_Lab"
GT_FILE_PATH = os.path.join(PROJECT_DIR, "rec_gt.txt")
# Model file
MODEL_BYT5_DIR = os.path.join(PROJECT_DIR, r"C:\Users\VINH\OneDrive - VNU-HCMUS\Attachments\Desktop\P_OCR\byt5_ocr_correction")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(">> Load ByT5...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_BYT5_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_BYT5_DIR).to(DEVICE)
model.eval()

print(">> Init PaddleOCR...")
ocr_engine = PaddleOCR(lang="vi", use_gpu=False, show_log=False)
def calculate_cer(pred: str, gt: str) -> float:
    pred = (pred or "").strip()
    gt = (gt or "").strip()
    if len(gt) == 0:
        return 1.0 if len(pred) > 0 else 0.0
    return Levenshtein.distance(pred, gt) / len(gt)

def extract_ocr_text(paddle_result) -> str:
    if not paddle_result:
        return ""
    lines = paddle_result[0] if isinstance(paddle_result, list) else paddle_result
    if not lines:
        return ""
    texts = []
    for item in lines:
        try:
            texts.append(item[1][0])
        except Exception as e:
            print(f"Lỗi tại dòng: {e}") # Thêm dòng này để thấy mặt mũi cái lỗi
            skipped += 1
            continue
    return " ".join(texts).strip()

def byt5_correct(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    with torch.inference_mode():
        outputs = model.generate(
            inputs.input_ids,
            max_length=256,
            num_beams=1,
            do_sample=False,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

with open(GT_FILE_PATH, "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

print("LINES READ =", len(lines))

total_cer_1 = 0.0
total_cer_2 = 0.0
processed = 0
skipped = 0

t0 = time.time()

for line in lines:
    line = line.strip()
    if not line or "\t" not in line:
        skipped += 1
        continue

    img_path, gt_text = line.split("\t", 1)
    img_path = img_path.strip()
    gt_text = gt_text.strip()

    if not os.path.isabs(img_path):
        img_path = os.path.join(PROJECT_DIR, img_path)
    img_path = os.path.normpath(img_path)

    if not os.path.exists(img_path) or not gt_text:
        skipped += 1
        continue

    try:
        ocr_pred = extract_ocr_text(ocr_engine.ocr(img_path))
        cer1 = calculate_cer(ocr_pred, gt_text)

        byt5_pred = byt5_correct(ocr_pred)
        cer2 = calculate_cer(byt5_pred, gt_text)
    except Exception as e:
        print(f"Lỗi tại dòng: {e}")
        skipped += 1
        continue

    processed += 1
    total_cer_1 += cer1
    total_cer_2 += cer2

    if processed % 50 == 0:
        print(f"Processed {processed}/100 | elapsed {time.time()-t0:.1f}s")

avg1 = total_cer_1 / processed if processed else 0.0
avg2 = total_cer_2 / processed if processed else 0.0

print("\n==================== TỔNG KẾT ====================")
print("Processed:", processed, "| Skipped:", skipped)
print(f"OCR CER avg : {avg1:.4f} ({avg1*100:.2f}%)")
print(f"ByT5 CER avg: {avg2:.4f} ({avg2*100:.2f}%)")
print(f"Time: {time.time()-t0:.1f}s")
