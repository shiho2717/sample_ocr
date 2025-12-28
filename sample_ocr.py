import os
import time
import mss
import cv2
import numpy as np
import pytesseract

def capture_monitor(monitor_index: int = 1) -> np.ndarray:
    """指定モニターを1枚キャプチャ（BGRAのnumpy配列）"""
    with mss.mss() as sct:
        monitor = sct.monitors[monitor_index]
        img = np.array(sct.grab(monitor))
    return img

def preprocess_for_digits(bgra_img: np.ndarray, threshold: int = 160) -> np.ndarray:
    """OCRしやすい前処理（白黒→二値化）"""
    bgr = cv2.cvtColor(bgra_img, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return bin_img

def ocr_digits(bin_img: np.ndarray):
    """数字のみOCR"""
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    raw = pytesseract.image_to_string(bin_img, config=config).strip()
    digits = "".join(ch for ch in raw if ch.isdigit())
    return (int(digits) if digits else None), raw

def main():
    os.makedirs("debug", exist_ok=True)

    # 1) 画面キャプチャ
    img = capture_monitor(monitor_index=1)
    ts = int(time.time())
    cv2.imwrite(f"debug/capture_{ts}.png", img)

    # 2) ROIをマウスで選択
    bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    r = cv2.selectROI(
        "Drag ROI then Enter (ESC to cancel)",
        bgr,
        fromCenter=False,
        showCrosshair=True
    )
    cv2.destroyAllWindows()

    x, y, w, h = map(int, r)
    if w == 0 or h == 0:
        print("ROI cancelled.")
        return

    roi = img[y:y+h, x:x+w]
    cv2.imwrite("debug/roi.png", roi)
    print(f"ROI: x={x}, y={y}, w={w}, h={h}")

    # 3) 前処理 → OCR
    bin_img = preprocess_for_digits(roi, threshold=160)
    cv2.imwrite("debug/roi_bin.png", bin_img)

    value, raw = ocr_digits(bin_img)
    print(f"OCR raw: {raw!r}")
    print(f"Detected number: {value}")

if __name__ == "__main__":
    main()
