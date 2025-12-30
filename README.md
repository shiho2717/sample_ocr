# sample_ocr

画面をキャプチャし、任意のROIから数字をOCRで取得するサンプル。

## Environment
- Python 3.x
- Tesseract OCR

## Setup (Mac)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install tesseract