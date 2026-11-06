from rapidocr_onnxruntime import RapidOCR
from PIL import Image
import numpy as np

ocr = RapidOCR()
img = Image.open('processed_data/cord/images/cord_test_6.png').convert('RGB')
res, _ = ocr(np.array(img))
print(f'Total detected lines: {len(res)}')
for i, (poly, text, conf) in enumerate(res):
    print(f'{i:2d}: {conf:.3f} | text=\"{text}\" | poly={poly}')
