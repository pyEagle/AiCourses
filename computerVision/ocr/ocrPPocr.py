import sys
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False, 
    use_doc_unwarping=False, 
    use_textline_orientation=False) # 文本检测+文本识别

img_file = sys.argv[1]
result = ocr.predict(img_file)
for res in result:
    rec_texts = res.get('rec_texts', '')
    rec_scores = res.get('rec_scores', 0)
    rec_boxes = res.get('rec_boxes', [])
    for t,s,b in zip(rec_texts, rec_scores, rec_boxes):
        print([t, s, b])
