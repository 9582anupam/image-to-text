import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
import io
import numpy as np
from PIL import Image
from preprocess import cropImage, deskew
from ocr import extract_text_from_image
from correct import correct_text

app = Flask(__name__)
CORS(app, resources={r"/process_pdf": {"origins": "http://localhost:3000"}})

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        pdf_bytes = file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            images.append(img)

        if not images:
            return jsonify({'error': 'No images found in PDF'}), 400

        results = []
        for img in images:
            cv_image = np.array(img)
            cropped_image = cropImage(cv_image)
            deskewed_image = deskew(cropped_image)

            # Perform OCR on the preprocessed image
            extracted_text = extract_text_from_image(deskewed_image)

            # Correct the extracted text
            corrected_text = correct_text(extracted_text)

            results.append(corrected_text)

        result = {'text': "\n".join(results)}
        return jsonify(result)

    except Exception as e:
        logger.error('Error processing PDF', exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
