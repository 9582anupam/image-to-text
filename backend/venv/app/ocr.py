from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import numpy as np
from PIL import Image
import io

# Load the OCR model
model = ocr_predictor(pretrained=True)

# Function to extract and concatenate useful text from the image
def extract_text_from_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    try:
        # Use Doctr to read the image
        img_bytes = img_bytes.getvalue()
        document = DocumentFile.from_images([img_bytes])
        result = model(document)

        useful_text = []
        for page_num, page in enumerate(result.pages, start=1):
            page_text = []
            for block in page.blocks:
                for line in block.lines:
                    line_text = " ".join([word.value for word in line.words])
                    page_text.append(line_text)
            page_text = "\n".join(page_text)
            useful_text.append(f"Page {page_num}:\n\n{page_text}\n")

        final_text = "\n".join(useful_text)
        return final_text

    except Exception as e:
        return f"Error extracting text: {str(e)}"
