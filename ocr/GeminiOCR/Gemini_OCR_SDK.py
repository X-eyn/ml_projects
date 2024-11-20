import google.generativeai as genai
import os
import base64
from PIL import Image
from io import BytesIO
from pdf2image import convert_from_path  # You'll need to install pdf2image
import tempfile

API_KEY = os.environ['GEMINI_AI_API_KEY']
genai.configure(api_key=API_KEY)

def encode_image_to_base64(image):
    """Encode PIL Image object to base64"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def prep_image(image, use_base64=False):
    """Prepare either PIL Image object or image path for Gemini API"""
    if use_base64:
        # Handle PIL Image object
        if isinstance(image, Image.Image):
            base64_image = encode_image_to_base64(image)
        # Handle image path
        else:
            with open(image, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        sample_file = {
            'data': base64_image,
            'mime_type': 'image/jpeg',
            'display_name': "Document Page"
        }
        return sample_file
    else:
        # For direct upload, we need to save PIL Image temporarily if it's not a path
        if isinstance(image, Image.Image):
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                image.save(tmp, format='JPEG')
                tmp_path = tmp.name
            
            sample_file = genai.upload_file(path=tmp_path,
                                          display_name="Document Page")
            os.unlink(tmp_path)  # Clean up temporary file
        else:
            sample_file = genai.upload_file(path=image,
                                          display_name="Document Page")
        
        return sample_file

def extract_text_from_image(image_data, prompt, use_base64=False):
    """Extract text from image using Gemini"""
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        
        if use_base64:
            content = {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": image_data['mime_type'],
                            "data": image_data['data']
                        }
                    },
                    {
                        "text": prompt
                    }
                ]
            }
            response = model.generate_content(content)
        else:
            response = model.generate_content([image_data, prompt])
        
        return response.text
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        return None

def process_pdf(pdf_path, prompt="Extract the text in the image", use_base64=True):
    """Process a PDF file page by page"""
    try:
        # Convert PDF to images
        images = convert_from_path(pdf_path)
        
        all_text = []
        for i, image in enumerate(images):
            print(f"Processing page {i+1}/{len(images)}")
            
            # Prepare image for Gemini
            sample_file = prep_image(image, use_base64=use_base64)
            
            # Extract text
            text = extract_text_from_image(sample_file, prompt, use_base64=use_base64)
            if text:
                all_text.append(f"=== Page {i+1} ===\n{text}")
        
        return "\n\n".join(all_text)
    
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return None

def main():
    # Can handle both PDFs and images
    document_path = r"C:\Users\Admin\Downloads\1706.03762v7.pdf"  # or 'ocr.png'
    prompt = "Extract the text in the image"
    
    if document_path.lower().endswith('.pdf'):
        text = process_pdf(document_path, prompt, use_base64=True)
    else:
        sample_file = prep_image(document_path, use_base64=True)
        text = extract_text_from_image(sample_file, prompt, use_base64=True)
    
    if text:
        print("Extracted Text:")
        print(text)
    else:
        print("Failed to extract text from the document.")

if __name__ == "__main__":
    main()