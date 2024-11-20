import google.generativeai as genai
import os
import base64
from PIL import Image
from io import BytesIO

# api
API_KEY = os.environ['GEMINI_AI_API_KEY']
genai.configure(api_key=API_KEY)

def encode_image_to_base64(image_path):
   
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def prep_image(image_path, use_base64=False):
   
    if use_base64:
        # base64 conversion
        base64_image = encode_image_to_base64(image_path)
        # mime
        sample_file = {
            'data': base64_image,
            'mime_type': 'image/jpeg',  
            'display_name': "Diagram"
        }
        print(f"Encoded image '{sample_file['display_name']}' as base64")
        return sample_file
    else:
        # direct upload
        sample_file = genai.upload_file(path=image_path,
                                    display_name="Diagram")
        print(f"Uploaded file '{sample_file.display_name}' as: {sample_file.uri}")
        file = genai.get_file(name=sample_file.name)
        print(f"Retrieved file '{file.display_name}' as: {sample_file.uri}")
        return sample_file

def extract_text_from_image(image_data, prompt, use_base64=False):
    
    try:
        # mdl
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
            # direct upload
            response = model.generate_content([image_data, prompt])
            
        return response.text
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        return None


def main():
    image_path = 'ocr.png'
    prompt = "Extract the text in the image "
    
    # base64 flagz
    sample_file = prep_image(image_path, use_base64=True)
    text = extract_text_from_image(sample_file, prompt, use_base64=True)
    
    if text:
        print("Extracted Text:")
        print(text)
    else:
        print("Failed to extract text from the image.")

if __name__ == "__main__":
    main()