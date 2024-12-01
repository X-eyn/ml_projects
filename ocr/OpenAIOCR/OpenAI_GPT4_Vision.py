from openai import OpenAI
import base64
import os
import json
import re

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Specific, structured prompt for JSON extraction
json_extraction_prompt = """
Return JSON document with data. Only return JSON not other text
"""

def extract_text(client, image_path):
    base64_img = f"data:image/png;base64,{encode_image(image_path)}"
    
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json_extraction_prompt},
                    {"type": "image_url", "image_url": {"url": base64_img}}
                ]
            }
        ],
        max_tokens=1500,
        temperature=0.2
    )
    
    return response.choices[0].message.content.strip()

# Usage
client = OpenAI()
image_local = r"C:\Users\Admin\Pictures\Screenshots\ocr.png"

# JSON extraction with robust parsing
try:
    extracted_json_str = extract_text(client, image_local)
    
    # Remove any code block markers or extra text
    extracted_json_str = re.sub(r'^```json\s*', '', extracted_json_str)
    extracted_json_str = re.sub(r'```$', '', extracted_json_str)
    
    # Try to parse the JSON
    parsed_data = json.loads(extracted_json_str)
    
    # Filename for JSON output
    json_filename = f"./Data/{os.path.splitext(os.path.basename(image_local))[0]}.json"
    
    # Save parsed JSON with proper formatting
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(parsed_data, f, ensure_ascii=False, indent=4)
    
    print(f"Successfully saved JSON to {json_filename}")

except json.JSONDecodeError as json_err:
    print(f"JSON Parsing Error: {json_err}")
    print("Extracted content:", extracted_json_str)
    
    # Fallback: save raw response, but clean it up
    json_filename = f"./Data/{os.path.splitext(os.path.basename(image_local))[0]}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        # Remove code block markers and try to clean up
        cleaned_content = re.sub(r'^```json\s*', '', extracted_json_str)
        cleaned_content = re.sub(r'```$', '', cleaned_content)
        f.write(cleaned_content)
    
except Exception as e:
    print(f"Unexpected error during JSON extraction: {e}")