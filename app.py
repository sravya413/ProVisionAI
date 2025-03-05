from flask import Flask, render_template, request, jsonify
import os
import base64
from PIL import Image
from io import BytesIO
import pytesseract
import re
import tempfile
import requests
import logging
import time
import json
import google.generativeai as genai

# Set the path to Tesseract OCR executable - Update this if needed for your system
# On Windows: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# On Mac/Linux: Typically not needed as it's in PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure API key for Gemini
GEMINI_API_KEY = "api key"  # <-- Replace with your actual Gemini API key

# Ensure upload folder exists
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Gemini
def setup_gemini():
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        return True
    except Exception as e:
        logger.error(f"Error configuring Gemini: {e}")
        return False

def test_gemini_api_key():
    """Test if the Gemini API key is valid"""
    try:
        # Initialize Gemini
        setup_gemini()
        
        # Test with a simple model call
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content("Hello, world!")
        
        if response:
            logger.info("Gemini API key test successful !")
            return True
        else:
            logger.error("Gemini API test failed - no response")
            return False
    except Exception as e:
        logger.error(f"Gemini API key test error: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Get language preference
        language = request.form.get('language', 'en')
        logger.info(f"Processing image with language preference: {language}")
        
        # Read the image data
        img_data = file.read()
        img = Image.open(BytesIO(img_data))
        
        # Process with a smaller image size if it's too large
        max_size = (1000, 1000)
        if img.width > max_size[0] or img.height > max_size[1]:
            img.thumbnail(max_size, Image.LANCZOS)
            logger.info(f"Image resized to {img.width}x{img.height}")
        
        # Save image for OCR
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                img.save(temp_path)
            
            # Prepare image for display
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Extract text with OCR
            try:
                extracted_text = pytesseract.image_to_string(temp_path) or "No text detected"
                logger.info(f"OCR extracted text length: {len(extracted_text)}")
            except Exception as ocr_error:
                logger.error(f"OCR error: {ocr_error}")
                extracted_text = "OCR unavailable - Tesseract not configured properly"
            
            # Get AI description
            logger.info("Calling Gemini Vision API")
            api_result = call_gemini_vision_api(temp_path, extracted_text)
            
            # Process API result
            if api_result and "text" in api_result:
                annotation = api_result["text"]
                logger.info(f"Got API annotation of length: {len(annotation)}")
            else:
                error_msg = api_result.get("error", "Unknown error") if api_result else "Failed to get API response"
                logger.warning(f"Using fallback description because: {error_msg}")
                annotation = generate_local_description(img, img_base64, extracted_text)
            
            # Extract keywords from both OCR text and annotation
            keywords = extract_keywords(annotation, extracted_text)
            logger.info(f"Extracted keywords: {keywords}")
            
            # Create translations
            logger.info(f"Translating to {language}")
            translated_text = translate_text(extracted_text, language)
            translated_annotation = translate_text(annotation, language)
            
            # Build response
            response = {
                'image': img_base64,
                'ocr_text': {
                    'original': extracted_text,
                    'translated': translated_text
                },
                'annotation': {
                    'original': annotation,
                    'translated': translated_annotation
                },
                'keywords': keywords
            }
            
            return jsonify(response)
        
        finally:
            # Cleanup temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.error(f"Error deleting temp file: {e}")
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return jsonify({'error': f"Error processing image: {str(e)}"}), 500

def call_gemini_vision_api(image_path, ocr_text=""):
    """Call Google's Gemini 1.5 API with the image"""
    if not GEMINI_API_KEY or GEMINI_API_KEY.strip() == "YOUR_GEMINI_API_KEY_HERE":
        logger.error("Gemini API key not configured properly")
        return {"error": "API key not configured", "text": "Image analysis unavailable - Gemini API key not configured"}
    
    try:
        # Initialize Gemini
        setup_gemini()
        
        # Prepare the prompt
        prompt_text = "Analyze this image and provide a detailed description of what you see, including any objects, people, scenes, colors, and notable elements."
        if ocr_text and ocr_text != "No text detected" and len(ocr_text.strip()) > 5:
            prompt_text += f" I've detected some text in the image that says: '{ocr_text.strip()[:100]}'. Please include this in your analysis and explain the context of this text."
        
        # Create a model instance with appropriate configuration
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Load the image
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Create the image part
        image_part = {"mime_type": "image/png", "data": image_data}
        
        for attempt in range(3):  # Try 3 times
            try:
                logger.info(f"Attempt {attempt+1} to call Gemini API")
                
                # Generate content using text and image
                response = model.generate_content(
                    [prompt_text, image_part],
                    generation_config={"max_output_tokens": 800}
                )
                
                logger.info("Gemini API response received")
                
                if response:
                    content = response.text
                    logger.info(f"Got content of length: {len(content)}")
                    return {"text": content}
                
                logger.warning(f"API call failed (attempt {attempt+1}): No response")
                
                if attempt < 2:  # Don't sleep on the last attempt
                    time.sleep(2)
            
            except Exception as e:
                logger.warning(f"Gemini API exception on attempt {attempt+1}: {e}")
                if attempt < 2:
                    time.sleep(2)
        
        return {"error": "Failed to get a valid response from Gemini API", "text": generate_local_description(None, None, ocr_text, image_path)}
    
    except Exception as e:
        logger.error(f"Error in Gemini API call: {e}", exc_info=True)
        return {"error": str(e), "text": "Error analyzing image"}

def generate_local_description(img, img_base64=None, ocr_text="", image_path=None):
    """Generate a more detailed description locally when API fails"""
    try:
        # If we have OCR text, use it in the description
        if ocr_text and ocr_text.strip() and ocr_text != "No text detected":
            return f"This image contains text that reads: '{ocr_text.strip()[:150]}...'. To get a more detailed analysis, please configure your Gemini API key."
        
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path)
        elif not img and img_base64:
            # Try to recreate the image from base64
            img_data = base64.b64decode(img_base64)
            img = Image.open(BytesIO(img_data))
        
        if not img:
            return "This appears to be an image, but details cannot be determined. To get a detailed analysis, please configure your Gemini API key."
            
        width, height = img.size
        colors = analyze_colors(img)
        
        description = f"This is an image of size {width}x{height} pixels. "
        description += f"The dominant colors appear to be {', '.join(colors[:3])}. "
        
        # Add more details based on image analysis
        brightness = analyze_brightness(img)
        description += f"The image has {brightness} brightness overall. "
        
        # Add instruction for getting better results
        description += "For detailed AI analysis, please configure your Gemini API key in the app.py file."
        
        return description
    except Exception as e:
        logger.error(f"Error generating local description: {e}")
        return "This appears to be an image, but details cannot be determined. For better results, please configure your Gemini API key."

def analyze_colors(img):
    """Simple color analysis"""
    # Convert to small image for faster processing
    small_img = img.resize((50, 50))
    
    # Convert to RGB if needed
    if small_img.mode != 'RGB':
        small_img = small_img.convert('RGB')
    
    # Get color data
    colors = small_img.getcolors(2500)  # Get all colors
    
    # Sort by count
    if colors:
        colors.sort(reverse=True, key=lambda x: x[0])
        
        # Map RGB values to color names (very simplified)
        color_names = []
        for count, (r, g, b) in colors[:5]:  # Top 5 colors
            if r > 200 and g > 200 and b > 200:
                color_names.append("white")
            elif r < 50 and g < 50 and b < 50:
                color_names.append("black")
            elif r > 200 and g < 100 and b < 100:
                color_names.append("red")
            elif r < 100 and g > 200 and b < 100:
                color_names.append("green")
            elif r < 100 and g < 100 and b > 200:
                color_names.append("blue")
            elif r > 200 and g > 200 and b < 100:
                color_names.append("yellow")
            else:
                color_names.append("mixed tone")
        
        # Remove duplicates while preserving order
        unique_colors = []
        for color in color_names:
            if color not in unique_colors:
                unique_colors.append(color)
        
        return unique_colors
    
    return ["unknown"]

def analyze_brightness(img):
    """Analyze image brightness"""
    # Convert to grayscale and get pixel data
    gray_img = img.convert('L')
    pixels = list(gray_img.getdata())
    avg_brightness = sum(pixels) / len(pixels)
    
    # Categorize brightness
    if avg_brightness < 50:
        return "low"
    elif avg_brightness < 150:
        return "medium"
    else:
        return "high"

def extract_keywords(text, ocr_text=""):
    """Extract keywords from text with improved fallback"""
    # If text is empty or indicates API is not configured, extract from OCR text
    if not text or "API key not configured" in text or len(text) < 10:
        if ocr_text and len(ocr_text) > 10:
            text = ocr_text
        else:
            # Generate some basic keywords if we don't have meaningful text
            return ["image", "analysis", "visual", "content"]
    
    # Convert to lowercase and split into words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Filter out common stopwords
    stopwords = {'the', 'and', 'is', 'in', 'to', 'a', 'of', 'for', 'that', 'with', 'as', 'this', 
                'from', 'are', 'on', 'it', 'an', 'was', 'has', 'be', 'image', 'appears', 'shows', 
                'contains', 'depicting', 'would', 'here', 'appear', 'translation'}
    filtered_words = [word for word in words if word not in stopwords]
    
    # Count word frequency
    word_counts = {}
    for word in filtered_words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    
    # Sort by frequency and return top keywords
    keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # If we have fewer than 5 keywords, add some generic ones
    result = [k for k, v in keywords[:10]]  # Return top 10 keywords
    
    if len(result) < 5:
        generic_keywords = ["image", "visual", "content", "analysis", "media"]
        for keyword in generic_keywords:
            if keyword not in result:
                result.append(keyword)
                if len(result) >= 10:
                    break
    
    return result

def translate_text(text, target_lang):
    """Translation function using Google Translate API"""
    if not text or text.strip() == "" or target_lang == 'en':
        return text
    
    try:
        # Use Google Translate API (free tier)
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": "auto",
            "tl": target_lang,
            "dt": "t",
            "q": text
        }
        
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            # Parse the response
            result = response.json()
            if result and isinstance(result, list) and len(result) > 0:
                translated_parts = []
                # Extract all translated parts
                for part in result[0]:
                    if part and isinstance(part, list) and len(part) > 0:
                        translated_parts.append(part[0])
                
                if translated_parts:
                    return "".join(translated_parts)
        
        # Fallback if API call fails
        logger.warning(f"Translation API call failed. Status: {response.status_code}")
        return f"[Translation to {target_lang} failed] {text[:100]}..."
    
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return f"[Translation error] {text[:100]}..."


if __name__ == '__main__':
    print("\n" + "="*80)
    print("ProVisionAI Image Analysis App with Google Gemini 1.5")
    print("="*80)
    
    # Modified messages to be more helpful
    if not GEMINI_API_KEY or GEMINI_API_KEY.strip() == "api key":
        print("\n⚠ WARNING: You need to configure a valid Gemini API key")
        print("⚠ The app will run but image analysis will use local fallback instead of Gemini API")
    else:
        print("\nTesting Gemini API key...")
        if test_gemini_api_key():
            print("API key is valid! Gemini 1.5 API should work properly.")
        else:
            print(" API key test failed. Check logs for details.")
            print(" Your API key appears to be invalid or expired. The app will run but image analysis will use local fallback.")
            print("Please check your API key in the GEMINI_API_KEY variable.")
    
    # Check if Tesseract OCR is installed
    try:
        pytesseract.get_tesseract_version()
        print("\n✓ Tesseract OCR is properly configured")
    except Exception as e:
        print(f"\n WARNING: Tesseract OCR might not be properly installed: {e}")
        print(" OCR functionality might not work correctly")
    
    print("\n➡ Starting Flask application on http://127.0.0.1:5000")
    print("="*80 + "\n")
    
    app.run(debug=True)
