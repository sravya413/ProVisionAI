# ProVisionAI - Image Annotation with Gemini Vision

ProVisionAI is a comprehensive image annotation platform that leverages Google's powerful Gemini Vision Pro model to provide rich descriptive captions, OCR text extraction, multi-language support, and SEO keyword extraction.

## Features

- **Image Upload**: Easily upload images via drag-and-drop or file browser
- **AI-Powered Annotation**: Generate detailed descriptions of images using Gemini Vision Pro
- **OCR Text Extraction**: Extract text from images using pytesseract
- **Multi-language Support**: Translate annotations and extracted text to English, Hindi, and Telugu
- **SEO Optimization**: Automatically extract keywords from annotations for search engine optimization
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Tesseract OCR installed on your system
- Openai API key

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/provisionai.git
   cd provisionai
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install Tesseract OCR:
   - **Windows**: Download and install from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
   - **Mac**: `brew install tesseract`
   - **Linux**: `sudo apt install tesseract-ocr`

4. Get a Gemini API key:
   - Visit [openai]
   - Create an account and generate an API key
   - Replace `Your api key` in app.py with your actual API key

### Running the Application

1. Start the Flask server:
   ```
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

## Project Structure

- `app.py`: Main Flask application with API endpoints and backend logic
- `templates/index.html`: Frontend HTML/CSS/JS for the user interface
- `static/uploads/`: Directory for storing uploaded images

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **AI Services**:
  - Google Gemini Vision Pro API for image analysis
  - Tesseract OCR for text extraction
  - Google Translate API for language translation

## License

This project is licensed under the MIT License - see the LICENSE file for details.
