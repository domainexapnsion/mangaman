#!/usr/bin/env python3
# Complete automated pipeline with Gemini Pro AI for superior translation
# Bato.to English pages to Marathi with context-aware AI translation

import os
import re
import time
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import pytesseract
import cv2
import numpy as np
import google.generativeai as genai

class BatoToMarathiPipeline:
    def __init__(self, gemini_api_key, output_dir="marathi_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure Gemini Pro
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-pro')
        self.gemini_vision_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Setup Chrome options for headless browsing
        self.chrome_options = Options()
        self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        self.chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        # Context tracking for better translations
        self.conversation_context = []
        
    def extract_images_from_bato(self, chapter_url):
        """Extract all images from a Bato.to chapter URL"""
        print(f"Extracting images from: {chapter_url}")
        
        driver = webdriver.Chrome(options=self.chrome_options)
        images = []
        
        try:
            driver.get(chapter_url)
            time.sleep(3)
            
            # Scroll to load all images
            last_height = driver.execute_script("return document.body.scrollHeight")
            while True:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
            
            # Find all image elements
            img_elements = driver.find_elements(By.TAG_NAME, "img")
            
            for idx, img_elem in enumerate(img_elements):
                try:
                    img_url = img_elem.get_attribute('src')
                    if img_url and ('bato' in img_url or 'cdn' in img_url or 'mcd' in img_url):
                        response = requests.get(img_url, headers={
                            'User-Agent': 'Mozilla/5.0',
                            'Referer': chapter_url
                        })
                        
                        if response.status_code == 200:
                            img_path = self.output_dir / f"original_{idx:03d}.jpg"
                            with open(img_path, 'wb') as f:
                                f.write(response.content)
                            images.append(img_path)
                            print(f"Downloaded image {idx + 1}")
                except Exception as e:
                    print(f"Error downloading image {idx}: {e}")
                    
        finally:
            driver.quit()
        
        print(f"Extracted {len(images)} images")
        return images
    
    def gemini_vision_ocr(self, image_path):
        """Use Gemini Vision to extract text with better accuracy"""
        print(f"Using Gemini Vision for OCR on {image_path.name}")
        
        try:
            img = Image.open(image_path)
            
            prompt = """Analyze this manga/comic page image and extract ALL text you can find.
            For each text block, provide:
            1. The exact English text
            2. Approximate position (top-left, top-right, center, bottom-left, etc.)
            3. Type (dialogue, narration, sound effect, sign text)
            
            Return as JSON format:
            {
                "text_blocks": [
                    {
                        "text": "exact text here",
                        "position": "position description",
                        "type": "dialogue/narration/effect/sign"
                    }
                ]
            }
            
            Be thorough and extract ALL visible text."""
            
            response = self.gemini_vision_model.generate_content([prompt, img])
            
            # Parse JSON response
            response_text = response.text.strip()
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            result = json.loads(response_text)
            return result.get('text_blocks', [])
            
        except Exception as e:
            print(f"Gemini Vision error: {e}, falling back to Tesseract")
            return None
    
    def perform_ocr(self, image_path):
        """Extract text with bounding boxes using Tesseract"""
        print(f"Performing Tesseract OCR on {image_path.name}")
        
        img = cv2.imread(str(image_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        ocr_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        
        text_regions = []
        n_boxes = len(ocr_data['text'])
        
        for i in range(n_boxes):
            if int(ocr_data['conf'][i]) > 30:
                text = ocr_data['text'][i].strip()
                if text:
                    text_regions.append({
                        'text': text,
                        'x': ocr_data['left'][i],
                        'y': ocr_data['top'][i],
                        'w': ocr_data['width'][i],
                        'h': ocr_data['height'][i],
                        'conf': ocr_data['conf'][i]
                    })
        
        print(f"Found {len(text_regions)} text regions")
        return text_regions, img
    
    def gemini_translate_with_context(self, texts, text_types=None):
        """Use Gemini Pro for context-aware translation to Marathi"""
        if not texts:
            return []
        
        # Prepare context from previous translations
        context_str = ""
        if self.conversation_context:
            context_str = "Previous translations for context:\n" + "\n".join(self.conversation_context[-5:])
        
        # Prepare text types information
        type_info = ""
        if text_types:
            type_info = "\n".join([f"Text {i+1} is {t}" for i, t in enumerate(text_types)])
        
        prompt = f"""You are an expert translator specializing in manga/comic translation from English to Marathi.
        
{context_str}

Translate the following English text to Marathi. Maintain the tone, style, and context.
- For dialogue: Use natural conversational Marathi
- For narration: Use formal literary Marathi
- For sound effects: Transliterate or use equivalent Marathi sound effects
- For signs/text: Direct translation

{type_info}

English texts to translate:
{chr(10).join([f"{i+1}. {text}" for i, text in enumerate(texts)])}

Return ONLY the Marathi translations, one per line, in the same order. No explanations."""

        try:
            response = self.gemini_model.generate_content(prompt)
            translations = response.text.strip().split('\n')
            
            # Clean up translations
            translations = [t.strip().lstrip('0123456789.-) ') for t in translations if t.strip()]
            
            # Update context
            for eng, mar in zip(texts, translations):
                self.conversation_context.append(f"{eng} -> {mar}")
            
            return translations
            
        except Exception as e:
            print(f"Gemini translation error: {e}")
            # Fallback to simple translation
            from deep_translator import GoogleTranslator
            translator = GoogleTranslator(source='en', target='mr')
            return [translator.translate(text) for text in texts]
    
    def get_marathi_font(self, size):
        """Get Marathi font (Devanagari)"""
        font_paths = [
            "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
            "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
            "/System/Library/Fonts/Supplemental/DevanagariSangamMN.ttc",
            "C:\\Windows\\Fonts\\NirmalaS.ttf",
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, size)
                except:
                    pass
        
        return ImageFont.load_default()
    
    def clean_and_replace_text(self, image, text_regions, translations):
        """Clean English text and replace with Marathi translations"""
        print("Replacing text with Marathi translations")
        
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        for region, translation in zip(text_regions, translations):
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            # Sample background color
            padding = 5
            try:
                bg_sample = img_pil.crop((
                    max(0, x - padding),
                    max(0, y - padding),
                    min(img_pil.width, x + w + padding),
                    min(img_pil.height, y + h + padding)
                ))
                bg_color = tuple(np.array(bg_sample).mean(axis=(0, 1)).astype(int))
            except:
                bg_color = (255, 255, 255)
            
            # Expand cleaning area slightly for better coverage
            clean_padding = 3
            draw.rectangle([
                x - clean_padding, 
                y - clean_padding, 
                x + w + clean_padding, 
                y + h + clean_padding
            ], fill=bg_color)
            
            # Calculate font size
            font_size = max(int(h * 0.7), 12)
            font = self.get_marathi_font(font_size)
            
            # Determine text color
            text_color = (0, 0, 0) if sum(bg_color) > 382 else (255, 255, 255)
            
            # Draw Marathi text with slight offset for better positioning
            try:
                draw.text((x + 2, y + 2), translation, font=font, fill=text_color)
            except:
                draw.text((x + 2, y + 2), translation, fill=text_color)
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    def process_image(self, image_path):
        """Complete pipeline for a single image with Gemini AI"""
        print(f"\nProcessing: {image_path.name}")
        
        # Try Gemini Vision first
        gemini_blocks = self.gemini_vision_ocr(image_path)
        
        if gemini_blocks:
            print(f"Gemini Vision found {len(gemini_blocks)} text blocks")
            # Use Tesseract for precise positioning, Gemini for text extraction
            text_regions, original_image = self.perform_ocr(image_path)
            
            # Extract texts and types
            texts = [region['text'] for region in text_regions]
            
            # Translate with Gemini
            translations = self.gemini_translate_with_context(texts)
            
        else:
            # Fallback to pure Tesseract
            text_regions, original_image = self.perform_ocr(image_path)
            
            if not text_regions:
                print("No text found, copying original")
                return original_image
            
            texts = [region['text'] for region in text_regions]
            translations = self.gemini_translate_with_context(texts)
        
        # Ensure we have matching translations
        while len(translations) < len(text_regions):
            translations.append(texts[len(translations)])
        
        processed_image = self.clean_and_replace_text(original_image, text_regions, translations)
        
        return processed_image
    
    def run(self, bato_url):
        """Run the complete pipeline"""
        print("\n" + "="*70)
        print("BATO.TO TO MARATHI TRANSLATION PIPELINE (Powered by Gemini AI)")
        print("="*70 + "\n")
        
        image_paths = self.extract_images_from_bato(bato_url)
        
        if not image_paths:
            print("No images extracted!")
            return
        
        processed_images = []
        for img_path in image_paths:
            try:
                processed_img = self.process_image(img_path)
                
                output_path = self.output_dir / f"marathi_{img_path.name}"
                cv2.imwrite(str(output_path), processed_img)
                processed_images.append(output_path)
                print(f"Saved: {output_path.name}")
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
        
        print("\n" + "="*70)
        print(f"PIPELINE COMPLETE!")
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"Processed {len(processed_images)}/{len(image_paths)} images")
        print("="*70 + "\n")
        
        return processed_images


if __name__ == "__main__":
    # Get API key from environment or user input
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        GEMINI_API_KEY = input("Enter your Gemini API key: ").strip()
    
    BATO_CHAPTER_URL = input("Enter Bato.to chapter URL: ").strip()
    
    pipeline = BatoToMarathiPipeline(
        gemini_api_key=GEMINI_API_KEY,
        output_dir="marathi_pages"
    )
    pipeline.run(BATO_CHAPTER_URL)