#!/usr/bin/env python3
"""
Bato.to to Marathi Translator - Pure Gemini Vision AI
No Tesseract needed - works perfectly on GitHub Actions
"""

import requests
import json
import re
import os
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import google.generativeai as genai

class BatoMarathiTranslator:
    def __init__(self, gemini_api_key):
        # Setup session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://bato.to/',
            'DNT': '1',
        })
        
        # Gemini AI - Use latest 2.5 Flash (supports vision + text)
        genai.configure(api_key=gemini_api_key)
        self.gemini_vision = genai.GenerativeModel('gemini-2.5-flash')
        self.gemini_text = genai.GenerativeModel('gemini-2.5-flash')
        
        self.translation_context = []
        self.base_output_dir = Path("translated_chapters")
        self.base_output_dir.mkdir(exist_ok=True)
    
    def extract_chapter_id(self, url):
        """Extract chapter ID"""
        match = re.search(r'/chapter/(\d+)', url)
        return match.group(1) if match else None
    
    def get_all_image_urls(self, url):
        """Get ALL image URLs from page"""
        print(f"\n🔍 Fetching page: {url}")
        
        try:
            response = self.session.get(url, timeout=30)
            html = response.text
            
            print(f"✅ Page loaded ({len(html)} bytes)")
            
            # Extract all possible image URLs
            all_urls = set()
            
            # Method 1: img tags
            img_tags = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', html, re.IGNORECASE)
            all_urls.update(img_tags)
            
            # Method 2: data-src
            data_src = re.findall(r'data-src=["\']([^"\']+)["\']', html, re.IGNORECASE)
            all_urls.update(data_src)
            
            # Method 3: JavaScript image URLs
            js_images = re.findall(r'["\']https?://[^"\']*\.(?:jpg|jpeg|png|webp|gif)[^"\']*["\']', html, re.IGNORECASE)
            for match in js_images:
                all_urls.add(match.strip('"\''))
            
            # Method 4: Direct URLs
            direct = re.findall(r'https?://[^\s<>"\']+\.(?:jpg|jpeg|png|webp|gif)[^\s<>"\']*', html, re.IGNORECASE)
            all_urls.update(direct)
            
            # Filter for manga images
            manga_images = []
            for img_url in all_urls:
                if (img_url.startswith('http') and 
                    len(img_url) > 40 and
                    not any(skip in img_url.lower() for skip in ['icon', 'logo', 'avatar', 'banner', 'button', 'thumb'])):
                    manga_images.append(img_url)
            
            # Deduplicate
            seen = set()
            unique = []
            for url in manga_images:
                if url not in seen:
                    seen.add(url)
                    unique.append(url)
            
            print(f"✅ Found {len(unique)} manga images")
            
            if unique:
                print(f"\n📋 First few URLs:")
                for i, u in enumerate(unique[:3], 1):
                    print(f"  {i}. {u[:80]}...")
            
            return unique
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return []
    
    def download_image(self, img_url, save_path):
        """Download image"""
        try:
            headers = self.session.headers.copy()
            headers['Referer'] = 'https://bato.to/'
            
            response = self.session.get(img_url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                return True
            return False
        except:
            return False
    
    def extract_text_with_gemini_vision(self, image_path):
        """Use Gemini Vision to extract text with positions"""
        print(f"  🤖 Using Gemini Vision for text extraction...")
        
        try:
            img = Image.open(image_path)
            
            prompt = """Analyze this manga/comic page image carefully.

Extract ALL English text you can see. For EACH text block, provide:
1. The exact English text
2. Approximate position (describe where on the page: "top-left", "center", "bottom-right", etc.)
3. Approximate coordinates as percentage (x%, y% from top-left corner)
4. Estimated size (small/medium/large)

Return in this JSON format:
{
  "text_blocks": [
    {
      "text": "exact English text here",
      "position": "top-left",
      "x_percent": 10,
      "y_percent": 15,
      "size": "medium"
    }
  ]
}

Extract EVERYTHING including dialogue, sound effects, signs, narration.
Be thorough and precise."""

            response = self.gemini_vision.generate_content([prompt, img])
            result_text = response.text.strip()
            
            # Parse JSON
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0].strip()
            elif '```' in result_text:
                result_text = result_text.split('```')[1].split('```')[0].strip()
            
            data = json.loads(result_text)
            text_blocks = data.get('text_blocks', [])
            
            print(f"  ✅ Gemini found {len(text_blocks)} text blocks")
            
            # Convert to pixel coordinates
            img_width, img_height = img.size
            
            regions = []
            for block in text_blocks:
                x_percent = block.get('x_percent', 50)
                y_percent = block.get('y_percent', 50)
                
                # Convert to pixels
                x = int(img_width * x_percent / 100)
                y = int(img_height * y_percent / 100)
                
                # Estimate size based on description
                size_map = {'small': 40, 'medium': 80, 'large': 120}
                estimated_size = size_map.get(block.get('size', 'medium'), 80)
                
                regions.append({
                    'text': block.get('text', ''),
                    'x': max(0, x - estimated_size//2),
                    'y': max(0, y - estimated_size//2),
                    'w': estimated_size,
                    'h': estimated_size // 3,
                    'position': block.get('position', 'unknown')
                })
            
            return regions
            
        except Exception as e:
            print(f"  ❌ Gemini Vision error: {e}")
            return []
    
    def translate_batch_with_context(self, texts):
        """Translate with Gemini maintaining context"""
        if not texts:
            return []
        
        context = "\n".join(self.translation_context[-10:]) if self.translation_context else "Starting new manga chapter"
        
        prompt = f"""You are an expert manga translator specializing in English to Marathi translation.

Previous translations for context:
{context}

Translate these English texts to natural, conversational Marathi:

{chr(10).join([f"{i+1}. {text}" for i, text in enumerate(texts)])}

Rules:
- Keep the natural flow of dialogue
- Use colloquial Marathi for casual speech
- Preserve character personality and emotion
- For sound effects, use Marathi equivalents (like "धम्म" for "THUD")
- Stay true to the original meaning

Return ONLY the Marathi translations, one per line, in exact order. No numbering, explanations, or extra text."""

        try:
            response = self.gemini_text.generate_content(prompt)
            result = response.text.strip()
            
            translations = []
            for line in result.split('\n'):
                line = line.strip()
                # Clean numbering
                line = re.sub(r'^\d+[\.\):\-]\s*', '', line)
                line = re.sub(r'^[*\-•]\s*', '', line)
                if line:
                    translations.append(line)
            
            # Update context
            for eng, mar in zip(texts, translations):
                self.translation_context.append(f"{eng} → {mar}")
                if len(self.translation_context) > 50:
                    self.translation_context.pop(0)
            
            # Ensure we have enough
            while len(translations) < len(texts):
                translations.append(texts[len(translations)])
            
            return translations[:len(texts)]
            
        except Exception as e:
            print(f"  ⚠️ Translation error: {e}")
            # Simple fallback
            return [f"[{text}]" for text in texts]
    
    def get_marathi_font(self, size):
        """Get Marathi font"""
        fonts = [
            "C:\\Windows\\Fonts\\NirmalaS.ttf",
            "C:\\Windows\\Fonts\\Nirmala.ttf",
            "C:\\Windows\\Fonts\\mangal.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
            "/System/Library/Fonts/Supplemental/DevanagariSangamMN.ttc",
            "arial.ttf",
        ]
        
        for font_path in fonts:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, size)
                except:
                    pass
        
        return ImageFont.load_default()
    
    def clean_and_replace_text(self, img_array, text_regions, translations):
        """Clean and replace with Marathi"""
        img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        replaced = 0
        
        for region, translation in zip(text_regions, translations):
            try:
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                
                # Expand cleaning area
                padding = 12
                clean_box = [
                    max(0, x - padding),
                    max(0, y - padding),
                    min(img_pil.width, x + w + padding),
                    min(img_pil.height, y + h + padding)
                ]
                
                # Sample background
                try:
                    sample = img_pil.crop(clean_box)
                    pixels = np.array(sample)
                    bg_color = tuple(np.median(pixels.reshape(-1, 3), axis=0).astype(int))
                except:
                    bg_color = (255, 255, 255)
                
                # Clean thoroughly
                draw.rectangle(clean_box, fill=bg_color)
                
                # Font size
                font_size = max(int(h * 0.8), 16)
                font = self.get_marathi_font(font_size)
                
                # Text color
                brightness = sum(bg_color) / 3
                text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
                
                # Draw text
                text_pos = (x + 4, y + 3)
                try:
                    draw.text(text_pos, translation, font=font, fill=text_color)
                except:
                    draw.text(text_pos, translation, fill=text_color)
                
                replaced += 1
                
            except Exception as e:
                print(f"    ⚠️ Replace error: {e}")
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR), replaced
    
    def process_image(self, img_path, index, total):
        """Process single image with Gemini Vision"""
        print(f"\n📄 [{index}/{total}] {img_path.name}")
        
        # Extract text with Gemini Vision
        text_regions = self.extract_text_with_gemini_vision(img_path)
        
        if not text_regions:
            print(f"  ⚠️  Gemini found no text")
            # Save original
            img = cv2.imread(str(img_path))
            return img, 0
        
        print(f"  📝 Texts found:")
        for i, region in enumerate(text_regions[:3], 1):
            print(f"    {i}. \"{region['text']}\" at {region['position']}")
        if len(text_regions) > 3:
            print(f"    ... and {len(text_regions)-3} more")
        
        # Extract texts
        texts = [r['text'] for r in text_regions]
        
        # Translate
        print(f"  🌐 Translating to Marathi...")
        translations = self.translate_batch_with_context(texts)
        
        print(f"  ✅ Translations:")
        for i, trans in enumerate(translations[:3], 1):
            print(f"    {i}. \"{trans}\"")
        
        # Load image
        img = cv2.imread(str(img_path))
        
        # Replace text
        print(f"  🎨 Replacing text...")
        processed, replaced = self.clean_and_replace_text(img, text_regions, translations)
        
        print(f"  ✅ Replaced {replaced}/{len(text_regions)} text blocks")
        
        return processed, replaced
    
    def run(self, bato_url):
        """Main pipeline"""
        print("\n" + "="*70)
        print("🌸 BATO.TO → MARATHI (Pure Gemini AI) 🌸")
        print("="*70)
        
        chapter_id = self.extract_chapter_id(bato_url)
        if not chapter_id:
            print("❌ Invalid URL")
            return
        
        print(f"\n📖 Chapter: {chapter_id}")
        
        # Create folder
        chapter_dir = self.base_output_dir / f"chapter_{chapter_id}"
        chapter_dir.mkdir(exist_ok=True)
        print(f"📁 Output: {chapter_dir}")
        
        # Get images
        image_urls = self.get_all_image_urls(bato_url)
        
        if not image_urls:
            print("\n💡 Manual mode - paste URLs:")
            manual = []
            while True:
                url = input().strip()
                if not url:
                    break
                if url.startswith('http'):
                    manual.append(url)
            image_urls = manual
        
        if not image_urls:
            print("❌ No images!")
            return
        
        # Download
        print(f"\n⬇️  Downloading {len(image_urls)} images...")
        downloaded = []
        
        for i, url in enumerate(image_urls, 1):
            print(f"  [{i}/{len(image_urls)}]", end=' ')
            save_path = chapter_dir / f"original_{i:03d}.jpg"
            
            if self.download_image(url, save_path):
                print("✅")
                downloaded.append(save_path)
            else:
                print("❌")
            
            time.sleep(0.3)
        
        print(f"\n✅ Downloaded {len(downloaded)}/{len(image_urls)}")
        
        if not downloaded:
            print("❌ Nothing to process!")
            return
        
        # Process
        print(f"\n🔄 Processing with Gemini Vision...")
        processed_count = 0
        total_replacements = 0
        
        for i, img_path in enumerate(downloaded, 1):
            try:
                processed_img, replacements = self.process_image(img_path, i, len(downloaded))
                
                output_path = chapter_dir / f"marathi_{i:03d}.jpg"
                cv2.imwrite(str(output_path), processed_img)
                
                processed_count += 1
                total_replacements += replacements
                
                # Rate limit - Gemini free tier
                time.sleep(2)
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        print("\n" + "="*70)
        print(f"✅ COMPLETE!")
        print(f"📊 {processed_count}/{len(downloaded)} images")
        print(f"📝 {total_replacements} total text replacements")
        print(f"📁 {chapter_dir.absolute()}")
        print("="*70 + "\n")


if __name__ == "__main__":
    import sys
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        api_key = input("🔑 Gemini API key: ").strip()
    
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("🔗 Bato.to URL: ").strip()
    
    translator = BatoMarathiTranslator(api_key)
    translator.run(url)