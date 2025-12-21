import os
import json
import shutil
import time
from pathlib import Path
import requests
from difflib import SequenceMatcher


class PPOCRFuzzyMatcher:
    """
    Fast PPOCRLabel relabeling using:
    1. OCR full images (not crops)
    2. Fuzzy match to find correct text for each bounding box
    """
    
    def __init__(self, label_dir, api_key=None):
        """
        Args:
            label_dir: Directory containing PPOCRLabel output files
            api_key: OCR.space API key
        """
        self.label_dir = Path(label_dir)
        self.label_file = self.label_dir / "Label.txt"
        self.api_key = api_key
        self.backup_dir = self.label_dir / "backup_fuzzy_match"
        
        # Supported image extensions
        self.supported_extensions = {
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', 
            '.webp', '.gif', '.JPG', '.JPEG', '.PNG', '.BMP'
        }
        
        # Try to import rapidfuzz, fallback to difflib
        try:
            from rapidfuzz import fuzz
            self.fuzz = fuzz
            self.fuzzy_lib = 'rapidfuzz'
            print("Using RapidFuzz")
        except ImportError:
            print("RapidFuzz not found, using difflib")
            print("   Install with: pip install rapidfuzz")
            self.fuzz = None
            self.fuzzy_lib = 'difflib'
    
    def fuzzy_similarity(self, str1, str2):
        """
        Calculate similarity between two strings (0-100)
        
        Args:
            str1, str2: Strings to compare
            
        Returns:
            Similarity score (0-100, higher is more similar)
        """
        if not str1 or not str2:
            return 0
        
        if self.fuzzy_lib == 'rapidfuzz':
            return self.fuzz.ratio(str1.lower(), str2.lower())
        else:
            return SequenceMatcher(None, str1.lower(), str2.lower()).ratio() * 100
    
    def find_image_file(self, img_path_from_label):
        """
        Find the actual image file, handling:
        - Different file extensions (.jpg, .png, etc.)
        - Relative paths
        - Files in subdirectories
        
        Args:
            img_path_from_label: Path from Label.txt
            
        Returns:
            Path object to actual image file, or None if not found
        """
        # Try the path as-is first
        full_path = self.label_dir / img_path_from_label
        if full_path.exists() and full_path.is_file():
            return full_path
        
        # Try different extensions
        base_path = full_path.with_suffix('')
        for ext in self.supported_extensions:
            test_path = base_path.with_suffix(ext)
            if test_path.exists() and test_path.is_file():
                return test_path
        
        # Try searching in subdirectories
        filename = Path(img_path_from_label).name
        filename_no_ext = Path(img_path_from_label).stem
        
        # Search recursively for matching filename
        for img_file in self.label_dir.rglob('*'):
            if img_file.is_file() and img_file.suffix.lower() in {ext.lower() for ext in self.supported_extensions}:
                # Check if filename matches (with or without extension)
                if img_file.name == filename or img_file.stem == filename_no_ext:
                    return img_file
        
        return None
    def find_best_match(self, target_text, ocr_results, min_similarity=60):
        """
        Find the best matching text from OCR results using fuzzy matching
        
        Args:
            target_text: Original text from Label.txt
            ocr_results: List of text strings from full image OCR
            min_similarity: Minimum similarity threshold (0-100)
            
        Returns:
            (best_match_text, similarity_score) or (None, 0) if no good match
        """
        if not ocr_results:
            return None, 0
        
        best_match = None
        best_score = 0
        
        for ocr_text in ocr_results:
            score = self.fuzzy_similarity(target_text, ocr_text)
            if score > best_score:
                best_score = score
                best_match = ocr_text
        
        # Only return match if above threshold
        if best_score >= min_similarity:
            return best_match, best_score
        
        return None, 0
    
    def ocr_full_image_ocrspace(self, image_path, language='auto'):
        """
        OCR entire image using OCR.space API
        Returns list of detected text strings
        """
        if not self.api_key:
            print("API key required for OCR.space")
            return []
        
        url = "https://api.ocr.space/parse/image"
        payload = {
            'isOverlayRequired': False,
            'apikey': self.api_key,
            'language': language,
            'OCREngine': '2',
            'detectOrientation': True,
        }
        
        try:
            with open(image_path, 'rb') as f:
                r = requests.post(
                    url,
                    files={image_path.name: f},
                    data=payload,
                    timeout=60
                )
            
            if r.status_code != 200:
                print(f"   HTTP {r.status_code}: {r.text[:100]}")
                return []
            
            result = json.loads(r.content.decode())
            
            if result.get('IsErroredOnProcessing'):
                error_msg = result.get('ErrorMessage', ['Unknown error'])
                print(f"   Error: {error_msg[0] if error_msg else 'Unknown'}")
                return []
            
            # Extract all text lines
            texts = []
            if result.get('ParsedResults'):
                full_text = result['ParsedResults'][0].get('ParsedText', '')
                # Split into lines and filter empty
                texts = [line.strip() for line in full_text.split('\n') if line.strip()]
            
            return texts
            
        except Exception as e:
            print(f"   Error: {e}")
            return []
    
    def ocr_full_image_paddleocr(self, image_path):
        """
        OCR entire image using PaddleOCR (local, free, unlimited!)
        Returns list of detected text strings
        """
        try:
            from paddleocr import PaddleOCR
            
            # Initialize PaddleOCR (cache it for reuse)
            if not hasattr(self, 'paddle_ocr'):
                self.paddle_ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                )
            
            # Run OCR
            result = self.paddle_ocr.ocr(str(image_path))
            
            # Extract text
            texts = []
            if result and result[0]:
                for line in result[0]:
                    text = line[1][0]  # line[1][0] is the text, line[1][1] is confidence
                    texts.append(text.strip())
            
            return texts
            
        except ImportError:
            print("PaddleOCR not found. Install with: pip install paddleocr")
            return []
        except Exception as e:
            print(f"   Error: {e}")
            return []
    
    def backup_files(self):
        """Backup Label.txt"""
        self.backup_dir.mkdir(exist_ok=True)
        if self.label_file.exists():
            shutil.copy2(self.label_file, self.backup_dir / self.label_file.name)
            print(f"Backup: {self.backup_dir / self.label_file.name}")
    
    def load_label_data(self):
        """Load Label.txt"""
        label_data = {}
        if not self.label_file.exists():
            print(f"{self.label_file} not found!")
            return label_data
        
        with open(self.label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    img_path = parts[0]
                    annotations = json.loads(parts[1])
                    label_data[img_path] = annotations
        
        print(f"Loaded {len(label_data)} images")
        return label_data
    
    def save_label_data(self, label_data):
        """Save updated Label.txt"""
        with open(self.label_file, 'w', encoding='utf-8') as f:
            for img_path, annotations in label_data.items():
                line = f"{img_path}\t{json.dumps(annotations, ensure_ascii=False)}\n"
                f.write(line)
        print(f"Saved Label.txt")
    
    def relabel_with_fuzzy_matching(self, use_paddleocr=True, language='auto', 
                                    min_similarity=60, rate_limit_delay=25, start_from=0):
        """
        Main relabeling function using fuzzy matching
        
        Args:
            use_paddleocr: True to use PaddleOCR, False for OCR.space
            language: Language for OCR
            min_similarity: Minimum fuzzy match score (0-100)
            rate_limit_delay: Delay for API calls in SECONDS (only if using OCR.space)
                             Recommended: 25 seconds = ~144 images/hour (under 180 limit)
            start_from: Skip first N images (for resuming after rate limit)
        """
        print(f"\n{'='*60}")
        print(f"FUZZY MATCHING RELABELING")
        print(f"{'='*60}")
        
        # Backup
        self.backup_files()
        
        # Load data
        label_data = self.load_label_data()
        if not label_data:
            return
        
        # Statistics
        total_images = len(label_data)
        total_boxes = sum(len(boxes) for boxes in label_data.values())
        updated = 0
        failed = 0
        skipped = 0
        
        print(f"   Dataset:")
        print(f"   Images: {total_images}")
        print(f"   Total boxes: {total_boxes}")
        print(f"   Method: {'PaddleOCR (local)' if use_paddleocr else 'OCR.space (API)'}")
        print(f"   Min similarity: {min_similarity}%")
        print(f"   Supported formats: {', '.join(sorted(self.supported_extensions))}")
        
        if not use_paddleocr:
            print(f"\nAPI RATE LIMITS:")
            print(f"   Delay: {rate_limit_delay} seconds between requests")
            images_per_hour = int(3600 / rate_limit_delay)
            print(f"   Speed: ~{images_per_hour} images/hour")
            estimated_hours = total_images / images_per_hour
            print(f"   Estimated time: {estimated_hours:.1f} hours")
            if images_per_hour > 180:
                print(f"   WARNING: {images_per_hour}/hour exceeds 180/hour limit!")
                print(f"   Recommended delay: 25+ seconds")
        
        if start_from > 0:
            print(f"\nResuming from image {start_from + 1}")
        
        print(f"\n{'='*60}")
        print(f"PROCESSING")
        print(f"{'='*60}\n")
        
        # Process each image
        import datetime
        start_time = datetime.datetime.now()
        
        for idx, (img_path, annotations) in enumerate(label_data.items(), 1):
            # Skip if resuming
            if idx <= start_from:
                skipped += 1
                continue
            
            # Find the actual image file (handles different extensions, subdirectories)
            full_img_path = self.find_image_file(img_path)
            
            if full_img_path is None:
                print(f"[{idx}/{total_images}] {img_path} - Image not found")
                print(f"   Searched for: {img_path}")
                print(f"   Supported formats: {', '.join(sorted(self.supported_extensions))}")
                failed += len(annotations)
                continue
            
            # Show actual path if different from Label.txt
            if str(full_img_path) != str(self.label_dir / img_path):
                actual_rel_path = full_img_path.relative_to(self.label_dir)
                print(f"[{idx}/{total_images}] {img_path} → {actual_rel_path}")
            else:
                print(f"[{idx}/{total_images}] {img_path}")
            
            # Calculate ETA
            if idx > start_from + 1:
                elapsed = (datetime.datetime.now() - start_time).total_seconds()
                processed = idx - start_from - 1
                avg_time = elapsed / processed
                remaining = (total_images - idx) * avg_time
                eta = datetime.timedelta(seconds=int(remaining))
                eta_str = f"ETA: {eta}"
            else:
                eta_str = "ETA: Calculating..."
            
            print(f"   {eta_str}")
            
            # OCR the full image
            if use_paddleocr:
                ocr_results = self.ocr_full_image_paddleocr(full_img_path)
            else:
                ocr_results = self.ocr_full_image_ocrspace(full_img_path, language)
            
            if not ocr_results:
                print(f" No OCR results!")
                failed += len(annotations)
                
                # Still delay even on failure to avoid rate limit
                if not use_paddleocr:
                    print(f" Waiting {rate_limit_delay}s...")
                    time.sleep(rate_limit_delay)
                continue
            
            print(f" OCR found {len(ocr_results)} text regions")
            
            # Match each bounding box
            matched = 0
            for box in annotations:
                old_text = box.get('transcription', '')
                
                # Find best match using fuzzy matching
                new_text, score = self.find_best_match(old_text, ocr_results, min_similarity)
                
                if new_text and new_text != old_text:
                    box['transcription'] = new_text
                    matched += 1
                    updated += 1
                elif not new_text:
                    failed += 1
            
            print(f"Matched {matched}/{len(annotations)} boxes")
            
            # Rate limiting for API
            if not use_paddleocr:
                print(f" Waiting {rate_limit_delay}s before next request...")
                time.sleep(rate_limit_delay)
        
        # Save
        self.save_label_data(label_data)
        
        # Summary
        total_time = (datetime.datetime.now() - start_time).total_seconds()
        print(f"\n{'='*60}")
        print(f"COMPLETE!")
        print(f"{'='*60}")
        print(f"   Total time: {datetime.timedelta(seconds=int(total_time))}")
        print(f"   Total boxes: {total_boxes}")
        print(f"   Updated: {updated} ({updated/total_boxes*100:.1f}%)")
        print(f"   Failed: {failed} ({failed/total_boxes*100:.1f}%)")
        if skipped > 0:
            print(f"   Skipped: {skipped} (start_from={start_from})")
        print(f"   Backup: {self.backup_dir}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    LABEL_DIR = "./img/T1"
    API_KEY = "K87099011988957"
    matcher = PPOCRFuzzyMatcher(LABEL_DIR, API_KEY)
    
    matcher.relabel_with_fuzzy_matching(
        use_paddleocr=False,
        language='auto',          # Auto-detect Vietnamese
        min_similarity=70,
        rate_limit_delay=25,      # 25 seconds = ~144 images/hour (safe for 180/hour limit)
        start_from=0              # Resume from this image index if interrupted
    )
    
    # If you hit "180 images per 3600 seconds" limit:
    # 1. Wait 1 hour
    # 2. Run again with start_from=NUMBER where you left off:
    #
    # matcher.relabel_with_fuzzy_matching(
    #     use_paddleocr=False,
    #     language='auto',
    #     min_similarity=60,
    #     rate_limit_delay=25,
    #     start_from=144  # Start from image 145 (skip first 144)
    # )
    
    print("\n" + "="*60)
    print(" RATE LIMIT GUIDE FOR OCR.SPACE:")
    print("="*60)
    print("OCR.space free tier: 180 requests per hour (3600 seconds)")
    print("")
    print("Recommended delays:")
    print("  • 20 seconds = 180 images/hour (exactly at limit)")
    print("  • 25 seconds = 144 images/hour (safe, recommended) ⭐")
    print("  • 30 seconds = 120 images/hour (very safe)")
    print("")
    print("If you hit rate limit:")
    print("  1. Wait 1 hour")
    print("  2. Check console output for last processed image number")
    print("  3. Resume with: start_from=THAT_NUMBER")
    print("")
    print("Example:")
    print("  [144/500] image_144.jpg  <- Hit rate limit here")
    print("  Wait 1 hour, then:")
    print("  start_from=144  <- This will skip first 144 and start from 145")
    
    print("\nADVANCED TIPS:")
    print("   • Install rapidfuzz for faster matching:")
    print("     pip install rapidfuzz")
    print("   • Adjust min_similarity based on your data:")
    print("     - 40-50: More matches, less accurate")
    print("     - 60-70: Balanced (recommended)")
    print("     - 80-90: Fewer matches, more accurate")
    print("   • PaddleOCR processes ~100-500 images/hour (depending on hardware)")
    print("   • OCR.space with 25s delay = ~144 images/hour")