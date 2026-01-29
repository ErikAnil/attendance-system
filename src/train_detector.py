"""
Enhanced ID Card Detector Training Script
Uses all available training images for better accuracy
"""

import cv2
import numpy as np
from pathlib import Path
import pickle
import random
from typing import List, Dict, Tuple
import os

class EnhancedIDCardTrainer:
    """
    Train an ID card detector using all available samples
    """
    
    def __init__(self, training_dir: str = "training_data", model_dir: str = "models"):
        self.training_dir = Path(training_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
    def load_all_training_images(self) -> List[np.ndarray]:
        """Load all ID card images from training directory"""
        images = []
        extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        for ext in extensions:
            for img_path in self.training_dir.glob(f'*{ext}'):
                img = cv2.imread(str(img_path))
                if img is not None:
                    images.append((img, img_path.name))
                    print(f"  ✓ Loaded: {img_path.name}")
                else:
                    print(f"  ✗ Failed to load: {img_path.name}")
        
        return images
    
    def augment_image(self, image: np.ndarray, n_augmentations: int = 20) -> List[np.ndarray]:
        """Generate augmented versions of an image"""
        augmented = [image.copy()]
        h, w = image.shape[:2]
        
        for _ in range(n_augmentations):
            aug_img = image.copy()
            
            # Random rotation (-15 to +15 degrees)
            angle = random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            
            # Random brightness (0.7 to 1.3)
            brightness = random.uniform(0.7, 1.3)
            aug_img = cv2.convertScaleAbs(aug_img, alpha=brightness, beta=0)
            
            # Random blur
            if random.random() > 0.5:
                kernel_size = random.choice([3, 5])
                aug_img = cv2.GaussianBlur(aug_img, (kernel_size, kernel_size), 0)
            
            # Random noise
            if random.random() > 0.7:
                noise = np.random.randint(-15, 15, aug_img.shape, dtype=np.int16)
                aug_img = np.clip(aug_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Random perspective transform
            if random.random() > 0.6:
                pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
                offset = 20
                pts2 = np.float32([
                    [random.randint(0, offset), random.randint(0, offset)],
                    [w - random.randint(0, offset), random.randint(0, offset)],
                    [random.randint(0, offset), h - random.randint(0, offset)],
                    [w - random.randint(0, offset), h - random.randint(0, offset)]
                ])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                aug_img = cv2.warpPerspective(aug_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            
            augmented.append(aug_img)
        
        return augmented
    
    def extract_features(self, image: np.ndarray) -> Dict:
        """Extract robust features from ID card image"""
        # Resize to standard size
        standard_size = (400, 250)  # w x h (ID card aspect ratio ~1.6)
        resized = cv2.resize(image, standard_size)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Color histogram (32 bins per channel)
        color_hist = []
        for i in range(3):
            hist = cv2.calcHist([resized], [i], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            color_hist.extend(hist)
        
        # Edge histogram
        edges = cv2.Canny(gray, 50, 150)
        edge_hist = np.histogram(edges.flatten(), bins=16)[0]
        edge_hist = edge_hist / (edge_hist.sum() + 1e-6)
        
        # HOG features (for shape detection)
        win_size = (400, 256)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        
        # Resize gray to match HOG window
        gray_hog = cv2.resize(gray, win_size)
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        hog_features = hog.compute(gray_hog)
        
        # Aspect ratio
        aspect_ratio = image.shape[1] / image.shape[0]
        
        return {
            'color_hist': np.array(color_hist),
            'edge_hist': edge_hist,
            'hog_features': hog_features.flatten() if hog_features is not None else np.array([]),
            'aspect_ratio': aspect_ratio,
            'gray_template': gray
        }
    
    def train(self, augment_per_image: int = 20):
        """Train the detector using all available images"""
        print("\n" + "="*60)
        print("🎓 ENHANCED ID CARD DETECTOR TRAINING")
        print("="*60)
        
        # Load all training images
        print(f"\n📁 Loading images from: {self.training_dir}")
        images_with_names = self.load_all_training_images()
        
        if not images_with_names:
            print("❌ No training images found!")
            return False
        
        print(f"\n✓ Loaded {len(images_with_names)} training images")
        
        # Process each image
        all_features = []
        all_templates = []
        
        print(f"\n🔄 Processing images (augmenting {augment_per_image}x each)...")
        
        for idx, (img, name) in enumerate(images_with_names):
            print(f"  [{idx+1}/{len(images_with_names)}] Processing {name}...")
            
            # Generate augmented versions
            augmented = self.augment_image(img, n_augmentations=augment_per_image)
            
            # Extract features from each augmented image
            for aug_img in augmented:
                features = self.extract_features(aug_img)
                all_features.append(features)
            
            # Store original as template
            orig_features = self.extract_features(img)
            all_templates.append(orig_features['gray_template'])
        
        print(f"\n✓ Generated {len(all_features)} training samples")
        
        # Compute model statistics
        print("\n📊 Computing model statistics...")
        
        # Average color histogram
        color_hists = np.array([f['color_hist'] for f in all_features])
        avg_color_hist = np.mean(color_hists, axis=0)
        std_color_hist = np.std(color_hists, axis=0)
        
        # Average edge histogram
        edge_hists = np.array([f['edge_hist'] for f in all_features])
        avg_edge_hist = np.mean(edge_hists, axis=0)
        std_edge_hist = np.std(edge_hists, axis=0)
        
        # Average HOG features
        hog_features = [f['hog_features'] for f in all_features if len(f['hog_features']) > 0]
        if hog_features:
            avg_hog = np.mean(hog_features, axis=0)
            std_hog = np.std(hog_features, axis=0)
        else:
            avg_hog = np.array([])
            std_hog = np.array([])
        
        # Average aspect ratio
        aspect_ratios = [f['aspect_ratio'] for f in all_features]
        avg_aspect_ratio = np.mean(aspect_ratios)
        std_aspect_ratio = np.std(aspect_ratios)
        
        # Create composite template from multiple images
        composite_template = np.mean(all_templates, axis=0).astype(np.uint8)
        
        # Save model
        model_data = {
            'avg_color_hist': avg_color_hist,
            'std_color_hist': std_color_hist,
            'avg_edge_hist': avg_edge_hist,
            'std_edge_hist': std_edge_hist,
            'avg_hog': avg_hog,
            'std_hog': std_hog,
            'avg_aspect_ratio': avg_aspect_ratio,
            'std_aspect_ratio': std_aspect_ratio,
            'composite_template': composite_template,
            'all_templates': all_templates,
            'num_training_samples': len(all_features),
            'num_original_images': len(images_with_names)
        }
        
        # Save as numpy file
        model_path = self.model_dir / "enhanced_id_model.npz"
        np.savez(
            model_path,
            avg_color_hist=avg_color_hist,
            std_color_hist=std_color_hist,
            avg_edge_hist=avg_edge_hist,
            std_edge_hist=std_edge_hist,
            avg_hog=avg_hog,
            std_hog=std_hog,
            avg_aspect_ratio=avg_aspect_ratio,
            std_aspect_ratio=std_aspect_ratio,
            composite_template=composite_template
        )
        
        # Save templates separately
        templates_path = self.model_dir / "templates.pkl"
        with open(templates_path, 'wb') as f:
            pickle.dump(all_templates, f)
        
        print(f"\n✓ Model saved to: {model_path}")
        print(f"✓ Templates saved to: {templates_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("📈 TRAINING SUMMARY")
        print("="*60)
        print(f"  Original images:     {len(images_with_names)}")
        print(f"  Total samples:       {len(all_features)}")
        print(f"  Avg aspect ratio:    {avg_aspect_ratio:.3f} ± {std_aspect_ratio:.3f}")
        print(f"  Templates stored:    {len(all_templates)}")
        print("="*60)
        
        return True


class EnhancedIDCardDetector:
    """
    Detect ID cards using the enhanced trained model
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_loaded = self.load_model()
        
    def load_model(self) -> bool:
        """Load trained model"""
        model_path = self.model_dir / "enhanced_id_model.npz"
        templates_path = self.model_dir / "templates.pkl"
        
        if not model_path.exists():
            print(f"⚠️ Model not found at {model_path}")
            print("  Run training first: python train_detector.py")
            return False
        
        data = np.load(model_path)
        self.avg_color_hist = data['avg_color_hist']
        self.std_color_hist = data['std_color_hist']
        self.avg_edge_hist = data['avg_edge_hist']
        self.std_edge_hist = data['std_edge_hist']
        self.avg_hog = data['avg_hog']
        self.std_hog = data['std_hog']
        self.avg_aspect_ratio = float(data['avg_aspect_ratio'])
        self.std_aspect_ratio = float(data['std_aspect_ratio'])
        self.composite_template = data['composite_template']
        
        # Load templates
        if templates_path.exists():
            with open(templates_path, 'rb') as f:
                self.templates = pickle.load(f)
        else:
            self.templates = [self.composite_template]
        
        print("✓ Enhanced model loaded successfully")
        return True
    
    def compute_similarity(self, roi: np.ndarray) -> Tuple[float, Dict]:
        """Compute similarity score between ROI and trained model"""
        if not self.model_loaded:
            return 0.0, {}
        
        # Resize to standard
        standard_size = (400, 250)
        try:
            resized = cv2.resize(roi, standard_size)
        except:
            return 0.0, {}
        
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # 1. Color histogram similarity
        color_hist = []
        for i in range(3):
            hist = cv2.calcHist([resized], [i], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            color_hist.extend(hist)
        color_hist = np.array(color_hist)
        
        # Z-score based similarity (accounts for variance)
        color_diff = np.abs(color_hist - self.avg_color_hist)
        color_z = color_diff / (self.std_color_hist + 1e-6)
        color_score = np.exp(-np.mean(color_z) / 2)  # Gaussian-like scoring
        
        # 2. Edge histogram similarity
        edges = cv2.Canny(gray, 50, 150)
        edge_hist = np.histogram(edges.flatten(), bins=16)[0]
        edge_hist = edge_hist / (edge_hist.sum() + 1e-6)
        
        edge_diff = np.abs(edge_hist - self.avg_edge_hist)
        edge_z = edge_diff / (self.std_edge_hist + 1e-6)
        edge_score = np.exp(-np.mean(edge_z) / 2)
        
        # 3. Aspect ratio similarity
        aspect_ratio = roi.shape[1] / roi.shape[0]
        aspect_z = abs(aspect_ratio - self.avg_aspect_ratio) / (self.std_aspect_ratio + 1e-6)
        aspect_score = np.exp(-aspect_z / 2)
        
        # 4. Template matching (best match across all templates)
        template_scores = []
        for template in self.templates[:10]:  # Use up to 10 templates
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            template_scores.append(float(result.max()))
        template_score = max(template_scores) if template_scores else 0
        
        # Weighted combination
        total_score = (
            0.25 * color_score +
            0.20 * edge_score +
            0.20 * aspect_score +
            0.35 * max(0, template_score)  # Template matching is most reliable
        )
        
        details = {
            'color_score': color_score,
            'edge_score': edge_score,
            'aspect_score': aspect_score,
            'template_score': template_score,
            'aspect_ratio': aspect_ratio
        }
        
        return total_score, details
    
    def detect_in_frame(self, frame: np.ndarray, 
                        min_confidence: float = 0.45) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect ID cards in a frame
        Returns: List of (x, y, w, h, confidence)
        """
        if not self.model_loaded:
            return []
        
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if w < 150 or h < 80:
                continue
            if w > width * 0.9 or h > height * 0.9:
                continue
            
            # Filter by aspect ratio (ID cards are ~1.4-1.8)
            aspect = w / float(h)
            if aspect < 1.2 or aspect > 2.2:
                continue
            
            # Filter by area
            area = w * h
            if area < 20000:
                continue
            
            # Extract ROI and compute similarity
            roi = frame[y:y+h, x:x+w]
            score, details = self.compute_similarity(roi)
            
            if score > min_confidence:
                candidates.append((x, y, w, h, score))
        
        # Non-maximum suppression
        candidates = self._nms(candidates, overlap_thresh=0.3)
        
        return candidates
    
    def _nms(self, boxes: List[Tuple], overlap_thresh: float = 0.3) -> List[Tuple]:
        """Non-maximum suppression"""
        if not boxes:
            return []
        
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        keep = []
        
        while boxes:
            current = boxes.pop(0)
            keep.append(current)
            
            boxes = [b for b in boxes if self._iou(current, b) < overlap_thresh]
        
        return keep
    
    def _iou(self, box1: Tuple, box2: Tuple) -> float:
        """Compute IoU between two boxes"""
        x1, y1, w1, h1, _ = box1
        x2, y2, w2, h2, _ = box2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = w1 * h1 + w2 * h2 - inter
        
        return inter / union if union > 0 else 0


def main():
    """Main training script"""
    import sys
    
    # Default paths
    training_dir = "training_data"
    model_dir = "models"
    
    # Check for command line args
    if len(sys.argv) > 1:
        training_dir = sys.argv[1]
    if len(sys.argv) > 2:
        model_dir = sys.argv[2]
    
    print("\n" + "="*60)
    print("🚀 STARTING ENHANCED ID CARD TRAINING")
    print("="*60)
    print(f"  Training data: {training_dir}")
    print(f"  Model output:  {model_dir}")
    
    # Create trainer and run
    trainer = EnhancedIDCardTrainer(training_dir, model_dir)
    success = trainer.train(augment_per_image=20)
    
    if success:
        print("\n✅ Training complete!")
        print("\nTo test the detector, run:")
        print("  python test_detector.py")
    else:
        print("\n❌ Training failed")


if __name__ == "__main__":
    main()