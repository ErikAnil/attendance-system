"""
ID Card Detection with Limited Training Data (3 samples)
Uses data augmentation + transfer learning for robust detection
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Dict
import random

class IDCardDetectorTrainer:
    """
    Train an ID card detector with minimal samples using data augmentation
    """
    
    def __init__(self, save_dir: str = "models"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.template = None
        self.card_features = {}
        
    def augment_image(self, image: np.ndarray, n_augmentations: int = 50) -> List[np.ndarray]:
        """
        Generate multiple augmented versions from a single image
        This compensates for having only 3 training samples
        """
        augmented = [image.copy()]
        
        for _ in range(n_augmentations):
            aug_img = image.copy()
            
            # Random rotation (-15 to +15 degrees)
            angle = random.uniform(-15, 15)
            h, w = aug_img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            aug_img = cv2.warpAffine(aug_img, M, (w, h), 
                                     borderMode=cv2.BORDER_REPLICATE)
            
            # Random scaling (0.85 to 1.15)
            scale = random.uniform(0.85, 1.15)
            new_w, new_h = int(w * scale), int(h * scale)
            aug_img = cv2.resize(aug_img, (new_w, new_h))
            
            # Pad or crop to original size
            if scale > 1.0:
                # Crop to original size
                start_x = (new_w - w) // 2
                start_y = (new_h - h) // 2
                aug_img = aug_img[start_y:start_y+h, start_x:start_x+w]
            else:
                # Pad to original size
                pad_x = (w - new_w) // 2
                pad_y = (h - new_h) // 2
                aug_img = cv2.copyMakeBorder(aug_img, pad_y, h-new_h-pad_y, 
                                            pad_x, w-new_w-pad_x, 
                                            cv2.BORDER_REPLICATE)
            
            # Random brightness (0.7 to 1.3)
            brightness = random.uniform(0.7, 1.3)
            aug_img = cv2.convertScaleAbs(aug_img, alpha=brightness, beta=0)
            
            # Random blur (simulate focus issues)
            if random.random() > 0.5:
                kernel_size = random.choice([3, 5])
                aug_img = cv2.GaussianBlur(aug_img, (kernel_size, kernel_size), 0)
            
            # Random noise
            if random.random() > 0.7:
                noise = np.random.randint(-20, 20, aug_img.shape, dtype=np.int16)
                aug_img = np.clip(aug_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Random perspective transform
            if random.random() > 0.6:
                pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
                offset = 30
                pts2 = np.float32([
                    [random.randint(0, offset), random.randint(0, offset)],
                    [w - random.randint(0, offset), random.randint(0, offset)],
                    [random.randint(0, offset), h - random.randint(0, offset)],
                    [w - random.randint(0, offset), h - random.randint(0, offset)]
                ])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                aug_img = cv2.warpPerspective(aug_img, M, (w, h),
                                             borderMode=cv2.BORDER_REPLICATE)
            
            augmented.append(aug_img)
        
        return augmented
    
    def extract_features(self, image: np.ndarray) -> Dict:
        """
        Extract robust features from ID card image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Color histogram (robust to lighting changes)
        color_hist = []
        for i in range(3):
            hist = cv2.calcHist([image], [i], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            color_hist.extend(hist)
        
        # Edge features (card shape)
        edges = cv2.Canny(gray, 50, 150)
        edge_hist = np.histogram(edges.flatten(), bins=10)[0]
        edge_hist = edge_hist / (edge_hist.sum() + 1e-6)
        
        # Aspect ratio
        aspect_ratio = image.shape[1] / image.shape[0]
        
        # SIFT features for template matching
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        return {
            'color_hist': np.array(color_hist),
            'edge_hist': edge_hist,
            'aspect_ratio': aspect_ratio,
            'sift_descriptors': descriptors,
            'sift_keypoints': keypoints,
            'size': image.shape[:2]
        }
    
    def train_from_samples(self, sample_images: List[np.ndarray]):
        """
        Train detector from 3 sample ID card images
        """
        print(f"Training with {len(sample_images)} sample images...")
        
        all_features = []
        all_augmented = []
        
        # Process each sample
        for idx, sample in enumerate(sample_images):
            print(f"\nProcessing sample {idx+1}/{len(sample_images)}...")
            
            # Generate augmented versions
            augmented = self.augment_image(sample, n_augmentations=50)
            all_augmented.extend(augmented)
            print(f"  Generated {len(augmented)} augmented images")
            
            # Extract features from augmented images
            for aug_img in augmented:
                features = self.extract_features(aug_img)
                all_features.append(features)
        
        # Compute average features (our "model")
        print("\nComputing average features...")
        
        # Average color histogram
        avg_color_hist = np.mean([f['color_hist'] for f in all_features], axis=0)
        
        # Average edge histogram
        avg_edge_hist = np.mean([f['edge_hist'] for f in all_features], axis=0)
        
        # Average aspect ratio
        avg_aspect_ratio = np.mean([f['aspect_ratio'] for f in all_features])
        
        # Store all SIFT descriptors for matching
        all_sift = [f['sift_descriptors'] for f in all_features if f['sift_descriptors'] is not None]
        
        self.card_features = {
            'avg_color_hist': avg_color_hist,
            'avg_edge_hist': avg_edge_hist,
            'avg_aspect_ratio': avg_aspect_ratio,
            'all_sift_descriptors': all_sift,
            'sample_size': sample_images[0].shape[:2]
        }
        
        # Use first sample as template
        self.template = cv2.cvtColor(sample_images[0], cv2.COLOR_BGR2GRAY)
        
        print(f"\n✓ Training complete!")
        print(f"  Total training images: {len(all_augmented)}")
        print(f"  Average aspect ratio: {avg_aspect_ratio:.2f}")
        
        # Save model
        self.save_model()
    
    def save_model(self):
        """Save trained model"""
        model_path = self.save_dir / "id_card_model.npz"
        
        np.savez(
            model_path,
            template=self.template,
            avg_color_hist=self.card_features['avg_color_hist'],
            avg_edge_hist=self.card_features['avg_edge_hist'],
            avg_aspect_ratio=self.card_features['avg_aspect_ratio'],
            sample_size=self.card_features['sample_size']
        )
        
        # Save SIFT descriptors separately (they're lists)
        import pickle
        sift_path = self.save_dir / "sift_descriptors.pkl"
        with open(sift_path, 'wb') as f:
            pickle.dump(self.card_features['all_sift_descriptors'], f)
        
        print(f"✓ Model saved to {model_path}")


class IDCardDetector:
    """
    Detect ID cards in images using the trained model
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.load_model()
    
    def load_model(self):
        """Load trained model"""
        model_path = self.model_dir / "id_card_model.npz"
        sift_path = self.model_dir / "sift_descriptors.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Train first!")
        
        data = np.load(model_path)
        self.template = data['template']
        self.avg_color_hist = data['avg_color_hist']
        self.avg_edge_hist = data['avg_edge_hist']
        self.avg_aspect_ratio = float(data['avg_aspect_ratio'])
        self.sample_size = tuple(data['sample_size'])
        
        if sift_path.exists():
            import pickle
            with open(sift_path, 'rb') as f:
                self.all_sift_descriptors = pickle.load(f)
        else:
            self.all_sift_descriptors = []
        
        print("✓ Model loaded successfully")
    
    def compute_similarity(self, image: np.ndarray) -> float:
        """
        Compute similarity score between image and trained model
        Returns score between 0 and 1 (higher = more similar)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Color histogram similarity
        color_hist = []
        for i in range(3):
            hist = cv2.calcHist([image], [i], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            color_hist.extend(hist)
        color_hist = np.array(color_hist)
        
        color_similarity = 1 - np.sum(np.abs(color_hist - self.avg_color_hist)) / 2
        
        # Edge similarity
        edges = cv2.Canny(gray, 50, 150)
        edge_hist = np.histogram(edges.flatten(), bins=10)[0]
        edge_hist = edge_hist / (edge_hist.sum() + 1e-6)
        
        edge_similarity = 1 - np.sum(np.abs(edge_hist - self.avg_edge_hist)) / 2
        
        # Aspect ratio similarity
        aspect_ratio = image.shape[1] / image.shape[0]
        aspect_diff = abs(aspect_ratio - self.avg_aspect_ratio)
        aspect_similarity = max(0, 1 - aspect_diff)
        
        # Template matching similarity
        resized = cv2.resize(gray, (self.template.shape[1], self.template.shape[0]))
        result = cv2.matchTemplate(resized, self.template, cv2.TM_CCOEFF_NORMED)
        template_similarity = float(result.max())
        
        # Weighted combination
        total_similarity = (
            0.3 * color_similarity +
            0.2 * edge_similarity +
            0.2 * aspect_similarity +
            0.3 * template_similarity
        )
        
        return total_similarity
    
    def detect_id_cards(self, frame: np.ndarray, threshold: float = 0.5) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect ID cards in frame
        Returns list of (x, y, w, h, confidence)
        """
        candidates = []
        
        # Method 1: Contour-based detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check each contour
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Look for rectangular shapes
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(approx)
                
                # Filter by size and aspect ratio
                aspect_ratio = w / float(h)
                area = w * h
                
                if (area > 10000 and  # Minimum size
                    1.3 < aspect_ratio < 2.0 and  # ID card aspect ratio
                    w > 200 and h > 100):
                    
                    # Extract region and compute similarity
                    roi = frame[y:y+h, x:x+w]
                    similarity = self.compute_similarity(roi)
                    
                    if similarity > threshold:
                        candidates.append((x, y, w, h, similarity))
        
        # Non-maximum suppression to remove overlapping detections
        candidates = self.non_max_suppression(candidates)
        
        return candidates
    
    def non_max_suppression(self, boxes: List[Tuple[int, int, int, int, float]], 
                           overlap_thresh: float = 0.3) -> List[Tuple[int, int, int, int, float]]:
        """Remove overlapping bounding boxes"""
        if len(boxes) == 0:
            return []
        
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        keep = []
        
        while boxes:
            current = boxes.pop(0)
            keep.append(current)
            
            boxes = [
                box for box in boxes
                if self.compute_iou(current, box) < overlap_thresh
            ]
        
        return keep
    
    def compute_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Compute Intersection over Union"""
        x1, y1, w1, h1, _ = box1
        x2, y2, w2, h2, _ = box2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0


def create_training_dataset():
    """
    Helper function to prepare your 3 ID card images for training
    """
    print("ID Card Training Dataset Creator")
    print("=" * 60)
    print("\nYou need to provide 3 clear photos of ID cards.")
    print("Take photos ensuring:")
    print("  - Good lighting")
    print("  - ID card is clearly visible")
    print("  - Minimal background clutter")
    print("  - Various angles (straight, slightly tilted)")
    
    sample_paths = []
    
    for i in range(3):
        path = input(f"\nEnter path to ID card image {i+1}: ").strip()
        if Path(path).exists():
            sample_paths.append(path)
            print(f"✓ Image {i+1} loaded")
        else:
            print(f"✗ File not found: {path}")
            return None
    
    # Load images
    samples = []
    for path in sample_paths:
        img = cv2.imread(path)
        if img is not None:
            # Resize to standard size
            img = cv2.resize(img, (800, 500))
            samples.append(img)
    
    return samples


def main():
    """
    Main training and testing script
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Training: python id_detector.py train <img1> <img2> <img3>")
        print("  Testing:  python id_detector.py test <video_or_image>")
        return
    
    mode = sys.argv[1]
    
    if mode == "train":
        if len(sys.argv) < 5:
            print("Please provide 3 ID card images for training")
            print("Example: python id_detector.py train id1.jpg id2.jpg id3.jpg")
            return
        
        # Load training images
        samples = []
        for img_path in sys.argv[2:5]:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error loading {img_path}")
                return
            # Resize to consistent size
            img = cv2.resize(img, (800, 500))
            samples.append(img)
            print(f"✓ Loaded {img_path}")
        
        # Train
        trainer = IDCardDetectorTrainer()
        trainer.train_from_samples(samples)
        
        print("\n" + "="*60)
        print("Training complete! You can now use the detector.")
        print("="*60)
    
    elif mode == "test":
        if len(sys.argv) < 3:
            print("Please provide an image or video to test")
            return
        
        test_path = sys.argv[2]
        
        # Load detector
        detector = IDCardDetector()
        
        # Test on image or video
        if test_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Test on single image
            img = cv2.imread(test_path)
            detections = detector.detect_id_cards(img)
            
            # Draw detections
            for x, y, w, h, conf in detections:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, f"{conf:.2f}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            cv2.imshow("Detections", img)
            cv2.waitKey(0)
            print(f"Found {len(detections)} ID cards")
        
        else:
            # Test on video
            cap = cv2.VideoCapture(test_path)
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % 30 == 0:  # Process every 30 frames
                    detections = detector.detect_id_cards(frame)
                    
                    for x, y, w, h, conf in detections:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{conf:.2f}", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    cv2.imshow("Video Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
            
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()