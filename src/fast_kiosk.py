"""
Optimized Live Kiosk - Fast and Accurate ID Detection
Press SPACEBAR when ready - instant capture and processing
"""

import cv2
import numpy as np
import pytesseract
from datetime import datetime
import sqlite3
import re
import time

class FastKiosk:
    def __init__(self, course_id="MSC_CS"):
        self.course_id = course_id
        self.db_path = "attendance.db"
        self.last_detected = {}
        self.cooldown = 3  # Reduced to 3 seconds
        self._init_database()
        
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT,
                last_seen DATETIME
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                course_id TEXT NOT NULL,
                date DATE NOT NULL,
                timestamp DATETIME NOT NULL,
                status TEXT DEFAULT 'present'
            )
        """)
        
        conn.commit()
        conn.close()
        print("✓ Database ready")
    
    def detect_id_card_center(self, frame):
        """
        Detect ID card in CENTER region only - faster and more accurate
        """
        height, width = frame.shape[:2]
        
        # Focus on center 60% of frame
        y_start = int(height * 0.2)
        y_end = int(height * 0.8)
        x_start = int(width * 0.2)
        x_end = int(width * 0.8)
        
        # Extract center region
        center_region = frame[y_start:y_end, x_start:x_end]
        
        gray = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_rect = None
        max_area = 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / float(h) if h > 0 else 0
            
            # Strict ID card criteria
            if (250 < w < 700 and
                150 < h < 350 and
                area > 40000 and
                1.4 < aspect_ratio < 1.9 and
                area > max_area):
                
                # Adjust coordinates back to full frame
                best_rect = (x + x_start, y + y_start, w, h)
                max_area = area
        
        return best_rect
    
    def extract_text_improved(self, roi):
        """Improved OCR with better preprocessing"""
        # Resize for better OCR (if too small)
        height, width = roi.shape[:2]
        if width < 600:
            scale = 600 / width
            roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Threshold
        _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR with better config for ID cards
        text = pytesseract.image_to_string(thresh, config='--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ')
        
        return text, thresh
    
    def parse_maynooth_id(self, text):
        """Parse Maynooth ID - improved patterns"""
        # 8-digit student ID
        student_id = re.search(r'(\d{8})', text)
        
        # Name - handle OCR errors
        # Look for 2+ uppercase words
        name_patterns = [
            r'\b([A-Z]{3,}\s+[A-Z]{3,})\b',  # ERIK ANIL
            r'\b([A-Z][A-Z]+\s+[A-Z][A-Z]+)\b',  # Handles minor OCR errors
        ]
        
        name = None
        for pattern in name_patterns:
            match = re.search(pattern, text)
            if match:
                name = match.group(1)
                # Clean up common OCR errors
                name = name.replace('0', 'O').replace('1', 'I')
                break
        
        if student_id:
            return {
                'student_id': student_id.group(1),
                'name': name if name else "Unknown"
            }
        return None
    
    def mark_attendance(self, student_id, name):
        """Mark student present"""
        now = datetime.now()
        today = now.strftime('%Y-%m-%d')
        timestamp = now.isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO students (student_id, name, last_seen)
            VALUES (?, ?, ?)
        """, (student_id, name, timestamp))
        
        cursor.execute("""
            SELECT COUNT(*) FROM attendance 
            WHERE student_id = ? AND course_id = ? AND date = ?
        """, (student_id, self.course_id, today))
        
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO attendance 
                (student_id, course_id, date, timestamp, status)
                VALUES (?, ?, ?, ?, 'present')
            """, (student_id, self.course_id, today, timestamp))
            
            conn.commit()
            conn.close()
            return True
        else:
            conn.close()
            return False
    
    def run(self):
        """Main loop - AUTOMATIC MODE"""
        print("\n" + "="*60)
        print("🎥 AUTOMATIC ATTENDANCE KIOSK")
        print("="*60)
        print("\nInstructions:")
        print("  1. Hold ID card in CENTER of frame")
        print("  2. System auto-detects and processes")
        print("  3. Wait for green SUCCESS message")
        print("  4. Press 'Q' to quit")
        print("\n" + "="*60 + "\n")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        # Set higher resolution for better OCR
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        cv2.namedWindow("Attendance Kiosk", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Attendance Kiosk", 1280, 720)
        
        print("✓ Camera started")
        print("\n💡 TIP: Hold ID steady in center. Auto-processes when detected.\n")
        
        success_message = None
        success_until = 0
        last_process_time = 0
        process_delay = 2  # Wait 2 seconds between auto-captures
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            height, width = frame.shape[:2]
            current_time = time.time()
            
            # Draw center guide box (where ID should be)
            guide_x = int(width * 0.25)
            guide_y = int(height * 0.30)
            guide_w = int(width * 0.50)
            guide_h = int(height * 0.40)
            
            cv2.rectangle(frame, (guide_x, guide_y), 
                         (guide_x + guide_w, guide_y + guide_h), 
                         (0, 255, 0), 2)
            cv2.putText(frame, "Hold ID here - Auto-detects", 
                       (guide_x + 10, guide_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Detect ID in center region
            detected = self.detect_id_card_center(frame)
            
            if detected:
                x, y, w, h = detected
                # Draw detected ID with thicker border
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 3)
                cv2.putText(frame, f"ID DETECTED - Processing...", 
                           (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # AUTO-PROCESS if enough time has passed
                if current_time - last_process_time > process_delay:
                    roi = frame[y:y+h, x:x+w]
                    
                    print("\n" + "="*60)
                    print("📸 AUTO-CAPTURING...")
                    print("="*60)
                    
                    # Save captured image
                    cv2.imwrite('captured_id.jpg', roi)
                    
                    # Process with improved OCR
                    text, processed = self.extract_text_improved(roi)
                    cv2.imwrite('captured_processed.jpg', processed)
                    
                    print("\n📝 OCR Text:")
                    print(text[:200])  # First 200 chars
                    print("\n" + "-"*60)
                    
                    # Parse
                    student_info = self.parse_maynooth_id(text)
                    
                    if student_info:
                        student_id = student_info['student_id']
                        name = student_info['name']
                        
                        print(f"✓ Student ID: {student_id}")
                        print(f"✓ Name: {name}")
                        
                        # Mark attendance
                        if self.mark_attendance(student_id, name):
                            print(f"\n✅ MARKED PRESENT: {student_id} - {name}")
                            success_message = f"{student_id} - {name}"
                            success_until = current_time + 3  # Show for 3 seconds
                        else:
                            print(f"\nℹ️  Already marked: {student_id} - {name}")
                            success_message = f"Already marked: {student_id}"
                            success_until = current_time + 3
                    else:
                        print("\n❌ Could not read student ID")
                    
                    print("="*60 + "\n")
                    last_process_time = current_time
            
            # Show success message if active
            if success_message and current_time < success_until:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, height//3), (width, 2*height//3), 
                            (0, 255, 0), -1)
                frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
                
                cv2.putText(frame, "✓ ATTENDANCE MARKED!", 
                           (width//2 - 250, height//2 - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                cv2.putText(frame, success_message, 
                           (width//2 - 200, height//2 + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Header
            cv2.rectangle(frame, (0, 0), (width, 70), (0, 0, 0), -1)
            cv2.putText(frame, f"AUTO ATTENDANCE KIOSK - {self.course_id}", 
                       (20, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            # Footer
            cv2.rectangle(frame, (0, height-50), (width, height), (0, 0, 0), -1)
            status = "ID DETECTED - Auto-processing..." if detected else "Waiting for ID card..."
            cv2.putText(frame, f"{status} | Q=Quit", 
                       (20, height-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Attendance Kiosk", frame)
            
            # Keyboard handling
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("✓ KIOSK STOPPED")
        print("="*60)
        
        # Show summary
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute("""
            SELECT COUNT(*) FROM attendance 
            WHERE course_id = ? AND date = ?
        """, (self.course_id, today))
        count = cursor.fetchone()[0]
        conn.close()
        
        print(f"\n📊 Today's Attendance: {count} students marked present")
        print(f"Course: {self.course_id}")
        print(f"Date: {today}\n")


if __name__ == "__main__":
    import sys
    course_id = sys.argv[1] if len(sys.argv) > 1 else "MSC_CS"
    
    kiosk = FastKiosk(course_id)
    kiosk.run()