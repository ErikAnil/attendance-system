"""
Simplified Live Kiosk - Guaranteed to detect ID cards
Uses very permissive detection settings
"""

import cv2
import numpy as np
import pytesseract
from datetime import datetime
import sqlite3
import re
import time

class SimpleKiosk:
    def __init__(self, course_id="MSC_CS"):
        self.course_id = course_id
        self.db_path = "attendance.db"
        self.last_detected = {}
        self.cooldown = 5  # seconds
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
    
    def detect_rectangles(self, frame):
        """Very simple rectangle detection - guaranteed to find cards"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection with VERY sensitive settings
        edges = cv2.Canny(blurred, 30, 100)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangles = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Very permissive filters
            area = w * h
            aspect_ratio = w / float(h) if h > 0 else 0
            
            # More specific for ID cards
            if (200 < w < 800 and
                120 < h < 400 and
                area > 30000 and
                1.3 < aspect_ratio < 2.0):
                
                rectangles.append((x, y, w, h))
        
        # Sort by size (largest first)
        rectangles.sort(key=lambda r: r[2] * r[3], reverse=True)
        
        return rectangles
    
    def extract_text(self, roi):
        """Extract text from region"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Threshold
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR
        text = pytesseract.image_to_string(thresh, config='--psm 6')
        return text
    
    def parse_maynooth_id(self, text):
        """Parse Maynooth ID from text"""
        # 8-digit student ID
        student_id = re.search(r'\b(\d{8})\b', text)
        
        # Name (all caps)
        name = re.search(r'\b([A-Z]{2,}\s+[A-Z]{2,})\b', text)
        
        if student_id:
            return {
                'student_id': student_id.group(1),
                'name': name.group(1) if name else "Unknown"
            }
        return None
    
    def mark_attendance(self, student_id, name):
        """Mark student present"""
        now = datetime.now()
        today = now.strftime('%Y-%m-%d')
        timestamp = now.isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update student
        cursor.execute("""
            INSERT OR REPLACE INTO students (student_id, name, last_seen)
            VALUES (?, ?, ?)
        """, (student_id, name, timestamp))
        
        # Check if already marked
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
            
            print(f"\n✓ MARKED PRESENT: {student_id} - {name}")
            return True
        else:
            conn.close()
            print(f"\nℹ Already marked: {student_id} - {name}")
            return False
    
    def run(self):
        """Main loop"""
        print("\n" + "="*60)
        print("🎥 SIMPLE LIVE KIOSK")
        print("="*60)
        print("\nHold your ID card in front of camera")
        print("Press SPACEBAR to manually capture")
        print("Press 'Q' to quit\n")
        print("="*60 + "\n")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        cv2.namedWindow("Attendance Kiosk", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Attendance Kiosk", 1280, 720)
        
        print("✓ Camera started\n")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Mirror frame
            #frame = cv2.flip(frame, 1)
            
            height, width = frame.shape[:2]
            
            # Detect rectangles
            rectangles = self.detect_rectangles(frame)
            
            # Only draw the LARGEST rectangle
            if rectangles:
                x, y, w, h = rectangles[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(frame, f"ID Card: {w}x{h}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Auto-process every 30 frames
            if rectangles and frame_count % 30 == 0:
                x, y, w, h = rectangles[0]
                roi = frame[y:y+h, x:x+w]
                
                try:
                    text = self.extract_text(roi)
                    student_info = self.parse_maynooth_id(text)
                    
                    if student_info:
                        student_id = student_info['student_id']
                        name = student_info['name']
                        
                        current_time = time.time()
                        last_time = self.last_detected.get(student_id, 0)
                        
                        if current_time - last_time > self.cooldown:
                            if self.mark_attendance(student_id, name):
                                # Show success
                                cv2.rectangle(frame, (0, height//3), (width, 2*height//3), 
                                            (0, 255, 0), -1)
                                cv2.putText(frame, "ATTENDANCE MARKED!", 
                                           (width//2 - 200, height//2),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                            
                            self.last_detected[student_id] = current_time
                except Exception as e:
                    print(f"Error: {e}")
            
            # UI
            cv2.rectangle(frame, (0, 0), (width, 60), (0, 0, 0), -1)
            cv2.putText(frame, f"ATTENDANCE KIOSK - {self.course_id}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.rectangle(frame, (0, height-40), (width, height), (0, 0, 0), -1)
            cv2.putText(frame, f"Detected: {len(rectangles)} | SPACEBAR=capture | Q=quit", 
                       (20, height-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Attendance Kiosk", frame)
            
            # Keyboard handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord(' '):  # SPACEBAR - manual capture
                if rectangles:
                    x, y, w, h = rectangles[0]
                    roi = frame[y:y+h, x:x+w]
                    
                    cv2.imwrite('captured_id.jpg', roi)
                    print("\n📸 Saved captured_id.jpg")
                    
                    text = self.extract_text(roi)
                    print(f"\n{'='*60}")
                    print("MANUAL CAPTURE - OCR TEXT:")
                    print('='*60)
                    print(text)
                    print('='*60)
                    
                    student_info = self.parse_maynooth_id(text)
                    if student_info:
                        print(f"\n✓ Student ID: {student_info['student_id']}")
                        print(f"✓ Name: {student_info['name']}")
                        self.mark_attendance(student_info['student_id'], student_info['name'])
                    else:
                        print("\n❌ Could not parse student info from OCR")
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("✓ KIOSK STOPPED")
        print("="*60)

if __name__ == "__main__":
    import sys
    course_id = sys.argv[1] if len(sys.argv) > 1 else "MSC_CS"
    
    kiosk = SimpleKiosk(course_id)
    kiosk.run()