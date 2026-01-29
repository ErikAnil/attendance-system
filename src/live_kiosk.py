"""
Live Camera Attendance Kiosk
Real-time ID card detection using MacBook camera
Emulates the iPad selfie camera setup used by professors
"""

import cv2
import numpy as np
from datetime import datetime
import sqlite3
from pathlib import Path
import threading
import queue
import time

class LiveAttendanceKiosk:
    """
    Live attendance kiosk that continuously monitors camera for ID cards
    """
    
    def __init__(self, course_id: str = "CS501", db_path: str = "attendance.db"):
        self.course_id = course_id
        self.db_path = db_path
        self.running = False
        self.detector = None
        self.processing_queue = queue.Queue()
        
        # Detection settings
        self.detection_cooldown = 5  # seconds between detections of same ID
        self.last_detected = {}  # Track last detection time per student
        self.confidence_threshold = 0.2
        
        # UI settings
        self.window_name = "Attendance Kiosk - Show Your ID"
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Initialize detector
        self._load_detector()
        self._init_database()
        
        print("🎥 Live Attendance Kiosk Initialized")
        print(f"📚 Course: {course_id}")
        print(f"💾 Database: {db_path}")
    
    def _load_detector(self):
        """Load the trained ID card detector"""
        try:
            import sys
            sys.path.insert(0, 'src')  # ADD THIS LINE
            from id_card_detector import IDCardDetector
            self.detector = IDCardDetector()
            print("✓ ID card detector loaded")
        except Exception as e:
            print(f"⚠ Warning: Could not load detector - {e}")
            print("  Will use basic detection")
            self.detector = None
    
    def _init_database(self):
        """Initialize database connection"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
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
                confidence REAL,
                status TEXT DEFAULT 'present'
            )
        """)
        
        conn.commit()
        conn.close()
        print("✓ Database ready")
    
    def extract_text_from_roi(self, roi):
        """Extract text from ID card region using OCR"""
        try:
            import pytesseract
            
            # Preprocess for better OCR
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Threshold
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Extract text
            text = pytesseract.image_to_string(thresh, config='--psm 6')
            
            return text
        except Exception as e:
            print(f"OCR error: {e}")
            return ""
    
    def parse_maynooth_id(self, text: str):
        """Parse Maynooth University ID from text"""
        import re
        
        # Look for 8-digit student ID
        student_id_match = re.search(r'\b(\d{8})\b', text)
        
        # Look for name (all caps, 2+ words)
        name_match = re.search(r'\b([A-Z]{2,}\s+[A-Z]{2,})\b', text)
        
        if student_id_match:
            student_id = student_id_match.group(1)
            name = name_match.group(1) if name_match else "Unknown"
            
            return {
                'student_id': student_id,
                'name': name
            }
        
        return None
    
    def detect_id_card_simple(self, frame):
        """Simple ID card detection using contours"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                area = w * h
                
                # ID card characteristics
                if (area > 20000 and 
                    1.3 < aspect_ratio < 2.0 and 
                    w > 200 and h > 100):
                    
                    detections.append((x, y, w, h, 0.7))
        
        return detections
    
    def mark_attendance(self, student_id: str, name: str, confidence: float):
        """Mark student as present"""
        now = datetime.now()
        today = now.strftime('%Y-%m-%d')
        timestamp = now.isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update or insert student
        cursor.execute("""
            INSERT OR REPLACE INTO students (student_id, name, last_seen)
            VALUES (?, ?, ?)
        """, (student_id, name, timestamp))
        
        # Check if already marked today
        cursor.execute("""
            SELECT COUNT(*) FROM attendance 
            WHERE student_id = ? AND course_id = ? AND date = ?
        """, (student_id, self.course_id, today))
        
        if cursor.fetchone()[0] == 0:
            # Mark attendance
            cursor.execute("""
                INSERT INTO attendance 
                (student_id, course_id, date, timestamp, confidence, status)
                VALUES (?, ?, ?, ?, ?, 'present')
            """, (student_id, self.course_id, today, timestamp, confidence))
            
            conn.commit()
            conn.close()
            
            print(f"✓ Marked present: {student_id} - {name} (confidence: {confidence:.2f})")
            return True
        else:
            conn.close()
            print(f"ℹ Already marked: {student_id} - {name}")
            return False
    
    def draw_ui(self, frame, detections, fps):
        """Draw user interface on frame"""
        height, width = frame.shape[:2]
        
        # Semi-transparent overlay for header
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Title
        cv2.putText(frame, "ATTENDANCE KIOSK", (20, 35),
                   self.font, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, f"Course: {self.course_id}", (20, 65),
                   self.font, 0.6, (200, 200, 200), 1)
        
        # Instructions in center (if no detection)
        if not detections:
            instruction_text = "Please show your Student ID to the camera"
            text_size = cv2.getTextSize(instruction_text, self.font, 1.0, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height // 2
            
            # Background for text
            cv2.rectangle(frame, 
                         (text_x - 20, text_y - 40),
                         (text_x + text_size[0] + 20, text_y + 10),
                         (0, 0, 0), -1)
            
            cv2.putText(frame, instruction_text, (text_x, text_y),
                       self.font, 1.0, (255, 255, 255), 2)
            
            # Arrow or icon
            cv2.putText(frame, "⬇", (width // 2 - 20, text_y + 60),
                       self.font, 2.0, (100, 200, 100), 3)
        
        # Draw detections
        for x, y, w, h, conf in detections:
            # Green box around detected ID
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Confidence badge
            label = f"ID Card: {conf:.0%}"
            label_size = cv2.getTextSize(label, self.font, 0.7, 2)[0]
            
            cv2.rectangle(frame,
                         (x, y - 35),
                         (x + label_size[0] + 10, y),
                         (0, 255, 0), -1)
            
            cv2.putText(frame, label, (x + 5, y - 10),
                       self.font, 0.7, (0, 0, 0), 2)
        
        # Footer with stats
        cv2.rectangle(frame, (0, height - 40), (width, height), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, height - 15),
                   self.font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Press 'Q' to quit | 'S' to save screenshot", 
                   (width - 400, height - 15),
                   self.font, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def show_success_message(self, frame, student_id, name):
        """Show success animation when student is detected"""
        height, width = frame.shape[:2]
        
        # Green overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, height//3), (width, 2*height//3), (0, 255, 0), -1)
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        
        # Success message
        cv2.putText(frame, "✓ ATTENDANCE MARKED", 
                   (width//2 - 200, height//2 - 30),
                   self.font, 1.2, (255, 255, 255), 3)
        
        cv2.putText(frame, f"{student_id} - {name}", 
                   (width//2 - 150, height//2 + 20),
                   self.font, 0.9, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        """Main kiosk loop"""
        print("\n" + "="*60)
        print("🎥 STARTING LIVE ATTENDANCE KIOSK")
        print("="*60)
        print("\nInstructions:")
        print("  1. Position MacBook like an iPad on stand")
        print("  2. Students walk up and show ID to camera")
        print("  3. System auto-detects and marks attendance")
        print("  4. Press 'Q' to quit")
        print("\n" + "="*60 + "\n")
        
        # Open camera (0 = built-in camera)
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.running = True
        fps = 0
        fps_counter = 0
        fps_start_time = time.time()
        
        success_frame = None
        success_display_until = 0
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        
        print("✓ Camera started. Waiting for ID cards...\n")
        
        while self.running:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error: Failed to read from camera")
                break
            
            # Mirror frame (like selfie camera)
            #frame = cv2.flip(frame, 1)
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter >= 30:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            
           # Detect ID cards - FORCE SIMPLE DETECTION
            detections = self.detect_id_card_simple(frame)
            print(f"DEBUG: Found {len(detections)} detections")  # ADD DEBUG
            
            # Process detections
            current_time = time.time()
            
            for x, y, w, h, conf in detections:
                # Extract ID card region
                roi = frame[y:y+h, x:x+w]
                
                # Run OCR
                text = self.extract_text_from_roi(roi)
                student_info = self.parse_maynooth_id(text)
                
                if student_info:
                    student_id = student_info['student_id']
                    name = student_info['name']
                    
                    # Check cooldown
                    last_time = self.last_detected.get(student_id, 0)
                    
                    if current_time - last_time > self.detection_cooldown:
                        # Mark attendance
                        if self.mark_attendance(student_id, name, conf):
                            # Show success message
                            success_frame = self.show_success_message(
                                frame.copy(), student_id, name
                            )
                            success_display_until = current_time + 2  # Show for 2 seconds
                        
                        self.last_detected[student_id] = current_time
            
            # Show success message if active
            if success_frame is not None and current_time < success_display_until:
                display_frame = success_frame
            else:
                display_frame = self.draw_ui(frame, detections, fps)
                success_frame = None
            
            # Display
            cv2.imshow(self.window_name, display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\n🛑 Stopping kiosk...")
                break
            elif key == ord('s') or key == ord('S'):
                # Save screenshot
                filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.running = False
        
        print("\n" + "="*60)
        print("✓ KIOSK STOPPED")
        print("="*60)
        
        # Show summary
        self.print_summary()
    
    def print_summary(self):
        """Print attendance summary"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        cursor.execute("""
            SELECT COUNT(*) FROM attendance 
            WHERE course_id = ? AND date = ?
        """, (self.course_id, today))
        
        count = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT s.student_id, s.name, a.timestamp
            FROM attendance a
            JOIN students s ON a.student_id = s.student_id
            WHERE a.course_id = ? AND a.date = ?
            ORDER BY a.timestamp DESC
        """, (self.course_id, today))
        
        records = cursor.fetchall()
        conn.close()
        
        print(f"\n📊 Attendance Summary - {today}")
        print(f"Course: {self.course_id}")
        print(f"Total Present: {count}")
        print("\nRecent Check-ins:")
        print("-" * 60)
        
        for student_id, name, timestamp in records[:10]:
            time_str = timestamp.split('T')[1][:8]
            print(f"  {student_id:12} {name:20} {time_str}")
        
        if len(records) > 10:
            print(f"  ... and {len(records) - 10} more")


def main():
    """Run the live attendance kiosk"""
    import sys
    
    course_id = sys.argv[1] if len(sys.argv) > 1 else "CS501"
    
    print("\n" + "="*60)
    print("🎓 LIVE ATTENDANCE KIOSK")
    print("   Maynooth University")
    print("="*60)
    
    kiosk = LiveAttendanceKiosk(course_id=course_id)
    
    try:
        kiosk.run()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Session ended. Check attendance records in database.\n")


if __name__ == "__main__":
    main()