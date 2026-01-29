"""
Professional Attendance Kiosk - Google Wallet Style UI
Clean, modern design with animated card frame and real-time feedback
"""

import cv2
import numpy as np
import pytesseract
from datetime import datetime
import sqlite3
import re
import time
import math

class ProKiosk:
    def __init__(self, course_id="MSC_CS"):
        self.course_id = course_id
        self.db_path = "attendance.db"
        self.last_detected = {}
        self.cooldown = 4
        self._init_database()
        
        # Animation state
        self.animation_frame = 0
        self.scan_progress = 0
        self.state = "waiting"  # waiting, detected, scanning, success, already_marked
        self.state_start_time = 0
        self.detected_student = None
        
        # Colors (BGR format)
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (128, 128, 128)
        self.LIGHT_GRAY = (200, 200, 200)
        self.DARK_BG = (30, 30, 30)
        self.BLUE = (255, 180, 0)  # Accent blue
        self.GREEN = (100, 200, 100)
        self.SUCCESS_GREEN = (80, 220, 80)
        self.ORANGE = (0, 165, 255)
        
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY, name TEXT, last_seen DATETIME
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL, course_id TEXT NOT NULL,
                date DATE NOT NULL, timestamp DATETIME NOT NULL,
                status TEXT DEFAULT 'present'
            )
        """)
        conn.commit()
        conn.close()

    def draw_rounded_rect(self, img, pt1, pt2, color, thickness, radius):
        """Draw a rounded rectangle"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        if thickness < 0:
            # Filled rounded rectangle
            cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
            cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
            cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
            cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
            cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
        else:
            # Outline only
            cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
            cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
            cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
            cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
            cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
            cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

    def draw_card_frame(self, frame, x, y, w, h, progress=0):
        """Draw Google-style card scanning frame with corner brackets"""
        corner_len = 40
        thickness = 4
        
        # Animated color based on state
        if self.state == "waiting":
            color = self.WHITE
        elif self.state == "detected":
            color = self.ORANGE
        elif self.state == "scanning":
            # Pulsing blue during scan
            pulse = int(127 + 127 * math.sin(self.animation_frame * 0.2))
            color = (pulse, 180, 0)
        elif self.state in ["success", "already_marked"]:
            color = self.SUCCESS_GREEN
        else:
            color = self.WHITE
        
        # Top-left corner
        cv2.line(frame, (x, y), (x + corner_len, y), color, thickness)
        cv2.line(frame, (x, y), (x, y + corner_len), color, thickness)
        
        # Top-right corner
        cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, thickness)
        cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, thickness)
        
        # Bottom-left corner
        cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, thickness)
        cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, thickness)
        
        # Bottom-right corner
        cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)
        
        # Scanning line animation
        if self.state == "scanning" and progress > 0:
            scan_y = int(y + (h * progress))
            cv2.line(frame, (x + 10, scan_y), (x + w - 10, scan_y), self.BLUE, 2)

    def draw_ui(self, frame):
        """Draw the complete UI overlay"""
        height, width = frame.shape[:2]
        
        # Semi-transparent dark overlay on edges (vignette effect)
        overlay = frame.copy()
        
        # Card frame position (center, ID card aspect ratio ~1.6)
        card_w = int(width * 0.55)
        card_h = int(card_w / 1.6)
        card_x = (width - card_w) // 2
        card_y = (height - card_h) // 2
        
        # Darken area outside card frame
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (card_x, card_y), (card_x + card_w, card_y + card_h), 255, -1)
        
        dark_overlay = np.zeros_like(frame)
        frame_darkened = cv2.addWeighted(frame, 0.4, dark_overlay, 0.6, 0)
        
        # Apply mask - keep card area bright
        mask_3ch = cv2.merge([mask, mask, mask])
        frame = np.where(mask_3ch == 255, frame, frame_darkened)
        
        # Draw card frame with corners
        self.draw_card_frame(frame, card_x, card_y, card_w, card_h, self.scan_progress)
        
        # Top instruction panel
        panel_h = 100
        cv2.rectangle(frame, (0, 0), (width, panel_h), self.DARK_BG, -1)
        
        # Title
        title = "Student Attendance"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        cv2.putText(frame, title, ((width - title_size[0]) // 2, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.WHITE, 2)
        
        # Subtitle based on state
        if self.state == "waiting":
            subtitle = "Position your Student ID within the frame"
            sub_color = self.LIGHT_GRAY
        elif self.state == "detected":
            subtitle = "ID Card detected - Hold still..."
            sub_color = self.ORANGE
        elif self.state == "scanning":
            subtitle = "Scanning ID Card..."
            sub_color = self.BLUE
        elif self.state == "success":
            subtitle = f"Welcome, {self.detected_student['name']}!"
            sub_color = self.SUCCESS_GREEN
        elif self.state == "already_marked":
            subtitle = "Already marked present today"
            sub_color = self.ORANGE
        else:
            subtitle = "Position your Student ID within the frame"
            sub_color = self.LIGHT_GRAY
            
        sub_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        cv2.putText(frame, subtitle, ((width - sub_size[0]) // 2, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, sub_color, 1)
        
        # Bottom panel
        bottom_panel_h = 80
        cv2.rectangle(frame, (0, height - bottom_panel_h), (width, height), self.DARK_BG, -1)
        
        # Course info
        course_text = f"Course: {self.course_id}"
        cv2.putText(frame, course_text, (30, height - 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.LIGHT_GRAY, 1)
        
        # Date/time
        now = datetime.now()
        time_text = now.strftime("%H:%M:%S")
        date_text = now.strftime("%d %b %Y")
        cv2.putText(frame, date_text, (width - 150, height - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.LIGHT_GRAY, 1)
        cv2.putText(frame, time_text, (width - 150, height - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.WHITE, 1)
        
        # Quit instruction
        cv2.putText(frame, "Press Q to exit", (30, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.GRAY, 1)
        
        # Success checkmark animation
        if self.state == "success":
            self.draw_success_overlay(frame, width, height)
        elif self.state == "already_marked":
            self.draw_already_marked_overlay(frame, width, height)
        
        return frame, (card_x, card_y, card_w, card_h)

    def draw_success_overlay(self, frame, width, height):
        """Draw success checkmark overlay"""
        # Semi-transparent green overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (width//4, height//3), (3*width//4, 2*height//3), 
                     self.SUCCESS_GREEN, -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Checkmark circle
        center = (width // 2, height // 2 - 20)
        cv2.circle(frame, center, 50, self.SUCCESS_GREEN, -1)
        
        # Checkmark
        pts = np.array([
            [center[0] - 25, center[1]],
            [center[0] - 8, center[1] + 20],
            [center[0] + 25, center[1] - 15]
        ], np.int32)
        cv2.polylines(frame, [pts], False, self.WHITE, 6)
        
        # Student info
        if self.detected_student:
            id_text = f"ID: {self.detected_student['student_id']}"
            name_text = self.detected_student['name']
            
            cv2.putText(frame, "ATTENDANCE MARKED", 
                       (width//2 - 150, height//2 + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.WHITE, 2)
            cv2.putText(frame, id_text, 
                       (width//2 - 80, height//2 + 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.WHITE, 1)

    def draw_already_marked_overlay(self, frame, width, height):
        """Draw already marked overlay"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (width//4, height//3), (3*width//4, 2*height//3), 
                     self.ORANGE, -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Info icon
        center = (width // 2, height // 2 - 20)
        cv2.circle(frame, center, 50, self.ORANGE, -1)
        cv2.putText(frame, "i", (center[0] - 12, center[1] + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, self.WHITE, 3)
        
        cv2.putText(frame, "ALREADY CHECKED IN", 
                   (width//2 - 140, height//2 + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.WHITE, 2)

    def detect_id_in_region(self, frame, roi_bounds):
        """Detect ID card within the specified region - FIXED to match FastKiosk"""
        x, y, w, h = roi_bounds
        roi = frame[y:y+h, x:x+w]
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Match FastKiosk's edge detection parameters
        edges = cv2.Canny(blurred, 50, 150)
        
        # NO dilation - it was merging contours and breaking detection
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_rect = None
        max_area = 0
        
        for contour in contours:
            cx, cy, cw, ch = cv2.boundingRect(contour)
            area = cw * ch
            aspect = cw / float(ch) if ch > 0 else 0
            
            # Stricter criteria matching FastKiosk
            if (cw > 200 and ch > 120 and      # Reasonable minimum for ID card
                area > 30000 and                # Meaningful area
                1.4 < aspect < 1.9 and          # ID card aspect ratio
                area > max_area):               # Find largest match
                
                best_rect = (x + cx, y + cy, cw, ch)
                max_area = area
        
        if best_rect:
            print(f"✓ ID detected: {best_rect[2]}x{best_rect[3]}, area={max_area}")
            return True, best_rect
        
        return False, None

    def extract_text(self, roi):
        """Extract text from ID region"""
        height, width = roi.shape[:2]
        if width < 600:
            scale = 600 / width
            roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if np.mean(thresh) < 127:
            thresh = cv2.bitwise_not(thresh)
        
        text = ""
        for config in ['--psm 6', '--psm 11', '--psm 3']:
            attempt = pytesseract.image_to_string(thresh, config=config)
            if re.search(r'\d{8}', attempt):
                text = attempt
                break
            if len(attempt) > len(text):
                text = attempt
        
        return text

    def parse_id(self, text):
        """Parse student info from OCR text"""
        student_id = re.search(r'(\d{8})', text)
        name = re.search(r'([A-Z]{3,}\s+[A-Z]{3,})', text)
        
        if student_id:
            found_name = name.group(1) if name else "Student"
            # Clean up name
            if 'MASTER' in found_name:
                found_name = found_name.split('MASTER')[0].strip()
            return {
                'student_id': student_id.group(1),
                'name': found_name if found_name else "Student"
            }
        return None

    def mark_attendance(self, student_id, name):
        """Mark attendance in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now()
        today = now.strftime('%Y-%m-%d')
        timestamp = now.isoformat()
        
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
                INSERT INTO attendance (student_id, course_id, date, timestamp, status)
                VALUES (?, ?, ?, ?, 'present')
            """, (student_id, self.course_id, today, timestamp))
            conn.commit()
            conn.close()
            return True
        
        conn.close()
        return False

    def run(self):
        """Main kiosk loop"""
        print("\n" + "="*60)
        print("🎓 PROFESSIONAL ATTENDANCE KIOSK")
        print("="*60)
        print(f"Course: {self.course_id}")
        print("Press Q to quit")
        print("="*60 + "\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        cv2.namedWindow("Attendance", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Attendance", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        detection_start = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = time.time()
            self.animation_frame += 1
            
            # Draw UI and get card region
            frame, card_bounds = self.draw_ui(frame)
            
            # State machine
            if self.state == "waiting":
                detected, detection_box = self.detect_id_in_region(frame, card_bounds)
                if detected:
                    self.state = "detected"
                    detection_start = current_time
                    print("📷 ID Card detected - hold still for 1 second...")
                    
            elif self.state == "detected":
                detected, _ = self.detect_id_in_region(frame, card_bounds)
                if not detected:
                    self.state = "waiting"
                    print("⚠️ Lost detection - reposition ID")
                elif current_time - detection_start > 1.0:  # Hold for 1 second
                    self.state = "scanning"
                    self.scan_progress = 0
                    print("🔍 Scanning ID card...")
                    
            elif self.state == "scanning":
                self.scan_progress += 0.03  # Slower scan for better OCR
                
                if self.scan_progress >= 1.0:
                    # Process the ID
                    x, y, w, h = card_bounds
                    roi = frame[y:y+h, x:x+w]
                    
                    # Save for debugging
                    cv2.imwrite('pro_captured.jpg', roi)
                    print("📸 Captured image saved to pro_captured.jpg")
                    
                    text = self.extract_text(roi)
                    print(f"📝 OCR Result: {text[:150]}...")  # Print first 150 chars
                    
                    student = self.parse_id(text)
                    
                    if student:
                        self.detected_student = student
                        student_id = student['student_id']
                        
                        last = self.last_detected.get(student_id, 0)
                        if current_time - last > self.cooldown:
                            if self.mark_attendance(student_id, student['name']):
                                self.state = "success"
                                print(f"✅ Marked: {student_id} - {student['name']}")
                            else:
                                self.state = "already_marked"
                                print(f"ℹ️ Already marked: {student_id}")
                            self.last_detected[student_id] = current_time
                        else:
                            self.state = "already_marked"
                    else:
                        print("❌ Could not read ID - try again")
                        self.state = "waiting"
                    
                    self.state_start_time = current_time
                    self.scan_progress = 0
                    
            elif self.state in ["success", "already_marked"]:
                if current_time - self.state_start_time > 3.0:
                    self.state = "waiting"
                    self.scan_progress = 0
            
            cv2.imshow("Attendance", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute("""
            SELECT COUNT(*) FROM attendance WHERE course_id = ? AND date = ?
        """, (self.course_id, today))
        count = cursor.fetchone()[0]
        conn.close()
        
        print(f"\n📊 Session Complete: {count} students marked present\n")


if __name__ == "__main__":
    import sys
    course = sys.argv[1] if len(sys.argv) > 1 else "MSC_CS"
    ProKiosk(course).run()