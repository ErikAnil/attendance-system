"""
Demo Attendance System - Uses clear pre-captured images
Perfect for master's defense demonstration
"""

import cv2
import pytesseract
import numpy as np
import re
import sqlite3
from datetime import datetime
import glob

def process_clear_image(image_path, course_id="MSC_CS"):
    """Process a clear ID card image"""
    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print('='*60)
    
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Failed to load image")
        return None
    
    # Resize if needed
    height, width = img.shape[:2]
    if width < 800:
        scale = 800 / width
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhance
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Threshold
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert if needed
    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)
    
    # Save for review
    cv2.imwrite('demo_processed.jpg', thresh)
    
    # OCR
    text = pytesseract.image_to_string(thresh, config='--psm 6')
    
    print("\nOCR Output:")
    print(text)
    print("-"*60)
    
    # Parse
    student_id = re.search(r'(\d{8})', text)
    name = re.search(r'([A-Z]{3,}\s+[A-Z]{3,})', text)
    
    if student_id:
        result = {
            'student_id': student_id.group(1),
            'name': name.group(1) if name else "Unknown"
        }
        print(f"\n✅ Extracted:")
        print(f"   ID: {result['student_id']}")
        print(f"   Name: {result['name']}")
        
        # Mark attendance
        mark_attendance(result['student_id'], result['name'], course_id)
        return result
    else:
        print("\n❌ Could not extract student ID")
        return None

def mark_attendance(student_id, name, course_id):
    """Mark attendance in database"""
    conn = sqlite3.connect('attendance.db')
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
    """, (student_id, course_id, today))
    
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO attendance 
            (student_id, course_id, date, timestamp, status)
            VALUES (?, ?, ?, ?, 'present')
        """, (student_id, course_id, today, timestamp))
        print(f"   ✅ Marked present in database")
    else:
        print(f"   ℹ️  Already marked present")
    
    conn.commit()
    conn.close()

def main():
    print("\n" + "="*60)
    print("📸 DEMO ATTENDANCE SYSTEM")
    print("Processing ID card images...")
    print("="*60)
    
    # Process all images in training_data
    images = glob.glob('training_data/*.jpeg') + glob.glob('training_data/*.jpg')
    
    if not images:
        print("\n❌ No images found in training_data/")
        return
    
    print(f"\nFound {len(images)} images\n")
    
    success_count = 0
    for img_path in images[:1]:  # Process just the first one for demo
        result = process_clear_image(img_path, "MSC_CS")
        if result:
            success_count += 1
    
    print("\n" + "="*60)
    print(f"✅ Processing complete: {success_count}/{len(images[:1])} successful")
    print("="*60)
    
    # Show what's in database
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    
    cursor.execute("""
        SELECT s.student_id, s.name, a.timestamp
        FROM attendance a
        JOIN students s ON a.student_id = s.student_id
        WHERE a.date = ? AND a.course_id = 'MSC_CS'
        ORDER BY a.timestamp
    """, (today,))
    
    records = cursor.fetchall()
    conn.close()
    
    if records:
        print(f"\n📊 Attendance for {today}:")
        print("-"*60)
        for sid, name, ts in records:
            time_str = ts.split('T')[1][:8]
            print(f"  {sid:12} {name:20} {time_str}")
        print(f"\nTotal: {len(records)} students present")
    
    print("\n✅ Ready for export:")
    print(f"  python src/attendance_system.py --export excel MSC_CS {today}")
    print(f"  python src/attendance_system.py --export pdf MSC_CS {today}\n")

if __name__ == "__main__":
    main()