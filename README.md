# Real-Time Student Attendance Tracking System

A computer vision-based attendance tracking system for Maynooth University that automatically detects and reads student ID cards using a webcam.

[Python]
[OpenCV]
[Flask]

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)
- [Author](#author)

## Features

- Real-time ID card detection using webcam
- OCR recognition of 8-digit student ID numbers using Tesseract
- Multi-scan voting algorithm for improved accuracy
- Date filtering to distinguish student IDs from dates of birth
- Web-based dashboard accessible via any browser
- Duplicate prevention for same-day scans
- Export attendance reports to PDF and Excel formats
- Fully offline operation after installation

## Requirements

### Hardware

| Component | Minimum Requirement                          |
| --------- | -------------------------------------------- |
| Computer  | Any modern laptop or desktop (2015 or newer) |
| Webcam    | 720p resolution (built-in or USB)            |
| RAM       | 4 GB (8 GB recommended)                      |
| Storage   | 500 MB free disk space                       |

### Software

| Software      | Version                          |
| ------------- | -------------------------------- |
| Python        | 3.8 or higher                    |
| Tesseract OCR | 4.0 or higher                    |
| Web Browser   | Chrome, Firefox, Safari, or Edge |

### Operating System

- Windows 10/11
- macOS 10.14+
- Ubuntu 20.04+

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/attendance-system.git
cd attendance-system
```

### 2. Install Tesseract OCR

**Windows:**

1. Download from https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer
3. Add Tesseract to your system PATH

**macOS:**

```bash
brew install tesseract
```

**Ubuntu/Debian:**

```bash
sudo apt update
sudo apt install tesseract-ocr
```

### 3. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify Installation

```bash
python --version          # Should show Python 3.8+
tesseract --version       # Should show Tesseract 4.0+
```

## Usage

### Starting the Application

```bash
cd src
python professor_dashboard.py
```

You should see:

```
PROFESSOR ATTENDANCE DASHBOARD
Open in browser: http://localhost:5000
```

### Using the Dashboard

1. Open your browser and navigate to http://localhost:5000
2. Click "Start Scanning" to activate the webcam
3. Hold a student ID card within the white corner brackets
4. Wait for confirmation - the frame turns green when successful
5. View recorded attendance in the right panel
6. Export reports using the PDF or Excel buttons

### Scanning Best Practices

- Ensure adequate lighting (avoid direct sunlight or very dim conditions)
- Hold the card flat, parallel to the camera
- Position the card 20-40 cm from the camera
- Keep the card steady for 1-2 seconds
- Ensure the student ID number is visible and not obscured

### Stopping the Application

Press Ctrl+C in the terminal to stop the server.

## Project Structure

```
attendance-system/
├── models/
│   ├── enhanced_id_model.npz    # Trained detection model
│   └── templates.pkl            # Template matching data
├── src/
│   ├── professor_dashboard.py   # Main application
│   ├── train_detector.py        # Model training script
│   └── templates/
│       └── dashboard.html       # Web interface
├── tests/                       # Test files
├── docs/                        # Documentation
│   └── user_manual.docx
├── .gitignore
├── README.md
└── requirements.txt
```

## How It Works

### Card Detection

The system uses OpenCV edge detection and contour analysis to identify rectangular objects matching ID card dimensions (aspect ratio 1.3-2.0).

### State Machine

A 4-state machine manages the scanning process:

| State    | Description                       |
| -------- | --------------------------------- |
| Waiting  | Looking for card in frame         |
| Detected | Card found, hold still (1 second) |
| Scanning | Performing OCR on captured frames |
| Success  | ID recorded, showing confirmation |

### Multi-Scan Voting Algorithm

Instead of relying on a single OCR reading, the system:

1. Captures 5 frames in rapid succession
2. Performs OCR on each frame
3. For each digit position, counts frequency of each recognized digit
4. Selects the most common digit for each position

This approach improves accuracy from approximately 78% to 95%.

### Date Filtering

Student IDs (e.g., 24250414) must be distinguished from dates of birth (e.g., 02102002). The system:

1. Parses each 8-digit number as DDMMYYYY and MMDDYYYY formats
2. Rejects numbers that form valid dates between 1980-2030
3. Prioritizes numbers starting with '2' (Maynooth ID format)

## Configuration

Edit `src/professor_dashboard.py` to customize settings:

```python
# Line ~48: Change course name
self.course_name = "MSC COMPUTER SCIENCE"

# Line ~50: Adjust cooldown between scans (seconds)
self.cooldown = 4

# Line ~52: Modify detection sensitivity
self.history_size = 5
```

## Troubleshooting

| Problem                    | Solution                                                               |
| -------------------------- | ---------------------------------------------------------------------- |
| Camera does not start      | Check webcam connection; close other applications using camera         |
| ID not detected            | Improve lighting; hold card steady; clean camera lens                  |
| Wrong ID recognized        | Hold card closer or further; ensure number is not worn                 |
| Application will not start | Verify Python and Tesseract installation; check if port 5000 is in use |
| Export fails               | Verify reportlab (PDF) and openpyxl (Excel) are installed              |

### Common Errors

**"No module named 'cv2'"**

```bash
pip install opencv-python
```

**"TesseractNotFoundError"**

Ensure Tesseract is installed and in system PATH. On Windows, you may need to set the path manually in the code:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

## Performance

| Metric                           | Result      |
| -------------------------------- | ----------- |
| Card Detection Rate              | 98%         |
| OCR Accuracy (Single Scan)       | 78%         |
| OCR Accuracy (Multi-Scan Voting) | 95%         |
| Average Processing Time          | 1.2 seconds |
| False Positive Rate              | 0%          |

## Future Work

- Mobile application support
- Facial recognition as secondary verification
- Integration with university student information system
- Attendance analytics dashboard
- Multi-camera support for large lectures

## Author

Erik  
MSc Computer Science  
Maynooth University  
2024-2025

## Acknowledgments

- OpenCV - Computer vision library
- Tesseract OCR - Optical character recognition engine
- Flask - Web application framework
- Maynooth University Department of Computer Science
