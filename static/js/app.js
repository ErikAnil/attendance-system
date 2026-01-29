// Sample data for demonstration
let attendanceData = [
  {
    student_id: "CS2024001",
    name: "John Smith",
    department: "Computer Science",
    course: "CS501",
    date: "2024-12-14",
    timestamp: "2024-12-14T09:15:00",
    status: "present",
  },
  {
    student_id: "CS2024002",
    name: "Emma Johnson",
    department: "Computer Science",
    course: "CS501",
    date: "2024-12-14",
    timestamp: "2024-12-14T09:16:00",
    status: "present",
  },
  {
    student_id: "CS2024003",
    name: "Michael Chen",
    department: "Computer Science",
    course: "CS502",
    date: "2024-12-14",
    timestamp: "2024-12-14T10:30:00",
    status: "present",
  },
];

// Initialize on page load
window.onload = function () {
  // Set today's date
  document.getElementById("dateSelect").valueAsDate = new Date();

  // Setup drag and drop
  setupDragAndDrop();

  // Load initial data
  renderAttendance();
  updateStats();
};

// Drag and Drop Setup
function setupDragAndDrop() {
  const uploadSection = document.getElementById("uploadSection");

  uploadSection.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadSection.classList.add("drag-over");
  });

  uploadSection.addEventListener("dragleave", () => {
    uploadSection.classList.remove("drag-over");
  });

  uploadSection.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadSection.classList.remove("drag-over");

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleVideoUpload(files[0]);
    }
  });

  document.getElementById("videoInput").addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleVideoUpload(e.target.files[0]);
    }
  });
}

// Handle Video Upload
function handleVideoUpload(file) {
  console.log("Uploading video:", file.name);

  // Show processing status
  document.getElementById("processingStatus").classList.add("active");

  // Simulate processing (in real app, this would call your Python backend)
  setTimeout(() => {
    // Add some dummy data
    const newRecords = [
      {
        student_id: "CS2024004",
        name: "Sarah Williams",
        department: "Computer Science",
        course: document.getElementById("courseSelect").value || "CS501",
        date: document.getElementById("dateSelect").value,
        timestamp: new Date().toISOString(),
        status: "present",
      },
      {
        student_id: "CS2024005",
        name: "James Brown",
        department: "Computer Science",
        course: document.getElementById("courseSelect").value || "CS501",
        date: document.getElementById("dateSelect").value,
        timestamp: new Date().toISOString(),
        status: "present",
      },
    ];

    attendanceData = [...attendanceData, ...newRecords];

    document.getElementById("processingStatus").classList.remove("active");
    alert(
      `✅ Video processed successfully!\n${newRecords.length} students marked present.`
    );

    renderAttendance();
    updateStats();
  }, 3000);
}

// Render Attendance Table
function renderAttendance(data = attendanceData) {
  const tbody = document.getElementById("attendanceBody");
  const emptyState = document.getElementById("emptyState");

  if (data.length === 0) {
    tbody.innerHTML = "";
    emptyState.style.display = "block";
    return;
  }

  emptyState.style.display = "none";

  tbody.innerHTML = data
    .map(
      (record) => `
        <tr>
            <td><strong>${record.student_id}</strong></td>
            <td>${record.name}</td>
            <td>${record.department}</td>
            <td>${record.course}</td>
            <td>${record.date}</td>
            <td>${new Date(record.timestamp).toLocaleTimeString()}</td>
            <td>
                <span class="status-badge status-${record.status}">
                    ${record.status.toUpperCase()}
                </span>
            </td>
        </tr>
    `
    )
    .join("");
}

// Update Statistics
function updateStats() {
  const uniqueStudents = new Set(attendanceData.map((r) => r.student_id)).size;
  const today = new Date().toISOString().split("T")[0];
  const todayPresent = attendanceData.filter((r) => r.date === today).length;

  document.getElementById("totalStudents").textContent = uniqueStudents;
  document.getElementById("todayPresent").textContent = todayPresent;
  document.getElementById("videosProcessed").textContent = Math.ceil(
    attendanceData.length / 15
  ); // Estimate
  document.getElementById("avgAttendance").textContent =
    uniqueStudents > 0
      ? Math.round((todayPresent / uniqueStudents) * 100) + "%"
      : "0%";
}

// Filter Attendance
function filterAttendance() {
  const course = document.getElementById("courseSelect").value;
  const date = document.getElementById("dateSelect").value;

  let filtered = attendanceData;

  if (course) {
    filtered = filtered.filter((r) => r.course === course);
  }

  if (date) {
    filtered = filtered.filter((r) => r.date === date);
  }

  renderAttendance(filtered);
}

// Export Functions
function exportToCSV() {
  const csv = [
    ["Student ID", "Name", "Department", "Course", "Date", "Time", "Status"],
    ...attendanceData.map((r) => [
      r.student_id,
      r.name,
      r.department,
      r.course,
      r.date,
      new Date(r.timestamp).toLocaleTimeString(),
      r.status,
    ]),
  ]
    .map((row) => row.join(","))
    .join("\n");

  downloadFile(csv, "attendance.csv", "text/csv");
}

function exportToExcel() {
  alert(
    "📊 Generating Excel file...\n\nFor full Excel export with formatting, use the Python command:\n\npython src/attendance_system.py --export excel CS501 2024-12-15"
  );

  // Fallback: download CSV
  exportToCSV();
}

function exportToPDF() {
  alert(
    "📄 Generating PDF file...\n\nFor full PDF export with formatting, use the Python command:\n\npython src/attendance_system.py --export pdf CS501 2024-12-15"
  );
}

function downloadFile(content, filename, type) {
  const blob = new Blob([content], { type: type });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  window.URL.revokeObjectURL(url);
}

// Refresh Data
function refreshData() {
  // In real app, this would fetch from backend
  renderAttendance();
  updateStats();
  alert("✅ Data refreshed!");
}
