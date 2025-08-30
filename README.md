
#Face Recognition Attendance System with Geolocation & Liveness Detection
This project is a robust, web-based attendance system that uses facial recognition to mark student attendance. It's built with Flask for the backend and features a comprehensive admin dashboard for management. Key features include geolocation verification to ensure students are physically present, liveness detection to prevent spoofing with photos, and automated email reporting.

✨ Core Features
Admin Dashboard: A central hub to manage students, view reports, configure the system, and start attendance sessions.

Face Recognition: Powered by the DeepFace library for accurate facial verification.

Geolocation Verification: Ensures students are within a specified radius (e.g., 100 meters) of the instructor when marking attendance.

Liveness Detection: Requires students to perform a simple action (smile) to verify they are a live person, preventing spoofing with static images.

Session-Based Attendance: Instructors generate unique, time-limited links for each class session.

Student Management: Easily add new students with their photos, rename them, add more photos, or delete them.

Twin Handling: Implements a stricter verification model for students identified as twins to improve accuracy.

Timetable Management: A dynamic, editable timetable to define class schedules, which automatically determines the active subject.

Automated Reporting:

View real-time daily attendance (Present vs. Absent).

Generate overall attendance percentage reports per student.

Filter reports by subject.

Email Notifications:

Configure a sender Gmail account (using a secure App Password).

Send daily or overall attendance reports to all registered students with a single click.

Continuous Learning: Automatically adds successfully verified photos back to a student's dataset to improve the recognition model over time.

Proof of Attendance: Saves a snapshot of the student's face at the time of attendance in a structured folder (attendance_proofs/YYYY-MM-DD/SubjectName/).

🛠️ Tech Stack
Backend: Flask

Face Recognition: DeepFace (utilizing VGGFace, OpenCV, and TensorFlow)

Frontend: HTML5, Tailwind CSS, Vanilla JavaScript

Data Handling: Pandas, NumPy

Geolocation: Geopy


📁 Project Structure
.
├── app.py
├── dataset/
│   └── 12345-John Doe/
│       ├── image1.jpg
│       └── image2.jpg
├── attendance_records/
│   └── 2025-08-30/
│       └── attendance.csv
├── attendance_proofs/
│   └── 2025-08-30/
│       └── Computer Vision/
│           └── 12345-John Doe_143005.jpg
├── sender_gmail.json
├── student_emails.json
├── timetable.json
├── twins.json
└── requirements.txt
🚀 Setup and Installation
1. Prerequisites
Python 3.8+

pip (Python package installer)

A C++ compiler (required by dlib, a dependency of DeepFace). On Windows, install Visual Studio Build Tools. On Linux, use sudo apt-get install build-essential.

2. Clone the Repository
Bash

git clone <your-repository-url>
cd <repository-folder>
3. Set Up a Virtual Environment (Recommended)
Bash

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
4. Install Dependencies
Create a requirements.txt file with the following content:

Plaintext

flask
pandas
numpy
opencv-python
deepface
geopy
gunicorn
Then, install the packages:

Bash

pip install -r requirements.txt
Note: The first run of DeepFace will automatically download pre-trained model weights, which may take some time.

5. Initial Configuration
The script will automatically create the necessary folders (dataset, attendance_records, attendance_proofs) when you first run it.

The essential JSON files will also be created as you use the admin dashboard features.

6. Run the Application
Bash

flask run --host=0.0.0.0 --port=5000
The application will be accessible at http://localhost:5000.

📖 How to Use
Admin Workflow
Login: Navigate to http://localhost:5000/admin and log in with the default password admin123.

Configure Timetable: Go to the "Manage Timetable" section and add your class schedule.

Add Students: Click "Add New Student", enter the student's Unique ID and Full Name, and upload at least 3 high-quality photos.

Configure Email Settings (Optional):

Click "Configure Sender Gmail". You must use a Google Account App Password.

Click "Manage Student Emails" to map names to email addresses.

Start an Attendance Session: During a class, click "Generate Session Link", allow location access, and share the generated link with students.

View Reports & Send Emails: Use the dashboard to view real-time data and send reports.

Student Workflow
Open Link: Open the attendance link from the instructor.

Grant Permissions: Allow the browser to access your camera and location.

Enter ID: Type your unique Student ID.

Mark Attendance: Position your face, smile for the liveness check, and click the button.

Get Confirmation: Receive a success or error message on the screen.

⚙️ Configuration
The following settings can be modified directly in the app.py file:

ADMIN_PASSWORD: Change the password for the admin dashboard.

MAX_DISTANCE_METERS: The maximum allowed distance (in meters) between student and instructor.

SESSION_TIMEOUT_MINUTES: How long a generated attendance link is valid.

CONFIDENCE_THRESHOLD: The tolerance for face matching (lower is stricter).

⚠️ Important Note on representations_vgg_face.pkl
DeepFace creates a file named representations_vgg_face.pkl inside the dataset/ directory to store pre-computed face embeddings, speeding up recognition. This application automatically deletes this file whenever you add, rename, or delete a student. This forces DeepFace to rebuild its database with the updated information.
