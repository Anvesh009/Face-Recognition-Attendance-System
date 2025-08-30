import os
import cv2
import json
import pandas as pd
import numpy as np
import base64
import uuid
import shutil
import smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta
from flask import Flask, render_template_string, request, jsonify, redirect, url_for
from geopy.distance import geodesic
from deepface import DeepFace

# --- Basic Flask App Setup ---
app = Flask(__name__)
app.secret_key = 'your_very_secret_key_for_sessions'

# --- Configuration ---
ADMIN_PASSWORD = "admin123"
ATTENDANCE_SESSIONS = {}
MAX_DISTANCE_METERS = 100
SESSION_TIMEOUT_MINUTES = 30
DATASET_PATH = "dataset"
ATTENDANCE_RECORDS_PATH = "attendance_records"
ATTENDANCE_PROOFS_PATH = "attendance_proofs"
SENDER_GMAIL_FILE = "sender_gmail.json"
STUDENT_EMAILS_FILE = "student_emails.json"
TIMETABLE_FILE = "timetable.json"
TWINS_FILE = "twins.json"

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.4

# --- Core Logic & Helper Functions ---

def sanitize_filename(filename):
    """Removes characters that are illegal in filenames."""
    return "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()

def get_all_students():
    """Gets a list of student names from the dataset folder names."""
    if not os.path.exists(DATASET_PATH): return []
    student_folders = sorted([f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))])
    student_names = []
    for folder in student_folders:
        try:
            student_names.append(folder.split('-', 1)[1])
        except IndexError:
            student_names.append(folder)
    return student_names

def find_folder_by_id(student_id):
    """Finds the full 'ID-Name' folder for a given student ID."""
    if not os.path.exists(DATASET_PATH): return None
    for folder in os.listdir(DATASET_PATH):
        if os.path.isdir(os.path.join(DATASET_PATH, folder)) and folder.startswith(f"{student_id}-"):
            return folder
    return None

def find_folder_by_name(student_name):
    """Finds the full 'ID-Name' folder for a given student display name."""
    if not os.path.exists(DATASET_PATH): return None
    for folder in os.listdir(DATASET_PATH):
        if os.path.isdir(os.path.join(DATASET_PATH, folder)):
            try:
                if folder.split('-', 1)[1] == student_name:
                    return folder
            except IndexError:
                if folder == student_name:
                    return folder
    return None

def load_twins():
    """Loads twin pairs from the twins.json file."""
    if not os.path.exists(TWINS_FILE):
        return {}
    with open(TWINS_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_twins(twins_data):
    """Saves twin pairs to the twins.json file."""
    with open(TWINS_FILE, 'w') as f:
        json.dump(twins_data, f, indent=4)

def mark_attendance(name, subject):
    """Marks a student's attendance in the CSV file for the current day inside a dated folder."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    # Create a directory for the current date if it doesn't exist
    date_folder_path = os.path.join(ATTENDANCE_RECORDS_PATH, date_str)
    os.makedirs(date_folder_path, exist_ok=True)
    
    file_path = os.path.join(date_folder_path, "attendance.csv")
    time_now = now.strftime("%H:%M:%S")

    new_entry = pd.DataFrame([[name, time_now, subject]], columns=["Name", "Time", "Subject"])

    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=["Name", "Time", "Subject"])
    else:
        df = pd.DataFrame(columns=["Name", "Time", "Subject"])

    if not df[(df['Name'] == name) & (df['Subject'] == subject)].empty:
        return False # Already marked

    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(file_path, index=False)
    return True

def get_current_subject():
    """Determines the current subject based on the timetable."""
    if not os.path.exists(TIMETABLE_FILE): return None
    try:
        with open(TIMETABLE_FILE, 'r') as f: timetable = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError): return None
    
    now = datetime.now()
    today_str = now.strftime('%A')
    
    for slot in timetable.get(today_str, []):
        try:
            start_dt = datetime.combine(now.date(), datetime.strptime(slot['start'], "%H:%M").time())
            end_dt = datetime.combine(now.date(), datetime.strptime(slot['end'], "%H:%M").time())
            if (start_dt - timedelta(minutes=10)) <= now <= (end_dt + timedelta(minutes=15)):
                return slot['subject']
        except (ValueError, KeyError): continue
    return None

# --- HTML Templates ---
def render_student_page(session_id, subject, message=None):
    return f"""
    <!DOCTYPE html>
    <head>
        <html lang="en">
        <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Face Attendance</title><script src="https://cdn.tailwindcss.com"></script>
        <style>
            #video-feed {{ transform: scaleX(-1); }} .status-box {{ transition: all 0.3s ease-in-out; }}
            .status-success {{ background-color: #d1fae5; border-color: #10b981; color: #065f46; }}
            .status-error {{ background-color: #fee2e2; border-color: #ef4444; color: #991b1b; }}
            .status-processing {{ background-color: #fef3c7; border-color: #f59e0b; color: #92400e; }}
        </style>
    </head>
    <body class="bg-gray-100 flex items-center justify-center min-h-screen">
        <div class="w-full max-w-2xl mx-auto bg-white rounded-2xl shadow-lg p-6 md:p-8 text-center">
            <h1 class="text-3xl md:text-4xl font-bold text-gray-800 mb-2">Attendance for: {subject}</h1>
            <p id="main-prompt" class="text-gray-600 mb-6">Enter your Student ID to begin.</p>
            <div id="student-id-entry-container" class="mb-6">
                <label for="student-id-entry" class="block text-lg font-medium text-gray-700 mb-2">Student ID</label>
                <input type="text" id="student-id-entry" class="w-full p-3 border border-gray-300 rounded-lg text-lg" placeholder="e.g., 12345" required>
            </div>
            <div class="relative w-full aspect-video bg-black rounded-lg overflow-hidden mb-6 border-4 border-gray-200">
                <video id="video-feed" class="w-full h-full object-cover" autoplay playsinline></video>
                <div id="loading-overlay" class="absolute inset-0 bg-black bg-opacity-75 flex items-center justify-center"><p class="text-white text-xl">Starting Camera...</p></div>
            </div>
            <p id="liveness-prompt" class="text-lg font-semibold text-blue-600 hidden">Please smile or move slightly to confirm you are live.</p>
            <button id="mark-attendance-btn" class="w-full bg-blue-600 text-white font-bold py-4 px-6 rounded-lg text-xl hover:bg-blue-700 disabled:bg-gray-400" disabled>Mark My Attendance</button>
            <div id="status-display" class="status-box mt-6 p-4 rounded-lg border-2 text-lg font-semibold text-center opacity-0">{message if message else 'Status will appear here'}</div>
        </div>
        <canvas id="canvas" class="hidden"></canvas>
        <script>
            const video = document.getElementById('video-feed'), canvas = document.getElementById('canvas'), markButton = document.getElementById('mark-attendance-btn');
            const statusDisplay = document.getElementById('status-display'), loadingOverlay = document.getElementById('loading-overlay'), mainPrompt = document.getElementById('main-prompt');
            const studentIdEntry = document.getElementById('student-id-entry'), livenessPrompt = document.getElementById('liveness-prompt');
            const sessionId = "{session_id}"; let studentLocation = null, isProcessing = false;

            function showStatus(message, type = 'info') {{ statusDisplay.textContent = message; statusDisplay.className = 'status-box mt-6 p-4 rounded-lg border-2 text-lg font-semibold text-center'; statusDisplay.classList.add(`status-${{type}}`); statusDisplay.style.opacity = 1; }}
            
            async function setupDevice() {{
                showStatus("Requesting location permission...", "processing"); mainPrompt.textContent = "Please allow location access and enter your Student ID.";
                navigator.geolocation.getCurrentPosition( (position) => {{ studentLocation = {{ latitude: position.coords.latitude, longitude: position.coords.longitude }}; showStatus("Location found! Enter your Student ID to proceed.", "success"); mainPrompt.textContent = "Enter your Student ID and point camera at face."; }}, (err) => {{ showStatus("Location access denied. You cannot mark attendance.", "error"); mainPrompt.textContent = "Location is required. Please enable it and refresh."; markButton.disabled = true; }}, {{ enableHighAccuracy: true }} );
                try {{ const stream = await navigator.mediaDevices.getUserMedia({{ video: {{ facingMode: 'user' }} }}); video.srcObject = stream; video.onloadedmetadata = () => {{ loadingOverlay.style.display = 'none'; }}; }} catch (err) {{ loadingOverlay.innerHTML = `<p class="text-red-400 text-xl px-4">Error: Could not access camera.</p>`; markButton.disabled = true; }}
            }}
            
            studentIdEntry.addEventListener('input', () => {{
                if (studentIdEntry.value.trim().length > 0 && studentLocation) {{
                    markButton.disabled = false;
                    showStatus("Student ID entered. Ready to mark attendance.", "success");
                }} else {{
                    markButton.disabled = true;
                    showStatus("Please enter your Student ID.", "processing");
                }}
            }});
            
            markButton.addEventListener('click', async () => {{
                if (isProcessing || !studentLocation || !studentIdEntry.value.trim()) return;
                isProcessing = true; markButton.disabled = true; markButton.textContent = 'Processing...'; showStatus('Capturing & verifying...', 'processing');
                
                canvas.width = video.videoWidth; canvas.height = video.videoHeight;
                const context = canvas.getContext('2d');
                context.translate(canvas.width, 0); context.scale(-1, 1);
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                const imageData = canvas.toDataURL('image/jpeg');

                let payload = {{ image: imageData, location: studentLocation, student_id: studentIdEntry.value.trim() }};

                try {{
                    const response = await fetch(`/api/mark_attendance/${{sessionId}}`, {{ method: 'POST', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify(payload) }});
                    const result = await response.json();
                    showStatus(result.message, result.success ? 'success' : 'error');

                    if (result.requires_liveness) {{
                        livenessPrompt.classList.remove('hidden');
                        markButton.textContent = 'Retry Liveness';
                    }} else if (result.success) {{
                        livenessPrompt.classList.add('hidden');
                        markButton.disabled = true;
                        markButton.textContent = 'Attendance Marked';
                    }}
                }} catch (error) {{ showStatus('Error: Could not connect to the server.', 'error'); }} finally {{ 
                    isProcessing = false;
                    if (!markButton.textContent.includes('Marked')) {{
                        markButton.disabled = false;
                    }}
                }}
            }});
            setupDevice();
        </script>
    </body>
    </html>
    """

ADMIN_LOGIN_PAGE = """
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Admin Login</title><script src="https://cdn.tailwindcss.com"></script></head><body class="bg-gray-200 flex items-center justify-center h-screen"><div class="bg-white p-8 rounded-lg shadow-md w-96 text-center"><h1 class="text-2xl font-bold mb-6">Admin Login</h1><form method="POST" action="/admin/login"><input type="password" name="password" placeholder="Enter Password" class="w-full p-2 border rounded mb-4"><button type="submit" class="w-full bg-blue-600 text-white p-2 rounded hover:bg-blue-700">Login</button></form></div></body></html>
"""

ADMIN_DASHBOARD_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><title>Admin Dashboard</title><script src="https://cdn.tailwindcss.com"></script>
   <style>.modal-overlay{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.5);display:flex;align-items:center;justify-content:center;z-index:100;}.modal-content{background:white;padding:2rem;border-radius:0.5rem;width:90%;max-width:500px;}</style>
</head>
<body class="bg-gray-100">
    <div class="container mx-auto p-4 md:p-8">
        <h1 class="text-4xl font-bold text-gray-800 mb-8">Admin Dashboard</h1>
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div class="lg:col-span-2 space-y-8">
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h2 class="text-2xl font-bold mb-4">1. Start Attendance Session</h2>
                    <p id="current-subject-info" class="mb-4 text-lg text-gray-700">Current active subject: <span class="font-bold text-blue-600">None</span></p>
                    <button id="generate-link-btn" class="w-full bg-green-600 text-white font-bold py-3 px-6 rounded-lg text-lg hover:bg-green-700 disabled:bg-gray-400">Generate Session Link</button>
                    <div id="link-display" class="mt-4 p-4 bg-gray-100 rounded border hidden"><p class="font-semibold">Share this link:</p><a id="attendance-link" href="#" target="_blank" class="text-blue-600 hover:underline break-all"></a></div>
                    <p id="session-status" class="mt-4 text-gray-600"></p>
                </div>
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h2 class="text-2xl font-bold mb-4">Attendance Reports</h2>
                    <div class="flex justify-between items-center mb-4">
                        <select id="subject-filter" class="p-2 border rounded-md"></select>
                        <button id="refresh-reports-btn" class="text-blue-600 hover:underline">Refresh</button>
                    </div>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                            <h3 class="text-xl font-semibold text-green-600 mb-2">Present Today (<span id="present-count">0</span>)</h3>
                            <div id="todays-present-list" class="space-y-2 h-48 overflow-y-auto border p-4 rounded-md"><p class="text-gray-500">No students present.</p></div>
                        </div>
                        <div>
                            <h3 class="text-xl font-semibold text-red-600 mb-2">Absent Today (<span id="absent-count">0</span>)</h3>
                            <div id="todays-absent-list" class="space-y-2 h-48 overflow-y-auto border p-4 rounded-md"><p class="text-gray-500">No students absent.</p></div>
                        </div>
                    </div>
                    <h3 class="text-xl font-semibold my-4">Overall Percentage</h3>
                    <div id="overall-report-table" class="h-64 overflow-y-auto"><p class="text-gray-500">Loading overall report...</p></div>
                </div>
            </div>
            <div class="space-y-8">
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h2 class="text-2xl font-bold mb-4">2. Student Management</h2>
                    <button id="add-student-btn" class="w-full bg-blue-600 text-white p-3 rounded hover:bg-blue-700 mb-4">Add New Student</button>
                    <h3 class="text-xl font-semibold mb-2">Registered Students (<span id="total-students-count">0</span>)</h3><div id="student-list" class="space-y-3 h-64 overflow-y-auto border p-2 rounded-md"><p class="text-gray-500">Loading...</p></div>
                </div>
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h2 class="text-2xl font-bold mb-4">3. Email Operations</h2>
                    <div class="space-y-3">
                        <button id="config-sender-btn" class="w-full bg-gray-700 text-white p-3 rounded hover:bg-gray-800">Configure Sender Gmail</button>
                        <button id="manage-student-emails-btn" class="w-full bg-gray-700 text-white p-3 rounded hover:bg-gray-800">Manage Student Emails</button>
                        <div class="mt-4 border-t pt-4">
                            <label for="email-subject-filter" class="font-semibold block mb-2">Select Report to Send:</label>
                            <select id="email-subject-filter" class="w-full p-2 border rounded-md mb-2"></select>
                            <button id="send-todays-report-btn" class="w-full bg-red-600 text-white p-3 rounded hover:bg-red-700">Send Today's Report</button>
                        </div>
                        <div class="mt-4 border-t pt-4">
                            <button id="send-overall-email-btn" class="w-full bg-purple-600 text-white p-3 rounded hover:bg-purple-700">Send Detailed Overall Report</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="bg-white p-6 rounded-lg shadow-md mt-8">
            <h2 class="text-2xl font-bold mb-4">4. Manage Timetable</h2>
            <div id="timetable-container" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"></div>
        </div>
    </div>
    <div id="add-student-modal" class="modal-overlay hidden"><div class="modal-content"><h3 class="text-xl font-bold mb-4">Add New Student</h3><form id="add-student-form"><input type="text" id="new-student-id" name="student_id" placeholder="Unique Student ID (e.g., 12345)" class="w-full p-2 border rounded mb-2" required><input type="text" id="new-student-name" name="name" placeholder="Student's Full Name" class="w-full p-2 border rounded mb-2" required><label class="block mb-2 text-sm font-medium text-gray-700">Upload Images (3+):</label><input type="file" id="student-images" name="images" class="w-full" multiple required><div class="flex items-center mb-2"><input type="checkbox" id="is-twin" name="is_twin" class="mr-2"><label for="is-twin" class="text-sm font-medium text-gray-700">This student is a twin</label></div><div class="flex justify-end gap-4 mt-4"><button type="button" id="cancel-add-student" class="bg-gray-300 p-2 px-4 rounded">Cancel</button><button type="submit" class="bg-blue-600 text-white p-2 px-4 rounded">Add</button></div></form></div></div>
    <div id="add-photos-modal" class="modal-overlay hidden"><div class="modal-content"><h3 class="text-xl font-bold mb-4">Add More Photos</h3><form id="add-photos-form"><input type="hidden" id="add-photos-name" name="name"><p class="mb-2">Adding photos for: <b id="add-photos-student-name"></b></p><label class="block mb-2 text-sm font-medium text-gray-700">Upload New Images:</label><input type="file" id="add-photos-images" name="images" class="w-full" multiple required><div class="flex justify-end gap-4 mt-4"><button type="button" id="cancel-add-photos" class="bg-gray-300 p-2 px-4 rounded">Cancel</button><button type="submit" class="bg-blue-600 text-white p-2 px-4 rounded">Add Photos</button></div></form></div></div>
    <div id="sender-modal" class="modal-overlay hidden"><div class="modal-content"><h3 class="text-xl font-bold mb-4">Configure Sender Gmail</h3><p class="text-sm text-gray-600 mb-4">Enter your Gmail and an <a href='https://myaccount.google.com/apppasswords' target='_blank' class='text-blue-600'>App Password</a>.</p><form id="sender-form"><input type="email" id="sender-email" name="sender-email" placeholder="your.email@gmail.com" class="w-full p-2 border rounded mb-2" required><input type="password" id="sender-password" name="sender-password" placeholder="Gmail App Password" class="w-full p-2 border rounded mb-4" required><div class="flex justify-end gap-4"><button type="button" id="cancel-sender" class="bg-gray-300 p-2 px-4 rounded">Cancel</button><button type="submit" class="bg-blue-600 text-white p-2 px-4 rounded">Save</button></div></form></div></div>
    <div id="student-emails-modal" class="modal-overlay hidden"><div class="modal-content"><h3 class="text-xl font-bold mb-4">Manage Student Emails</h3><form id="student-emails-form"><div id="student-emails-list" class="space-y-2 mb-4 max-h-80 overflow-y-auto"></div><div class="flex justify-end gap-4"><button type="button" id="cancel-student-emails" class="bg-gray-300 p-2 px-4 rounded">Cancel</button><button type="submit" class="bg-blue-600 text-white p-2 px-4 rounded">Save All</button></div></form></div></div>
    <div id="timetable-modal" class="modal-overlay hidden"><div class="modal-content"><h3 class="text-xl font-bold mb-4">Add/Edit Class Slot</h3><form id="timetable-form"><input type="hidden" id="slot-day" name="slot-day"><input type="hidden" id="slot-id-input" name="slot-id" value=""><input type="text" id="slot-subject" name="slot-subject" placeholder="Subject Name" class="w-full p-2 border rounded mb-2" required><input type="time" id="slot-start" name="slot-start" class="w-full p-2 border rounded mb-2" required><input type="time" id="slot-end" name="slot-end" class="w-full p-2 border rounded mb-4" required><div class="flex justify-end gap-4"><button type="button" id="cancel-timetable" class="bg-gray-300 p-2 px-4 rounded">Cancel</button><button type="submit" class="bg-blue-600 text-white p-2 px-4 rounded">Save Slot</button></div></form></div></div>
<script>
document.addEventListener('DOMContentLoaded', () => {
    const api = {
        get: async (url) => fetch(url).then(res => res.json()),
        post: async (url, body) => fetch(url, { method: 'POST', body: body }).then(res => res.json())
    };
    const modals = { 'add-student-modal': document.getElementById('add-student-modal'), 'add-photos-modal': document.getElementById('add-photos-modal'), 'sender-modal': document.getElementById('sender-modal'), 'student-emails-modal': document.getElementById('student-emails-modal'), 'timetable-modal': document.getElementById('timetable-modal') };
    window.openModal = (id) => modals[id].classList.remove('hidden');
    window.closeModal = (id) => modals[id].classList.add('hidden');

    async function loadAll() {
        await Promise.all([ loadStudents(), loadReports(), loadTimetable(), loadCurrentSubject(), loadSubjectFilter() ]);
    }
    async function loadStudents() {
        const studentListDiv = document.getElementById('student-list');
        studentListDiv.innerHTML = '<p class="text-gray-500">Loading...</p>';
        try {
            const data = await api.get('/api/students');
            document.getElementById('total-students-count').textContent = data.students ? data.students.length : 0;
            studentListDiv.innerHTML = '';
            if (data.students && data.students.length > 0) {
                data.students.forEach(name => {
                    const el = document.createElement('div');
                    el.className = 'flex items-center justify-between bg-gray-50 p-2 rounded';
                    el.innerHTML = `<span class="font-medium">${name}</span><div class="flex items-center"><button class="text-sm text-green-600 hover:underline mr-2 add-photos-btn" data-name="${name}">+ Photos</button><button class="text-sm text-blue-500 hover:underline mr-2 rename-btn" data-name="${name}">Rename</button><button class="text-sm text-red-500 hover:underline delete-btn" data-name="${name}">Delete</button></div>`;
                    studentListDiv.appendChild(el);
                });
            } else { studentListDiv.innerHTML = '<p class="text-gray-500">No students registered.</p>'; }
        } catch (e) { studentListDiv.innerHTML = '<p class="text-red-500">Error loading students.</p>'; }
    }
    async function loadReports() {
        const subjectFilter = document.getElementById('subject-filter').value;
        const presentListDiv = document.getElementById('todays-present-list');
        const absentListDiv = document.getElementById('todays-absent-list');
        presentListDiv.innerHTML = '<p class="text-gray-500">Loading...</p>';
        absentListDiv.innerHTML = '<p class="text-gray-500">Loading...</p>';
        try {
            const todayData = await api.get(`/api/todays_attendance?subject=${subjectFilter}`);
            document.getElementById('present-count').textContent = todayData.present ? todayData.present.length : 0;
            document.getElementById('absent-count').textContent = todayData.absent ? todayData.absent.length : 0;
            presentListDiv.innerHTML = ''; absentListDiv.innerHTML = '';
            if (todayData.present && todayData.present.length > 0) {
                todayData.present.forEach(item => {
                    const p = document.createElement('p');
                    p.innerHTML = `<b>${item.Name}</b> <span class="text-gray-600">(${item.Subject})</span> at ${item.Time}`;
                    presentListDiv.appendChild(p);
                });
            } else { presentListDiv.innerHTML = '<p class="text-gray-500">No students present.</p>'; }
            if (todayData.absent && todayData.absent.length > 0) {
                todayData.absent.forEach(name => { const p = document.createElement('p'); p.textContent = name; absentListDiv.appendChild(p); });
            } else { absentListDiv.innerHTML = '<p class="text-gray-500">No students absent.</p>'; }
        } catch (e) {
            presentListDiv.innerHTML = '<p class="text-red-500">Error loading report.</p>';
            absentListDiv.innerHTML = '<p class="text-red-500">Error loading report.</p>';
        }
        const overallTableDiv = document.getElementById('overall-report-table');
        overallTableDiv.innerHTML = '<p class="text-gray-500">Loading...</p>';
        try {
            const overallData = await api.get(`/api/overall_attendance?subject=${subjectFilter}`);
            if (overallData.report && overallData.report.length > 0) {
                let tableHTML = `<table class="w-full text-left"><thead class="bg-gray-100"><tr><th class="p-2">Name</th><th>Present</th><th>Total Classes</th><th>%</th></tr></thead><tbody>`;
                overallData.report.forEach(item => { tableHTML += `<tr class="border-b"><td class="p-2 font-medium">${item.student}</td><td>${item.present_count}</td><td>${item.total_classes}</td><td class="font-semibold">${item.percentage.toFixed(1)}%</td></tr>`; });
                tableHTML += '</tbody></table>'; overallTableDiv.innerHTML = tableHTML;
            } else { overallTableDiv.innerHTML = '<p class="text-gray-500">No overall data found.</p>';}
        } catch (e) { overallTableDiv.innerHTML = '<p class="text-red-500">Error loading overall report.</p>'; }
    }
    async function loadCurrentSubject() {
        const data = await api.get('/api/current_subject');
        document.querySelector('#current-subject-info span').textContent = data.subject || 'None';
    }
    async function loadTimetable() {
        const data = await api.get('/api/timetable');
        const container = document.getElementById('timetable-container');
        container.innerHTML = '';
        const days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
        days.forEach(day => {
            let dayHtml = `<div class="bg-gray-50 p-4 rounded-lg"><h4 class="font-bold text-lg mb-2">${day}</h4><div class="space-y-2">`;
            const slots = data.timetable[day] || [];
            slots.forEach((slot) => {
                dayHtml += `<div class="flex justify-between items-center bg-white p-2 rounded shadow-sm cursor-pointer hover:bg-gray-100 slot-item" data-day="${day}" data-id="${slot.id}"><span>${slot.subject} (${slot.start}-${slot.end})</span><button class="text-sm text-red-500 hover:underline delete-slot-btn" data-day="${day}" data-id="${slot.id}">Delete</button></div>`;
            });
            dayHtml += `</div><button class="mt-2 text-blue-600 text-sm w-full text-center add-slot-btn" data-day="${day}">+ Add Slot</button></div>`;
            container.innerHTML += dayHtml;
        });
    }
    async function loadSubjectFilter() {
        const reportFilter = document.getElementById('subject-filter');
        const emailFilter = document.getElementById('email-subject-filter');
        const data = await api.get('/api/subjects');
        reportFilter.innerHTML = '<option value="all">All Subjects</option>';
        emailFilter.innerHTML = '<option value="all_today">All Subjects Today</option>';
        if(data.subjects) {
            data.subjects.forEach(s => {
                reportFilter.innerHTML += `<option value="${s}">${s}</option>`;
                emailFilter.innerHTML += `<option value="${s}">Today - ${s}</option>`;
            });
        }
    }

    // --- CORRECTED EVENT LISTENER LOGIC ---
    document.body.addEventListener('click', async (e) => {
        if (e.target.closest('.rename-btn')) {
            const oldName = e.target.closest('.rename-btn').dataset.name;
            const newName = prompt(`Enter new name for "${oldName}":`);
            if (newName && newName.trim() !== "") {
                const formData = new FormData(); formData.append('old_name', oldName); formData.append('new_name', newName.trim());
                const result = await api.post('/api/rename_student', formData);
                alert(result.message); loadAll();
            }
        }
        else if (e.target.closest('.delete-btn')) {
            const name = e.target.closest('.delete-btn').dataset.name;
            if (confirm(`Are you sure you want to delete "${name}"? This is irreversible.`)) {
                const formData = new FormData(); formData.append('name', name);
                const result = await api.post('/api/delete_student', formData);
                alert(result.message); loadAll();
            }
        }
        else if (e.target.closest('.add-photos-btn')) {
            const name = e.target.closest('.add-photos-btn').dataset.name;
            document.getElementById('add-photos-student-name').textContent = name;
            document.getElementById('add-photos-name').value = name;
            document.getElementById('add-photos-form').reset();
            openModal('add-photos-modal');
        }
        else if (e.target.closest('.add-slot-btn')) {
            const day = e.target.closest('.add-slot-btn').dataset.day;
            document.getElementById('timetable-form').reset();
            document.getElementById('slot-day').value = day;
            document.getElementById('slot-id-input').value = ""; // No ID for new slots
            openModal('timetable-modal');
        }
        // HANDLE DELETE FIRST to prevent event bubbling issues
        else if (e.target.closest('.delete-slot-btn')) {
            e.stopPropagation(); // Prevent the 'edit' click from firing
            const day = e.target.closest('.delete-slot-btn').dataset.day;
            const id = e.target.closest('.delete-slot-btn').dataset.id;
            if (!confirm('Delete this slot?')) return;
            const formData = new FormData();
            formData.append('day', day);
            formData.append('id', id);
            const result = await api.post('/api/delete_slot', formData);
            if (result.success) {
                loadAll();
            } else {
                alert(result.message);
            }
        } 
        // HANDLE EDIT SECOND
        else if (e.target.closest('.slot-item')) {
            const day = e.target.closest('.slot-item').dataset.day;
            const id = e.target.closest('.slot-item').dataset.id;
            const data = await api.get('/api/timetable');
            const slot = data.timetable[day].find(s => s.id === id);
            if (slot) {
                document.getElementById('timetable-form').reset();
                document.getElementById('slot-day').value = day;
                document.getElementById('slot-id-input').value = id;
                document.getElementById('slot-subject').value = slot.subject;
                document.getElementById('slot-start').value = slot.start;
                document.getElementById('slot-end').value = slot.end;
                openModal('timetable-modal');
            }
        }
    });

    document.getElementById('generate-link-btn').addEventListener('click', () => {
        const sessionStatus = document.getElementById('session-status'), generateBtn = document.getElementById('generate-link-btn');
        sessionStatus.textContent = 'Getting your location...'; generateBtn.disabled = true;
        navigator.geolocation.getCurrentPosition(async (position) => {
            sessionStatus.textContent = 'Location found! Generating link...';
            const formData = new FormData(); formData.append('location', JSON.stringify({ latitude: position.coords.latitude, longitude: position.coords.longitude }));
            const response = await api.post('/api/generate_link', formData);
            if (response.success) {
                document.getElementById('attendance-link').href = response.url; document.getElementById('attendance-link').textContent = response.url;
                document.getElementById('link-display').classList.remove('hidden');
                sessionStatus.textContent = `Link for ${response.subject} generated! Valid for ${response.timeout} minutes.`;
            } else { sessionStatus.textContent = `Error: ${response.message}`; generateBtn.disabled = false; }
        }, (err) => { sessionStatus.textContent = 'Error: Could not get location.'; generateBtn.disabled = false; });
    });
    document.getElementById('add-student-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const name = document.getElementById('new-student-name').value;
        const studentId = document.getElementById('new-student-id').value;
        const files = document.getElementById('student-images').files;
        const isTwin = document.getElementById('is-twin').checked;
        if (!name || !studentId || files.length < 1) { alert("Please provide a name, a unique student ID, and at least one image."); return; }
        const formData = new FormData();
        formData.append('name', name);
        formData.append('student_id', studentId);
        formData.append('is_twin', isTwin);
        for (let i = 0; i < files.length; i++) { formData.append('images', files[i]); }
        const result = await api.post('/api/add_student', formData); alert(result.message);
        if (result.success) { closeModal('add-student-modal'); e.target.reset(); loadAll(); }
    });
    document.getElementById('add-photos-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(e.target);
        const files = document.getElementById('add-photos-images').files;
        if (files.length < 1) { alert("Please select at least one image to upload."); return; }
        const result = await api.post('/api/add_photos', formData);
        alert(result.message);
        if (result.success) {
            closeModal('add-photos-modal');
        }
    });
    document.getElementById('timetable-form').addEventListener('submit', async (e) => {
        e.preventDefault(); const formData = new FormData(e.target);
        const result = await api.post('/api/save_slot', formData);
        if (result.success) { closeModal('timetable-modal'); e.target.reset(); loadAll(); } else { alert(result.message); }
    });
    document.getElementById('sender-form').addEventListener('submit', async (e) => { e.preventDefault(); const formData = new FormData(e.target); const result = await api.post('/api/save_sender_creds', formData); alert(result.message); if (result.success) closeModal('sender-modal'); });
    document.getElementById('student-emails-form').addEventListener('submit', async (e) => { e.preventDefault(); const formData = new FormData(e.target); const result = await api.post('/api/save_student_emails', formData); alert(result.message); if (result.success) closeModal('student-emails-modal'); });
    document.getElementById('add-student-btn').addEventListener('click', () => { openModal('add-student-modal'); });
    document.getElementById('cancel-add-student').addEventListener('click', () => closeModal('add-student-modal'));
    document.getElementById('cancel-add-photos').addEventListener('click', () => closeModal('add-photos-modal'));
    document.getElementById('config-sender-btn').addEventListener('click', async () => { const creds = await api.get('/api/get_sender_creds'); document.getElementById('sender-email').value = creds.email || ''; document.getElementById('sender-password').value = creds.password || ''; openModal('sender-modal'); });
    document.getElementById('cancel-sender').addEventListener('click', () => closeModal('sender-modal'));
    document.getElementById('manage-student-emails-btn').addEventListener('click', async () => {
        const data = await api.get('/api/get_student_emails'); const listDiv = document.getElementById('student-emails-list'); listDiv.innerHTML = '';
        data.students.forEach(student => { const email = data.emails[student] || ''; listDiv.innerHTML += `<div class="grid grid-cols-2 gap-2 items-center"><label class="font-medium">${student}</label><input type="email" name="${student}" value="${email}" placeholder="Email address" class="w-full p-2 border rounded"></div>`; });
        openModal('student-emails-modal');
    });
    document.getElementById('cancel-student-emails').addEventListener('click', () => closeModal('student-emails-modal'));
    document.getElementById('cancel-timetable').addEventListener('click', () => closeModal('timetable-modal'));
    document.getElementById('refresh-reports-btn').addEventListener('click', loadAll);
    document.getElementById('subject-filter').addEventListener('change', loadReports);
    document.getElementById('send-todays-report-btn').addEventListener('click', async () => {
        const selectedValue = document.getElementById('email-subject-filter').value;
        if (confirm(`This will email today's attendance report to all registered students. Proceed?`)) {
            const formData = new FormData();
            formData.append('subject', selectedValue);
            const result = await api.post('/api/send_todays_email', formData);
            alert(result.message);
        }
    });
    document.getElementById('send-overall-email-btn').addEventListener('click', async () => { if (confirm("This will email the DETAILED overall attendance summary to all registered students. Proceed?")) { const result = await api.post('/api/send_overall_email', new FormData()); alert(result.message); } });
    
    loadAll();
});
</script>
</body>
</html>
"""

# --- Server Routes ---

@app.route('/')
def home():
    return redirect(url_for('admin_login'))

@app.route('/admin')
def admin_login():
    return render_template_string(ADMIN_LOGIN_PAGE)

@app.route('/admin/login', methods=['POST'])
def handle_admin_login():
    if request.form.get('password') == ADMIN_PASSWORD:
        return redirect(url_for('admin_dashboard'))
    return 'Invalid Password', 401

@app.route('/admin/dashboard')
def admin_dashboard():
    return render_template_string(ADMIN_DASHBOARD_PAGE)

@app.route('/attend/<session_id>')
def attend_page(session_id):
    session = ATTENDANCE_SESSIONS.get(session_id)
    if not session or datetime.now() > session['expires_at']:
        if session_id in ATTENDANCE_SESSIONS:
            del ATTENDANCE_SESSIONS[session_id]
        return "Attendance session not found or has expired.", 404
    return render_template_string(render_student_page(session_id, session.get('subject', 'General')))

# --- API Endpoints ---

@app.route('/api/generate_link', methods=['POST'])
def api_generate_link():
    current_subject = get_current_subject()
    if not current_subject:
        return jsonify({'success': False, 'message': 'No class is currently in session according to the timetable.'})
    location_data = json.loads(request.form.get('location'))
    session_id = str(uuid.uuid4().hex[:10])
    expires_at = datetime.now() + timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    ATTENDANCE_SESSIONS[session_id] = {
        'admin_location': (location_data['latitude'], location_data['longitude']),
        'expires_at': expires_at,
        'subject': current_subject
    }
    full_url = request.host_url + 'attend/' + session_id
    return jsonify({'success': True, 'url': full_url, 'timeout': SESSION_TIMEOUT_MINUTES, 'subject': current_subject})

@app.route('/api/mark_attendance/<session_id>', methods=['POST'])
def api_mark_attendance(session_id):
    """
    Handles the core logic for verifying and marking a student's attendance.
    This updated version saves proof images in a nested directory structure:
    attendance_proofs/YYYY-MM-DD/SubjectName/student_id-student_name_time.jpg
    """
    session = ATTENDANCE_SESSIONS.get(session_id)
    if not session or datetime.now() > session['expires_at']:
        return jsonify({'success': False, 'message': 'Session expired.'}), 404

    try:
        data = request.get_json()
        admin_loc = session['admin_location']
        student_loc = (data['location']['latitude'], data['location']['longitude'])
        distance = geodesic(admin_loc, student_loc).meters

        if distance > MAX_DISTANCE_METERS:
            return jsonify({'success': False, 'message': f"Too far: {int(distance)}m. Must be within {MAX_DISTANCE_METERS}m."})

        student_id = data.get('student_id', '').strip()
        if not student_id:
            return jsonify({'success': False, 'message': 'Student ID is required.'})

        student_folder = find_folder_by_id(student_id)
        if not student_folder:
            return jsonify({'success': False, 'message': f'No student found with ID: {student_id}.'})

        # Decode the image data from the frontend
        image_data = data['image'].split(',')[1]
        img_buffer = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_buffer, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            return jsonify({'success': False, 'message': 'Could not decode image from webcam. Please try again.'})

        # Liveness detection: Check for a smile to prevent using static photos
        try:
            liveness_result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
            # Check if the dominant emotion is happy or if the happiness score is high
            has_smile = liveness_result[0]['dominant_emotion'] == 'happy' or liveness_result[0]['emotion']['happy'] > 0.7
            if not has_smile:
                return jsonify({'success': False, 'message': 'Liveness not detected. Please smile to confirm you are live.', 'requires_liveness': True})
        except Exception as e:
            app.logger.error(f"Liveness detection error: {e}")
            return jsonify({'success': False, 'message': 'Liveness check failed. Please try again.', 'requires_liveness': True})

        student_path = os.path.join(DATASET_PATH, student_folder)
        twins = load_twins()
        is_twin = any(student_id in pair for pair in twins.values())
        name = "" # Initialize name variable

        # --- Face Recognition Logic ---
        try:
            if is_twin:
                # Enhanced, stricter analysis for twins using multiple models
                dfs = DeepFace.find(
                    img_path=frame,
                    db_path=student_path,
                    model_name=["VGG-Face", "Age", "Gender"],
                    distance_metric="cosine",
                    enforce_detection=True,
                    silent=True
                )
                if not dfs or len(dfs) < 3 or any(df.empty for df in dfs):
                    return jsonify({'success': False, 'message': 'Face recognition failed for twin analysis.'})

                df_identity = dfs[0]
                distance_col = 'distance'
                if distance_col not in df_identity.columns:
                    return jsonify({'success': False, 'message': 'Internal error: Result format is unexpected.'})

                # Use a 10% stricter confidence threshold for twins
                potential_matches = df_identity[df_identity[distance_col] <= CONFIDENCE_THRESHOLD * 0.9]
                if potential_matches.empty:
                    return jsonify({'success': False, 'message': 'Face did not match with sufficient confidence.'})

                match_identity = potential_matches.iloc[0]['identity']
                folder_name = os.path.basename(os.path.dirname(match_identity))
                student_id_verified, name = folder_name.split('-', 1)

                if student_id_verified != student_id:
                    return jsonify({'success': False, 'message': 'Student ID mismatch with recognized face.'})
            else:
                # Standard analysis for non-twins
                dfs = DeepFace.find(
                    img_path=frame,
                    db_path=student_path,
                    model_name="VGG-Face",
                    distance_metric="cosine",
                    enforce_detection=True,
                    silent=True
                )
                if not dfs or dfs[0].empty:
                    return jsonify({'success': False, 'message': 'Face did not match the registered student.'})

                df = dfs[0]
                distance_col = 'distance'
                if distance_col not in df.columns:
                    return jsonify({'success': False, 'message': 'Internal error: Result format is unexpected.'})

                potential_matches = df[df[distance_col] <= CONFIDENCE_THRESHOLD]
                if potential_matches.empty:
                    return jsonify({'success': False, 'message': 'Face did not match with sufficient confidence.'})

                identity_path = potential_matches.iloc[0]['identity']
                folder_name = os.path.basename(os.path.dirname(identity_path))
                try:
                    student_id_verified, name = folder_name.split('-', 1)
                except ValueError:
                    name, student_id_verified = folder_name, "UnknownID"

                if student_id_verified != student_id:
                    return jsonify({'success': False, 'message': 'Student ID mismatch with recognized face.'})

            # --- Attendance Marking & File Saving ---
            if mark_attendance(name, session['subject']):
                now = datetime.now()
                date_str = now.strftime("%Y-%m-%d")
                time_str = now.strftime("%H%M%S")
                
                # *** MODIFICATION FOR NESTED PROOF FOLDER ***
                # Sanitize subject name to make it a valid folder name
                safe_subject = sanitize_filename(session['subject'])
                
                # Create the nested directory: /proofs/YYYY-MM-DD/SubjectName/
                proof_subject_folder = os.path.join(ATTENDANCE_PROOFS_PATH, date_str, safe_subject)
                os.makedirs(proof_subject_folder, exist_ok=True)
                
                safe_name = sanitize_filename(name)
                proof_filename = f"{student_id}-{safe_name}_{time_str}.jpg"
                proof_path = os.path.join(proof_subject_folder, proof_filename)
                
                cv2.imwrite(proof_path, frame)
                
                # Add the verified photo back to the dataset for continuous learning
                try:
                    retrain_filename = f"upload_{date_str}_{time_str}.jpg"
                    retrain_path = os.path.join(student_path, retrain_filename)
                    cv2.imwrite(retrain_path, frame)
                except Exception as e:
                    app.logger.error(f"Could not save retraining image: {e}")
                
                return jsonify({'success': True, 'message': f"Success! Welcome, {name}. Attendance marked for {session['subject']}."})
            else:
                return jsonify({'success': True, 'message': f"Info: Hello, {name}. You are already marked present for {session['subject']}."})

        except Exception as e:
            app.logger.error(f"Face recognition error: {e}", exc_info=True)
            return jsonify({'success': False, 'message': 'Face recognition failed. Please try again.'})

    except Exception as e:
        app.logger.error(f"A critical error occurred in api_mark_attendance: {e}", exc_info=True)
        return jsonify({'success': False, 'message': 'An unexpected server error occurred. Please try again.'})

# --- Management & Report APIs ---
@app.route('/api/students', methods=['GET'])
def api_get_students():
    return jsonify({'students': get_all_students()})

@app.route('/api/add_student', methods=['POST'])
def api_add_student():
    name = request.form.get('name')
    student_id = request.form.get('student_id')
    is_twin = request.form.get('is_twin', 'false').lower() == 'true'

    if not name or not name.strip() or not student_id or not student_id.strip():
        return jsonify({'success': False, 'message': 'Student Name and a unique Student ID are required.'})

    name, student_id = name.strip(), student_id.strip()
    
    if any(folder.startswith(f"{student_id}-") for folder in os.listdir(DATASET_PATH)):
        return jsonify({'success': False, 'message': f'Student ID "{student_id}" is already in use.'})

    images = request.files.getlist('images')
    if not images: return jsonify({'success': False, 'message': 'At least one image is required.'})

    student_folder_name = f"{student_id}-{name}"
    student_path = os.path.join(DATASET_PATH, student_folder_name)

    if os.path.exists(student_path):
        return jsonify({'success': False, 'message': f'A student with this ID and Name combination already exists.'})
    
    os.makedirs(student_path)
    for i, image in enumerate(images):
        unique_filename = f"upload_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}.jpg"
        image.save(os.path.join(student_path, unique_filename))
        
    db_file = os.path.join(DATASET_PATH, "representations_vgg_face.pkl")
    if os.path.exists(db_file): os.remove(db_file)

    if is_twin:
        twins = load_twins()
        twin_pair = None
        for pair in twins.values():
            if len(pair) < 2:
                twin_pair = pair
                break
        if twin_pair:
            twin_pair.append(student_id)
        else:
            twins[student_id] = [student_id]
        save_twins(twins)

    return jsonify({'success': True, 'message': f'Student "{name}" added successfully. Database will be updated.'})

@app.route('/api/add_photos', methods=['POST'])
def api_add_photos():
    student_name = request.form.get('name')
    if not student_name:
        return jsonify({'success': False, 'message': 'Student name was not provided.'})

    images = request.files.getlist('images')
    if not images:
        return jsonify({'success': False, 'message': 'No images were selected.'})

    student_folder = find_folder_by_name(student_name)
    if not student_folder:
        return jsonify({'success': False, 'message': f'Student "{student_name}" not found.'})

    student_path = os.path.join(DATASET_PATH, student_folder)
    for i, image in enumerate(images):
        unique_filename = f"upload_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}.jpg"
        image.save(os.path.join(student_path, unique_filename))

    db_file = os.path.join(DATASET_PATH, "representations_vgg_face.pkl")
    if os.path.exists(db_file):
        os.remove(db_file)
        
    return jsonify({'success': True, 'message': f'Added {len(images)} more photos for "{student_name}". Database will be updated.'})

@app.route('/api/rename_student', methods=['POST'])
def api_rename_student():
    old_name, new_name = request.form.get('old_name'), request.form.get('new_name')
    if not old_name or not new_name: return jsonify({'success': False, 'message': 'Both old and new names are required.'})

    old_folder = find_folder_by_name(old_name)
    if not old_folder: return jsonify({'success': False, 'message': 'Original student not found.'})
    
    student_id = old_folder.split('-', 1)[0]
    new_folder_name = f"{student_id}-{new_name.strip()}"
    old_path, new_path = os.path.join(DATASET_PATH, old_folder), os.path.join(DATASET_PATH, new_folder_name)

    if os.path.exists(new_path): return jsonify({'success': False, 'message': 'A student with the new name already exists for that ID.'})
    
    os.rename(old_path, new_path)
    
    # Iterate through dated subfolders to update records
    for date_folder in os.listdir(ATTENDANCE_RECORDS_PATH):
        folder_path = os.path.join(ATTENDANCE_RECORDS_PATH, date_folder)
        if os.path.isdir(folder_path):
            file_path = os.path.join(folder_path, "attendance.csv")
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    if 'Name' in df.columns and old_name in df['Name'].values:
                        df['Name'] = df['Name'].replace(old_name, new_name.strip())
                        df.to_csv(file_path, index=False)
                except (pd.errors.EmptyDataError, KeyError):
                    continue

    db_file = os.path.join(DATASET_PATH, "representations_vgg_face.pkl")
    if os.path.exists(db_file): os.remove(db_file)

    return jsonify({'success': True, 'message': f'Renamed "{old_name}" to "{new_name}".'})

@app.route('/api/delete_student', methods=['POST'])
def api_delete_student():
    name = request.form.get('name')
    folder_to_delete = find_folder_by_name(name)
    if not folder_to_delete: return jsonify({'success': False, 'message': 'Student not found.'})

    student_path = os.path.join(DATASET_PATH, folder_to_delete)
    if os.path.exists(student_path):
        shutil.rmtree(student_path)
        db_file = os.path.join(DATASET_PATH, "representations_vgg_face.pkl")
        if os.path.exists(db_file): os.remove(db_file)
        twins = load_twins()
        for pair in list(twins.values()):
            if any(folder_to_delete.split('-', 1)[0] in p for p in pair):
                pair[:] = [p for p in pair if p != folder_to_delete.split('-', 1)[0]]
                if not pair:
                    twins.pop(next(k for k, v in twins.items() if v == pair), None)
        save_twins(twins)
        return jsonify({'success': True, 'message': f'Deleted "{name}"'})
    return jsonify({'success': False, 'message': 'Student not found.'})

@app.route('/api/todays_attendance', methods=['GET'])
def api_todays_attendance():
    subject_filter = request.args.get('subject', 'all')
    all_students = get_all_students()
    
    date_str = datetime.now().strftime('%Y-%m-%d')
    file_path = os.path.join(ATTENDANCE_RECORDS_PATH, date_str, "attendance.csv")
    
    if not os.path.exists(file_path):
        return jsonify({'present': [], 'absent': all_students})

    try:
        df = pd.read_csv(file_path)
        if subject_filter != 'all':
            df = df[df['Subject'] == subject_filter]
        
        present_today = df['Name'].unique().tolist()
        absent_students = [s for s in all_students if s not in present_today]
        return jsonify({'present': df.to_dict('records'), 'absent': absent_students})
    except pd.errors.EmptyDataError:
        return jsonify({'present': [], 'absent': all_students})

@app.route('/api/overall_attendance', methods=['GET'])
def api_overall_attendance():
    subject_filter = request.args.get('subject', 'all')
    all_students = get_all_students()
    if not all_students: return jsonify({'report': []})

    # Scan subdirectories for record files
    record_files = []
    if os.path.exists(ATTENDANCE_RECORDS_PATH):
        for date_folder in os.listdir(ATTENDANCE_RECORDS_PATH):
            folder_path = os.path.join(ATTENDANCE_RECORDS_PATH, date_folder)
            if os.path.isdir(folder_path):
                csv_path = os.path.join(folder_path, "attendance.csv")
                if os.path.exists(csv_path):
                    record_files.append(csv_path)
    
    report = []

    if subject_filter == 'all':
        total_days = len(record_files)
        present_days_count = {student: 0 for student in all_students}
        for file in record_files:
            try:
                df = pd.read_csv(file)
                if 'Name' in df.columns:
                    for student in df['Name'].unique():
                        if student in present_days_count: present_days_count[student] += 1
            except pd.errors.EmptyDataError: continue
        for s in all_students:
            present = present_days_count.get(s, 0)
            report.append({'student': s, 'present_count': present, 'total_classes': total_days, 'percentage': (present / total_days * 100) if total_days > 0 else 0})
    else:
        total_subject_classes = 0
        present_subject_count = {student: 0 for student in all_students}
        for file in record_files:
            try:
                df = pd.read_csv(file)
                if 'Subject' in df.columns and subject_filter in df['Subject'].unique():
                    total_subject_classes += 1
                    if 'Name' in df.columns:
                        present_for_subject = df[df['Subject'] == subject_filter]['Name'].unique()
                        for student in present_for_subject:
                            if student in present_subject_count: present_subject_count[student] += 1
            except (pd.errors.EmptyDataError, KeyError): continue
        for s in all_students:
            present = present_subject_count.get(s, 0)
            report.append({'student': s, 'present_count': present, 'total_classes': total_subject_classes, 'percentage': (present / total_subject_classes * 100) if total_subject_classes > 0 else 0})
            
    return jsonify({'report': report})

# --- CORRECTED AND SELF-HEALING TIMETABLE FUNCTION ---
@app.route('/api/timetable', methods=['GET'])
def api_get_timetable():
    if not os.path.exists(TIMETABLE_FILE):
        return jsonify({'timetable': {}})
    
    try:
        with open(TIMETABLE_FILE, 'r') as f:
            timetable = json.load(f)
    except json.JSONDecodeError:
        return jsonify({'timetable': {}})

    # Self-healing logic to add missing IDs to legacy entries
    changes_made = False
    for day, slots in timetable.items():
        if isinstance(slots, list): # Ensure we're working with a list of slots
            for slot in slots:
                if isinstance(slot, dict) and 'id' not in slot:
                    slot['id'] = str(uuid.uuid4())
                    changes_made = True
    
    # If we added any IDs, save the updated file to fix it permanently
    if changes_made:
        try:
            with open(TIMETABLE_FILE, 'w') as f:
                json.dump(timetable, f, indent=4)
            app.logger.info("Timetable updated with new unique IDs for legacy slots.")
        except Exception as e:
            app.logger.error(f"Could not save updated timetable with new IDs: {e}")

    return jsonify({'timetable': timetable})

@app.route('/api/current_subject', methods=['GET'])
def api_get_current_subject():
    return jsonify({'subject': get_current_subject()})

@app.route('/api/subjects', methods=['GET'])
def api_get_subjects():
    subjects = set()
    if os.path.exists(TIMETABLE_FILE):
        with open(TIMETABLE_FILE, 'r') as f:
            try:
                timetable = json.load(f)
                for day in timetable.values():
                    for slot in day: subjects.add(slot['subject'])
            except json.JSONDecodeError: pass
    return jsonify({'subjects': sorted(list(subjects))})

@app.route('/api/save_slot', methods=['POST'])
def api_save_slot():
    day = request.form.get('slot-day')
    subject = request.form.get('slot-subject')
    start = request.form.get('slot-start')
    end = request.form.get('slot-end')
    slot_id = request.form.get('slot-id')

    if not all([day, subject, start, end]):
        return jsonify({'success': False, 'message': 'All fields are required.'})

    timetable = {}
    try:
        if os.path.exists(TIMETABLE_FILE):
            with open(TIMETABLE_FILE, 'r') as f:
                timetable = json.load(f)
        else:
            timetable = {"Monday": [], "Tuesday": [], "Wednesday": [], "Thursday": [], "Friday": [], "Saturday": [], "Sunday": []}
            os.makedirs(os.path.dirname(TIMETABLE_FILE) or '.', exist_ok=True)
    except Exception as e:
        app.logger.error(f"Error reading timetable: {e}")
        return jsonify({'success': False, 'message': f'Error accessing timetable file: {str(e)}'})

    if day not in timetable:
        timetable[day] = []
        
    if slot_id:
        slot_to_update = next((slot for slot in timetable[day] if slot.get('id') == slot_id), None)
        if slot_to_update:
            slot_to_update['subject'] = subject
            slot_to_update['start'] = start
            slot_to_update['end'] = end
            message = f"Slot for {subject} on {day} updated successfully."
        else:
            return jsonify({'success': False, 'message': f'Could not find slot with ID {slot_id} to update.'})
    else:
        new_slot = {
            'id': str(uuid.uuid4()),
            'subject': subject,
            'start': start,
            'end': end
        }
        timetable[day].append(new_slot)
        message = f'Slot for {subject} on {day} saved successfully.'
    
    timetable[day].sort(key=lambda x: x['start'])

    try:
        with open(TIMETABLE_FILE, 'w') as f:
            json.dump(timetable, f, indent=4)
        return jsonify({'success': True, 'message': message})
    except Exception as e:
        app.logger.error(f"Failed to save timetable: {e}")
        return jsonify({'success': False, 'message': f'Failed to save slot due to a file error: {str(e)}'})

@app.route('/api/delete_slot', methods=['POST'])
def api_delete_slot():
    day = request.form.get('day')
    slot_id = request.form.get('id')

    if not os.path.exists(TIMETABLE_FILE):
        return jsonify({'success': False, 'message': 'Timetable not found.'})

    try:
        with open(TIMETABLE_FILE, 'r') as f:
            timetable = json.load(f)
    except json.JSONDecodeError:
        return jsonify({'success': False, 'message': 'Timetable file is corrupted.'})

    if day in timetable:
        original_length = len(timetable[day])
        timetable[day] = [slot for slot in timetable[day] if slot.get('id') != slot_id]
        
        if len(timetable[day]) < original_length:
            with open(TIMETABLE_FILE, 'w') as f:
                json.dump(timetable, f, indent=4)
            return jsonify({'success': True, 'message': 'Slot deleted.'})

    return jsonify({'success': False, 'message': 'Slot not found or already deleted.'})


@app.route('/api/get_sender_creds', methods=['GET'])
def api_get_sender_creds():
    if not os.path.exists(SENDER_GMAIL_FILE): return jsonify({'email': '', 'password': ''})
    with open(SENDER_GMAIL_FILE, 'r') as f: return jsonify(json.load(f))

@app.route('/api/save_sender_creds', methods=['POST'])
def api_save_sender_creds():
    email, password = request.form.get('sender-email'), request.form.get('sender-password')
    if not email or not password: return jsonify({'success': False, 'message': 'Email and Password are required.'})
    with open(SENDER_GMAIL_FILE, 'w') as f: json.dump({'email': email, 'password': password}, f, indent=4)
    return jsonify({'success': True, 'message': 'Sender credentials saved.'})

@app.route('/api/get_student_emails', methods=['GET'])
def api_get_student_emails():
    emails = {}
    if os.path.exists(STUDENT_EMAILS_FILE):
        with open(STUDENT_EMAILS_FILE, 'r') as f: emails = json.load(f)
    return jsonify({'students': get_all_students(), 'emails': emails})

@app.route('/api/save_student_emails', methods=['POST'])
def api_save_student_emails():
    emails = {name: email for name, email in request.form.items()}
    with open(STUDENT_EMAILS_FILE, 'w') as f: json.dump(emails, f, indent=4)
    return jsonify({'success': True, 'message': 'Student emails saved.'})

@app.route('/api/send_todays_email', methods=['POST'])
def api_send_todays_email():
    try:
        subject_filter = request.form.get('subject')
        date_str = datetime.now().strftime("%Y-%m-%d")
        file_path = os.path.join(ATTENDANCE_RECORDS_PATH, date_str, "attendance.csv")
        
        present_df = pd.DataFrame()
        if os.path.exists(file_path):
            try:
                present_df = pd.read_csv(file_path)
            except pd.errors.EmptyDataError:
                pass
        
        if subject_filter == 'all_today':
            email_subject = f"Attendance Summary for {date_str}"
            def content_generator(name):
                student_records = present_df[present_df['Name'] == name]
                if not student_records.empty:
                    subjects = ", ".join(student_records['Subject'].tolist())
                    return f"Hi {name},\n\nOn {date_str}, you were marked PRESENT for: {subjects}.\n\nThank you."
                else:
                    return f"Hi {name},\n\nOn {date_str}, you were marked ABSENT for all subjects held.\n\nThank you."
        else:
            email_subject = f"Attendance for {subject_filter} on {date_str}"
            subject_records = present_df[present_df['Subject'] == subject_filter]
            present_students = subject_records['Name'].unique().tolist()
            def content_generator(name):
                if name in present_students:
                    return f"Hi {name},\n\nYou were marked PRESENT for {subject_filter} today, {date_str}.\n\nThank you."
                else:
                    return f"Hi {name},\n\nYou were marked ABSENT for {subject_filter} today, {date_str}.\n\nThank you."

        return _send_email_logic(email_subject, content_generator)
    except Exception as e:
        app.logger.error(f"Error in api_send_todays_email: {e}")
        return jsonify({'success': False, 'message': f'An unexpected server error occurred: {e}'})

# --- MODIFIED FUNCTION ---
def _get_detailed_overall_report():
    all_students = get_all_students()
    if not all_students: return []

    subjects = set()
    if os.path.exists(TIMETABLE_FILE):
        with open(TIMETABLE_FILE, 'r') as f:
            try:
                timetable = json.load(f)
                for day in timetable.values():
                    for slot in day: subjects.add(slot['subject'])
            except json.JSONDecodeError: pass
    all_subjects = sorted(list(subjects))

    record_files = []
    if os.path.exists(ATTENDANCE_RECORDS_PATH):
        for date_folder in os.listdir(ATTENDANCE_RECORDS_PATH):
            folder_path = os.path.join(ATTENDANCE_RECORDS_PATH, date_folder)
            if os.path.isdir(folder_path):
                csv_path = os.path.join(folder_path, "attendance.csv")
                if os.path.exists(csv_path):
                    record_files.append(csv_path)
    
    record_dfs = []
    for file in record_files:
        try:
            record_dfs.append(pd.read_csv(file))
        except pd.errors.EmptyDataError:
            continue
    
    final_report = []

    for student in all_students:
        student_report = {
            "student_name": student,
            "subject_breakdown": [],
        }
        
        grand_total_present = 0
        grand_total_classes = 0

        if all_subjects:
            for subject in all_subjects:
                total_subject_classes = 0
                present_subject_classes = 0
                for df in record_dfs:
                    if 'Subject' in df.columns and subject in df['Subject'].unique():
                        total_subject_classes += 1
                        if 'Name' in df.columns and not df[(df['Name'] == student) & (df['Subject'] == subject)].empty:
                            present_subject_classes += 1
                
                # Add subject to breakdown regardless of whether classes were held, for consistency
                student_report["subject_breakdown"].append({
                    "subject": subject,
                    "present": present_subject_classes,
                    "total": total_subject_classes,
                })
                grand_total_present += present_subject_classes
                grand_total_classes += total_subject_classes
        
        student_report["grand_total_present"] = grand_total_present
        student_report["grand_total_classes"] = grand_total_classes
            
        final_report.append(student_report)
        
    return final_report

# --- MODIFIED FUNCTION ---
@app.route('/api/send_overall_email', methods=['POST'])
def api_send_overall_email():
    try:
        report_data = _get_detailed_overall_report()
        if not report_data:
            return jsonify({'success': False, 'message': 'No attendance data to report.'})
        
        def content_generator(name):
            student_data = next((item for item in report_data if item['student_name'] == name), None)
            if not student_data: return None

            body = f"Hi {name},\n\nHere is your detailed overall attendance summary:\n\n"
            
            if student_data["subject_breakdown"]:
                body += "--- Subject-wise Attendance ---\n"
                for item in student_data['subject_breakdown']:
                    body += f"- {item['subject']}: Attended {item['present']} out of {item['total']} classes held.\n"
            else:
                body += "No subject-specific attendance recorded.\n"
                
            body += f"\n--- Overall Summary ---\n"
            body += f"Total Attendance: You have attended {student_data['grand_total_present']} out of {student_data['grand_total_classes']} total classes held across all subjects.\n\n"
            body += "Thank you."
            
            return body
            
        return _send_email_logic("Detailed Overall Attendance Summary", content_generator)
    except Exception as e:
        app.logger.error(f"Error in api_send_overall_email: {e}")
        return jsonify({'success': False, 'message': f'An unexpected server error occurred: {e}'})

def _send_email_logic(subject, content_generator):
    """Handles the logic of connecting to Gmail and sending emails."""
    try:
        if not os.path.exists(SENDER_GMAIL_FILE) or not os.path.exists(STUDENT_EMAILS_FILE):
            return jsonify({'success': False, 'message': 'Sender or student emails not configured. Please set them up in the admin dashboard.'})
        
        with open(SENDER_GMAIL_FILE, 'r') as f: sender_creds = json.load(f)
        with open(STUDENT_EMAILS_FILE, 'r') as f: student_emails = json.load(f)
        
        SENDER_EMAIL = sender_creds.get('email')
        SENDER_PASSWORD = sender_creds.get('password')
        
        if not SENDER_EMAIL or not SENDER_PASSWORD:
            return jsonify({'success': False, 'message': 'Sender credentials are incomplete.'})

        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        try:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
        except smtplib.SMTPAuthenticationError:
            return jsonify({'success': False, 'message': 'Gmail login failed. Check your email and App Password.'})

        sent_count = 0
        errors = []
        for student_name, recipient_email in student_emails.items():
            if not recipient_email:
                continue
            
            body = content_generator(student_name)
            if not body:
                continue
            
            try:
                msg = EmailMessage()
                msg['Subject'] = subject
                msg['From'] = SENDER_EMAIL
                msg['To'] = recipient_email
                msg.set_content(body)
                server.send_message(msg)
                sent_count += 1
            except Exception as e:
                errors.append(f"Could not send to {student_name} ({recipient_email}): {str(e)}")

        server.quit()

        if errors:
            return jsonify({'success': False, 'message': f'Completed with errors. Sent {sent_count} emails. Failed: {", ".join(errors)}'})
        
        return jsonify({'success': True, 'message': f'Successfully sent {sent_count} emails.'})
    
    except json.JSONDecodeError:
        return jsonify({'success': False, 'message': 'Error reading configuration file. Please re-save sender and student emails.'})
    except Exception as e:
        app.logger.error(f"Critical error in _send_email_logic: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'A critical error occurred: {str(e)}'})

# --- Main Entry Point ---
if __name__ == '__main__':
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(ATTENDANCE_RECORDS_PATH, exist_ok=True)
    os.makedirs(ATTENDANCE_PROOFS_PATH, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
