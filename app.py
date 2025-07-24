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
MAX_DISTANCE_METERS = 15
SESSION_TIMEOUT_MINUTES = 5
DATASET_PATH = "dataset"
ATTENDANCE_RECORDS_PATH = "attendance_records"
SENDER_GMAIL_FILE = "sender_gmail.json"
STUDENT_EMAILS_FILE = "student_emails.json"
TIMETABLE_FILE = "timetable.json"

# --- Core Logic & Helper Functions ---

def get_all_students():
    if not os.path.exists(DATASET_PATH): return []
    return sorted([f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))])

def mark_attendance(name, subject):
    date_str = datetime.now().strftime("%Y-%m-%d")
    file_path = os.path.join(ATTENDANCE_RECORDS_PATH, f"attendance-{date_str}.csv")
    time_now = datetime.now().strftime("%H:%M:%S")

    new_entry = pd.DataFrame([[name, time_now, subject]], columns=["Name", "Time", "Subject"])

    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            if 'Subject' not in df.columns:
                df['Subject'] = 'Unknown'
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=["Name", "Time", "Subject"])
    else:
        df = pd.DataFrame(columns=["Name", "Time", "Subject"])

    if not df[(df['Name'] == name) & (df['Subject'] == subject)].empty:
        return False  # Already marked for this subject today

    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(file_path, index=False)
    return True

def get_current_subject():
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
    <html lang="en">
    <head>
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
            <p id="main-prompt" class="text-gray-500 mb-6">Enable location to begin.</p>
            <div class="relative w-full aspect-video bg-black rounded-lg overflow-hidden mb-6 border-4 border-gray-200">
                <video id="video-feed" class="w-full h-full object-cover" autoplay playsinline></video>
                <div id="loading-overlay" class="absolute inset-0 bg-black bg-opacity-75 flex items-center justify-center"><p class="text-white text-xl">Starting Camera...</p></div>
            </div>
            <button id="mark-attendance-btn" class="w-full bg-blue-600 text-white font-bold py-4 px-6 rounded-lg text-xl hover:bg-blue-700 disabled:bg-gray-400" disabled>Mark My Attendance</button>
            <div id="status-display" class="status-box mt-6 p-4 rounded-lg border-2 text-lg font-semibold text-center opacity-0">{message if message else 'Status will appear here'}</div>
        </div>
        <canvas id="canvas" class="hidden"></canvas>
        <script>
            const video = document.getElementById('video-feed'), canvas = document.getElementById('canvas'), markButton = document.getElementById('mark-attendance-btn');
            const statusDisplay = document.getElementById('status-display'), loadingOverlay = document.getElementById('loading-overlay'), mainPrompt = document.getElementById('main-prompt');
            const sessionId = "{session_id}"; let studentLocation = null, isProcessing = false;
            function showStatus(message, type = 'info') {{ statusDisplay.textContent = message; statusDisplay.className = 'status-box mt-6 p-4 rounded-lg border-2 text-lg font-semibold text-center'; statusDisplay.classList.add(`status-${{type}}`); statusDisplay.style.opacity = 1; }}
            async function setupDevice() {{
                showStatus("Requesting location permission...", "processing"); mainPrompt.textContent = "Please allow location access to continue.";
                navigator.geolocation.getCurrentPosition( (position) => {{ studentLocation = {{ latitude: position.coords.latitude, longitude: position.coords.longitude }}; showStatus("Location found! Ready for attendance.", "success"); mainPrompt.textContent = "Point camera at face and mark your attendance."; markButton.disabled = false; }}, (err) => {{ showStatus("Location access denied. You cannot mark attendance.", "error"); mainPrompt.textContent = "Location is required. Please enable it and refresh."; markButton.disabled = true; }}, {{ enableHighAccuracy: true }} );
                try {{ const stream = await navigator.mediaDevices.getUserMedia({{ video: {{ facingMode: 'user' }} }}); video.srcObject = stream; video.onloadedmetadata = () => {{ loadingOverlay.style.display = 'none'; }}; }} catch (err) {{ loadingOverlay.innerHTML = `<p class="text-red-400 text-xl px-4">Error: Could not access camera.</p>`; markButton.disabled = true; }}
            }}
            markButton.addEventListener('click', async () => {{
                if (isProcessing || !studentLocation) return;
                isProcessing = true; markButton.disabled = true; markButton.textContent = 'Processing...'; showStatus('Capturing & verifying...', 'processing');
                canvas.width = video.videoWidth; canvas.height = video.videoHeight; const context = canvas.getContext('2d');
                context.translate(canvas.width, 0); context.scale(-1, 1); context.drawImage(video, 0, 0, canvas.width, canvas.height);
                const imageData = canvas.toDataURL('image/jpeg');
                try {{
                    const response = await fetch(`/api/mark_attendance/${{sessionId}}`, {{ method: 'POST', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify({{ image: imageData, location: studentLocation }}) }});
                    const result = await response.json(); showStatus(result.message, result.success ? 'success' : 'error');
                }} catch (error) {{ showStatus('Error: Could not connect to the server.', 'error'); }} finally {{ setTimeout(() => {{ isProcessing = false; markButton.disabled = false; markButton.textContent = 'Mark My Attendance'; }}, 3000); }}
            }});
            setupDevice();
        </script>
    </body>
    </html>
    """

ADMIN_LOGIN_PAGE = """
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Admin Login</title><script src="https://cdn.tailwindcss.com"></script></head><body class="bg-gray-200 flex items-center justify-center h-screen"><div class="bg-white p-8 rounded-lg shadow-md w-96 text-center"><h1 class="text-2xl font-bold mb-6">Admin Login</h1><form method="POST" action="/admin/login"><input type="password" name="password" placeholder="Enter Password" class="w-full p-2 border rounded mb-4"><button type="submit" class="w-full bg-blue-600 text-white p-2 rounded hover:bg-blue-700">Login</button></form></div></body></html>
"""

ADMIN_DASHBOARD_PAGE_V5 = """
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
                        <button id="send-today-email-btn" class="w-full bg-red-600 text-white p-3 rounded hover:bg-red-700">Send Today's Report</button>
                        <button id="send-overall-email-btn" class="w-full bg-red-600 text-white p-3 rounded hover:bg-red-700">Send Overall Report</button>
                    </div>
                </div>
            </div>
        </div>
        <div class="bg-white p-6 rounded-lg shadow-md mt-8">
            <h2 class="text-2xl font-bold mb-4">4. Manage Timetable</h2>
            <div id="timetable-container" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"></div>
        </div>
    </div>
    <div id="add-student-modal" class="modal-overlay hidden"><div class="modal-content"><h3 class="text-xl font-bold mb-4">Add New Student</h3><form id="add-student-form"><input type="text" id="new-student-name" name="name" placeholder="Student's Full Name" class="w-full p-2 border rounded mb-2" required><label class="block mb-2 text-sm font-medium text-gray-700">Upload Images:</label><input type="file" id="student-images" name="images" class="w-full" multiple required><div class="flex justify-end gap-4 mt-4"><button type="button" id="cancel-add-student" class="bg-gray-300 p-2 px-4 rounded">Cancel</button><button type="submit" class="bg-blue-600 text-white p-2 px-4 rounded">Add</button></div></form></div></div>
    <div id="sender-modal" class="modal-overlay hidden"><div class="modal-content"><h3 class="text-xl font-bold mb-4">Configure Sender Gmail</h3><p class="text-sm text-gray-600 mb-4">Enter your Gmail and an <a href='https://myaccount.google.com/apppasswords' target='_blank' class='text-blue-600'>App Password</a>.</p><form id="sender-form"><input type="email" id="sender-email" name="sender-email" placeholder="your.email@gmail.com" class="w-full p-2 border rounded mb-2" required><input type="password" id="sender-password" name="sender-password" placeholder="Gmail App Password" class="w-full p-2 border rounded mb-4" required><div class="flex justify-end gap-4"><button type="button" id="cancel-sender" class="bg-gray-300 p-2 px-4 rounded">Cancel</button><button type="submit" class="bg-blue-600 text-white p-2 px-4 rounded">Save</button></div></form></div></div>
    <div id="student-emails-modal" class="modal-overlay hidden"><div class="modal-content"><h3 class="text-xl font-bold mb-4">Manage Student Emails</h3><form id="student-emails-form"><div id="student-emails-list" class="space-y-2 mb-4 max-h-80 overflow-y-auto"></div><div class="flex justify-end gap-4"><button type="button" id="cancel-student-emails" class="bg-gray-300 p-2 px-4 rounded">Cancel</button><button type="submit" class="bg-blue-600 text-white p-2 px-4 rounded">Save All</button></div></form></div></div>
    <div id="timetable-modal" class="modal-overlay hidden"><div class="modal-content"><h3 class="text-xl font-bold mb-4">Add/Edit Class Slot</h3><form id="timetable-form"><input type="hidden" id="slot-day" name="slot-day"><input type="hidden" id="slot-index" name="slot-index" value="-1"><input type="text" id="slot-subject" name="slot-subject" placeholder="Subject Name" class="w-full p-2 border rounded mb-2" required><input type="time" id="slot-start" name="slot-start" class="w-full p-2 border rounded mb-2" required><input type="time" id="slot-end" name="slot-end" class="w-full p-2 border rounded mb-4" required><div class="flex justify-end gap-4"><button type="button" id="cancel-timetable" class="bg-gray-300 p-2 px-4 rounded">Cancel</button><button type="submit" class="bg-blue-600 text-white p-2 px-4 rounded">Save Slot</button></div></form></div></div>
<script>
document.addEventListener('DOMContentLoaded', () => {
    const api = {
        get: async (url) => fetch(url).then(res => res.json()),
        post: async (url, body) => fetch(url, { method: 'POST', body: body }).then(res => res.json())
    };
    const modals = {
        'add-student-modal': document.getElementById('add-student-modal'),
        'sender-modal': document.getElementById('sender-modal'),
        'student-emails-modal': document.getElementById('student-emails-modal'),
        'timetable-modal': document.getElementById('timetable-modal')
    };
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
            // MODIFIED: Update total students count
            document.getElementById('total-students-count').textContent = data.students ? data.students.length : 0;
            studentListDiv.innerHTML = '';
            if (data.students && data.students.length > 0) {
                data.students.forEach(name => {
                    const el = document.createElement('div');
                    el.className = 'flex items-center justify-between bg-gray-50 p-2 rounded';
                    el.innerHTML = `<span class="font-medium">${name}</span><div><button class="text-sm text-blue-500 hover:underline mr-2 rename-btn" data-name="${name}">Rename</button><button class="text-sm text-red-500 hover:underline delete-btn" data-name="${name}">Delete</button></div>`;
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
            // MODIFIED: Update present and absent counts
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
                todayData.absent.forEach(name => {
                    const p = document.createElement('p'); p.textContent = name;
                    absentListDiv.appendChild(p);
                });
            } else { absentListDiv.innerHTML = '<p class="text-gray-500">No students absent.</p>'; }
        } catch (e) {
            presentListDiv.innerHTML = '<p class="text-red-500">Error loading report.</p>';
            absentListDiv.innerHTML = '<p class="text-red-500">Error loading report.</p>';
        }
        const overallTableDiv = document.getElementById('overall-report-table');
        overallTableDiv.innerHTML = '<p class="text-gray-500">Loading...</p>';
        try {
            const overallData = await api.get(`/api/overall_attendance?subject=${subjectFilter}`);
            if (overallData.report) {
                let tableHTML = `<table class="w-full text-left"><thead class="bg-gray-100"><tr><th class="p-2">Name</th><th>Present</th><th>Total Classes</th><th>%</th></tr></thead><tbody>`;
                overallData.report.forEach(item => { tableHTML += `<tr class="border-b"><td class="p-2 font-medium">${item.student}</td><td>${item.present_count}</td><td>${item.total_classes}</td><td class="font-semibold">${item.percentage.toFixed(1)}%</td></tr>`; });
                tableHTML += '</tbody></table>'; overallTableDiv.innerHTML = tableHTML;
            }
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
            slots.forEach((slot, index) => {
                dayHtml += `<div class="flex justify-between items-center bg-white p-2 rounded shadow-sm cursor-pointer hover:bg-gray-100 slot-item" data-day="${day}" data-index="${index}"><span>${slot.subject} (${slot.start}-${slot.end})</span><button class="text-red-500 text-xs delete-slot-btn" data-day="${day}" data-index="${index}">Delete</button></div>`;
            });
            dayHtml += `</div><button class="mt-2 text-blue-600 text-sm w-full text-center add-slot-btn" data-day="${day}">+ Add Slot</button></div>`;
            container.innerHTML += dayHtml;
        });
    }
    async function loadSubjectFilter() {
        const filter = document.getElementById('subject-filter');
        const data = await api.get('/api/subjects');
        filter.innerHTML = '<option value="all">All Subjects</option>';
        if(data.subjects) { data.subjects.forEach(s => { filter.innerHTML += `<option value="${s}">${s}</option>`; }); }
    }
    document.body.addEventListener('click', async (e) => {
        if (e.target.closest('.rename-btn')) {
            const oldName = e.target.closest('.rename-btn').dataset.name;
            const newName = prompt(`Enter new name for "${oldName}":`);
            if (newName && newName.trim() !== "") {
                const formData = new FormData(); formData.append('old_name', oldName); formData.append('new_name', newName.trim());
                const result = await api.post('/api/rename_student', formData); alert(result.message); loadAll();
            }
        }
        if (e.target.closest('.delete-btn')) {
            const name = e.target.closest('.delete-btn').dataset.name;
            if (confirm(`Are you sure you want to delete "${name}"?`)) {
                const formData = new FormData(); formData.append('name', name);
                const result = await api.post('/api/delete_student', formData); alert(result.message); loadAll();
            }
        }
        if (e.target.closest('.add-slot-btn')) {
            const day = e.target.closest('.add-slot-btn').dataset.day;
            const form = document.getElementById('timetable-form'); form.reset();
            document.getElementById('slot-day').value = day;
            document.getElementById('slot-index').value = -1;
            openModal('timetable-modal');
        }
        if (e.target.closest('.slot-item')) {
            const day = e.target.closest('.slot-item').dataset.day;
            const index = e.target.closest('.slot-item').dataset.index;
            const form = document.getElementById('timetable-form'); form.reset();
            document.getElementById('slot-day').value = day;
            document.getElementById('slot-index').value = index;
            const data = await api.get('/api/timetable');
            const slot = data.timetable[day][index];
            document.getElementById('slot-subject').value = slot.subject;
            document.getElementById('slot-start').value = slot.start;
            document.getElementById('slot-end').value = slot.end;
            openModal('timetable-modal');
        }
        if (e.target.closest('.delete-slot-btn')) {
            e.stopPropagation();
            const day = e.target.closest('.delete-slot-btn').dataset.day;
            const index = e.target.closest('.delete-slot-btn').dataset.index;
            if (!confirm('Delete this slot?')) return;
            const formData = new FormData(); formData.append('day', day); formData.append('index', index);
            const result = await api.post('/api/delete_slot', formData);
            if (result.success) loadAll(); else alert(result.message);
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
        e.preventDefault(); const name = document.getElementById('new-student-name').value, files = document.getElementById('student-images').files;
        if (!name || files.length < 1) { alert("Please provide a name and at least one image."); return; }
        const formData = new FormData(); formData.append('name', name);
        for (let i = 0; i < files.length; i++) { formData.append('images', files[i]); }
        const result = await api.post('/api/add_student', formData); alert(result.message);
        if (result.success) { closeModal('add-student-modal'); e.target.reset(); loadAll(); }
    });
    document.getElementById('timetable-form').addEventListener('submit', async (e) => {
        e.preventDefault(); const formData = new FormData(e.target);
        const result = await api.post('/api/save_slot', formData);
        if (result.success) { closeModal('timetable-modal'); e.target.reset(); loadAll(); } else { alert(result.message); }
    });
    document.getElementById('sender-form').addEventListener('submit', async (e) => { e.preventDefault(); const formData = new FormData(e.target); const result = await api.post('/api/save_sender_creds', formData); alert(result.message); if (result.success) closeModal('sender-modal'); });
    document.getElementById('student-emails-form').addEventListener('submit', async (e) => { e.preventDefault(); const formData = new FormData(e.target); const result = await api.post('/api/save_student_emails', formData); alert(result.message); if (result.success) closeModal('student-emails-modal'); });
    document.getElementById('add-student-btn').addEventListener('click', () => openModal('add-student-modal'));
    document.getElementById('cancel-add-student').addEventListener('click', () => closeModal('add-student-modal'));
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
    document.getElementById('send-today-email-btn').addEventListener('click', async () => { if (confirm("Email today's report?")) { const result = await api.post('/api/send_todays_email', new FormData()); alert(result.message); } });
    document.getElementById('send-overall-email-btn').addEventListener('click', async () => { if (confirm("Email overall report?")) { const result = await api.post('/api/send_overall_email', new FormData()); alert(result.message); } });

    loadAll();
});
</script>
</body>
</html>
"""

# --- Server Routes ---

@app.route('/')
def home(): return redirect(url_for('admin_login'))

@app.route('/admin')
def admin_login(): return render_template_string(ADMIN_LOGIN_PAGE)

@app.route('/admin/login', methods=['POST'])
def handle_admin_login():
    if request.form.get('password') == ADMIN_PASSWORD: return redirect(url_for('admin_dashboard'))
    return 'Invalid Password', 401

@app.route('/admin/dashboard')
def admin_dashboard(): return render_template_string(ADMIN_DASHBOARD_PAGE_V5)

@app.route('/attend/<session_id>')
def attend_page(session_id):
    session = ATTENDANCE_SESSIONS.get(session_id)
    if not session or datetime.now() > session['expires_at']:
        if session_id in ATTENDANCE_SESSIONS: del ATTENDANCE_SESSIONS[session_id]
        return "Attendance session not found or has expired.", 404
    return render_template_string(render_student_page(session_id, session.get('subject', 'General')))

# --- API Endpoints ---

@app.route('/api/generate_link', methods=['POST'])
def api_generate_link():
    current_subject = get_current_subject()
    if not current_subject: return jsonify({'success': False, 'message': 'No class is currently in session according to the timetable.'})
    location_data = json.loads(request.form.get('location'))
    session_id = str(uuid.uuid4().hex[:10])
    expires_at = datetime.now() + timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    ATTENDANCE_SESSIONS[session_id] = { 'admin_location': (location_data['latitude'], location_data['longitude']), 'expires_at': expires_at, 'subject': current_subject }
    full_url = request.host_url + 'attend/' + session_id
    return jsonify({'success': True, 'url': full_url, 'timeout': SESSION_TIMEOUT_MINUTES, 'subject': current_subject})

@app.route('/api/mark_attendance/<session_id>', methods=['POST'])
def api_mark_attendance(session_id):
    session = ATTENDANCE_SESSIONS.get(session_id)
    if not session or datetime.now() > session['expires_at']:
        return jsonify({'success': False, 'message': 'Session expired.'}), 404
    
    data = request.get_json()
    admin_loc = session['admin_location']
    student_loc = (data['location']['latitude'], data['location']['longitude'])
    distance = geodesic(admin_loc, student_loc).meters
    
    if distance > MAX_DISTANCE_METERS:
        return jsonify({'success': False, 'message': f"Too far: {int(distance)}m. Must be within {MAX_DISTANCE_METERS}m."})

    try:
        image_data = data['image'].split(',')[1]
        img_buffer = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_buffer, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        dfs = DeepFace.find(
            img_path=frame,
            db_path=DATASET_PATH,
            model_name="VGG-Face",
            enforce_detection=True,
            silent=True
        )

        if not dfs or dfs[0].empty:
            return jsonify({'success': False, 'message': 'No registered student was recognized.'})
        
        name = os.path.basename(os.path.dirname(dfs[0].iloc[0]['identity']))
        
        if mark_attendance(name, session['subject']):
            return jsonify({'success': True, 'message': f"Success! Welcome, {name}. Attendance marked for {session['subject']}."})
        else:
            return jsonify({'success': True, 'message': f"Info: Hello, {name}. You are already marked present for {session['subject']}."})

    except ValueError as e:
        if "Face could not be detected" in str(e):
            return jsonify({
                'success': False,
                'message': 'No face detected. Please ensure your face is clear, well-lit, and centered in the frame.'
            }), 400
        return jsonify({'success': False, 'message': f'Verification error: {str(e)}'}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred: {e}")
        return jsonify({'success': False, 'message': 'An unexpected server error occurred. Please try again.'}), 500


# --- Management & Report APIs ---
@app.route('/api/students', methods=['GET'])
def api_get_students(): return jsonify({'students': get_all_students()})

@app.route('/api/add_student', methods=['POST'])
def api_add_student():
    name = request.form.get('name')
    if not name or not name.strip():
        return jsonify({'success': False, 'message': 'Student name cannot be empty.'})
    name = name.strip()
    
    images = request.files.getlist('images')
    if not images:
        return jsonify({'success': False, 'message': 'At least one image is required.'})

    student_path = os.path.join(DATASET_PATH, name)
    if os.path.exists(student_path):
        return jsonify({'success': False, 'message': f'Student "{name}" already exists.'})
    
    os.makedirs(student_path)
    for i, image in enumerate(images):
        image.save(os.path.join(student_path, f"upload_{i}.jpg"))
        
    db_file = os.path.join(DATASET_PATH, "representations_vgg_face.pkl")
    if os.path.exists(db_file):
        os.remove(db_file)
        
    return jsonify({'success': True, 'message': f'Student "{name}" added successfully. Database will be updated.'})

@app.route('/api/rename_student', methods=['POST'])
def api_rename_student():
    old_name, new_name = request.form.get('old_name'), request.form.get('new_name')
    if not old_name or not new_name:
        return jsonify({'success': False, 'message': 'Both old and new names are required.'})

    old_path, new_path = os.path.join(DATASET_PATH, old_name), os.path.join(DATASET_PATH, new_name)
    if not os.path.exists(old_path):
        return jsonify({'success': False, 'message': 'Original student not found.'})
    if os.path.exists(new_path):
        return jsonify({'success': False, 'message': 'A student with the new name already exists.'})
    
    os.rename(old_path, new_path)
    
    for filename in os.listdir(ATTENDANCE_RECORDS_PATH):
        if filename.endswith('.csv'):
            file_path = os.path.join(ATTENDANCE_RECORDS_PATH, filename)
            try:
                df = pd.read_csv(file_path)
                if old_name in df['Name'].values:
                    df['Name'] = df['Name'].replace(old_name, new_name)
                    df.to_csv(file_path, index=False)
            except (pd.errors.EmptyDataError, KeyError):
                continue

    db_file = os.path.join(DATASET_PATH, "representations_vgg_face.pkl")
    if os.path.exists(db_file):
        os.remove(db_file)

    return jsonify({'success': True, 'message': f'Renamed "{old_name}" to "{new_name}" and updated records.'})


@app.route('/api/delete_student', methods=['POST'])
def api_delete_student():
    name = request.form.get('name')
    student_path = os.path.join(DATASET_PATH, name)
    if os.path.exists(student_path):
        shutil.rmtree(student_path)
        db_file = os.path.join(DATASET_PATH, "representations_vgg_face.pkl")
        if os.path.exists(db_file):
            os.remove(db_file)
        return jsonify({'success': True, 'message': f'Deleted "{name}".'})
    return jsonify({'success': False, 'message': 'Student not found.'})

### MODIFIED ###
@app.route('/api/todays_attendance', methods=['GET'])
def api_todays_attendance():
    subject_filter = request.args.get('subject', 'all')
    all_students = get_all_students()
    file_path = os.path.join(ATTENDANCE_RECORDS_PATH, f"attendance-{datetime.now().strftime('%Y-%m-%d')}.csv")
    
    if not os.path.exists(file_path):
        return jsonify({'present': [], 'absent': all_students})
    try:
        df = pd.read_csv(file_path)
        # Filter by subject if a specific subject is requested
        if subject_filter != 'all':
            df = df[df['Subject'] == subject_filter]
        
        present_students = df['Name'].unique().tolist()
        absent_students = [s for s in all_students if s not in present_students]
        return jsonify({'present': df.to_dict('records'), 'absent': absent_students})
    except pd.errors.EmptyDataError:
        return jsonify({'present': [], 'absent': all_students})

### MODIFIED ###
@app.route('/api/overall_attendance', methods=['GET'])
def api_overall_attendance():
    subject_filter = request.args.get('subject', 'all')
    all_students = get_all_students()
    if not all_students: return jsonify({'report': []})

    record_files = [f for f in os.listdir(ATTENDANCE_RECORDS_PATH) if f.endswith('.csv')]
    report = []

    if subject_filter == 'all':
        # --- LOGIC FOR "ALL SUBJECTS" ---
        # Total classes is the number of days with any attendance record
        total_days = len(record_files)
        present_days_count = {student: 0 for student in all_students}
        
        for file in record_files:
            try:
                df = pd.read_csv(os.path.join(ATTENDANCE_RECORDS_PATH, file))
                # Count a student present if their name appears at all on that day
                for student in df['Name'].unique():
                    if student in present_days_count:
                        present_days_count[student] += 1
            except pd.errors.EmptyDataError:
                continue
                
        for s in all_students:
            present = present_days_count.get(s, 0)
            report.append({'student': s, 'present_count': present, 'total_classes': total_days, 'percentage': (present / total_days * 100) if total_days > 0 else 0})
    
    else:
        # --- LOGIC FOR A SPECIFIC SUBJECT ---
        total_subject_classes = 0
        present_subject_count = {student: 0 for student in all_students}
        
        for file in record_files:
            try:
                df = pd.read_csv(os.path.join(ATTENDANCE_RECORDS_PATH, file))
                # Check if the selected subject was taught on this day
                if subject_filter in df['Subject'].unique():
                    total_subject_classes += 1
                    # Get students present for that specific subject
                    present_for_subject = df[df['Subject'] == subject_filter]['Name'].unique()
                    for student in present_for_subject:
                        if student in present_subject_count:
                            present_subject_count[student] += 1
            except (pd.errors.EmptyDataError, KeyError):
                continue
                
        for s in all_students:
            present = present_subject_count.get(s, 0)
            report.append({'student': s, 'present_count': present, 'total_classes': total_subject_classes, 'percentage': (present / total_subject_classes * 100) if total_subject_classes > 0 else 0})
            
    return jsonify({'report': report})


@app.route('/api/timetable', methods=['GET'])
def api_get_timetable():
    if not os.path.exists(TIMETABLE_FILE): return jsonify({'timetable': {}})
    with open(TIMETABLE_FILE, 'r') as f:
        try: return jsonify({'timetable': json.load(f)})
        except json.JSONDecodeError: return jsonify({'timetable': {}})

@app.route('/api/current_subject', methods=['GET'])
def api_get_current_subject(): return jsonify({'subject': get_current_subject()})

### NEW ###
@app.route('/api/subjects', methods=['GET'])
def api_get_subjects():
    subjects = set()
    if os.path.exists(TIMETABLE_FILE):
        with open(TIMETABLE_FILE, 'r') as f:
            try:
                timetable = json.load(f)
                for day in timetable.values():
                    for slot in day:
                        subjects.add(slot['subject'])
            except json.JSONDecodeError:
                pass # Return empty set if file is corrupt
    return jsonify({'subjects': sorted(list(subjects))})


@app.route('/api/save_slot', methods=['POST'])
def api_save_slot():
    day, subject, start, end = request.form.get('slot-day'), request.form.get('slot-subject'), request.form.get('slot-start'), request.form.get('slot-end')
    index = int(request.form.get('slot-index', -1))
    if not all([day, subject, start, end]): return jsonify({'success': False, 'message': 'All fields are required.'})
    
    timetable = {}
    if os.path.exists(TIMETABLE_FILE):
        with open(TIMETABLE_FILE, 'r') as f:
            try: timetable = json.load(f)
            except json.JSONDecodeError: timetable = {}
    
    if day not in timetable: timetable[day] = []
    new_slot = {'subject': subject, 'start': start, 'end': end}

    if index > -1 and index < len(timetable[day]):
        timetable[day][index] = new_slot
    else:
        timetable[day].append(new_slot)
    
    timetable[day].sort(key=lambda x: x['start']) # Sort slots by start time
    with open(TIMETABLE_FILE, 'w') as f: json.dump(timetable, f, indent=4)
    return jsonify({'success': True})

@app.route('/api/delete_slot', methods=['POST'])
def api_delete_slot():
    day, index = request.form.get('day'), int(request.form.get('index'))
    if not os.path.exists(TIMETABLE_FILE): return jsonify({'success': False, 'message': 'Timetable not found.'})
    with open(TIMETABLE_FILE, 'r') as f: timetable = json.load(f)
    if day in timetable and 0 <= index < len(timetable[day]):
        del timetable[day][index]
        with open(TIMETABLE_FILE, 'w') as f: json.dump(timetable, f, indent=4)
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': 'Slot not found.'})

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

def _send_email_logic(subject, content_generator):
    if not os.path.exists(SENDER_GMAIL_FILE) or not os.path.exists(STUDENT_EMAILS_FILE): return jsonify({'success': False, 'message': 'Sender or student emails not configured.'})
    with open(SENDER_GMAIL_FILE, 'r') as f: sender_creds = json.load(f)
    with open(STUDENT_EMAILS_FILE, 'r') as f: student_emails = json.load(f)
    SENDER_EMAIL, SENDER_PASSWORD = sender_creds.get('email'), sender_creds.get('password')
    if not SENDER_EMAIL or not SENDER_PASSWORD: return jsonify({'success': False, 'message': 'Sender credentials are incomplete.'})
    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        sent_count = 0
        for student_name, recipient_email in student_emails.items():
            if not recipient_email: continue
            body = content_generator(student_name)
            if not body: continue
            msg = EmailMessage()
            msg['Subject'], msg['From'], msg['To'] = subject, SENDER_EMAIL, recipient_email
            msg.set_content(body)
            server.send_message(msg)
            sent_count += 1
        server.quit()
        return jsonify({'success': True, 'message': f'Successfully sent {sent_count} emails.'})
    except smtplib.SMTPAuthenticationError: return jsonify({'success': False, 'message': 'Gmail login failed. Check email/App Password.'})
    except Exception as e: return jsonify({'success': False, 'message': f'An error occurred: {e}'})

@app.route('/api/send_todays_email', methods=['POST'])
def api_send_todays_email():
    date_str = datetime.now().strftime("%Y-%m-%d")
    file_path = os.path.join(ATTENDANCE_RECORDS_PATH, f"attendance-{date_str}.csv")
    present_df = pd.DataFrame()
    if os.path.exists(file_path):
        try: present_df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError: pass
    def content_generator(name):
        student_records = present_df[present_df['Name'] == name]
        if not student_records.empty:
            subjects = ", ".join(student_records['Subject'].tolist())
            return f"Hi {name},\n\nYou were marked PRESENT for: {subjects} on {date_str}.\n\nThank you."
        else:
            return f"Hi {name},\n\nYou were marked ABSENT for all classes on {date_str}.\n\nThank you."
    return _send_email_logic(f"Attendance Report: {date_str}", content_generator)

@app.route('/api/send_overall_email', methods=['POST'])
def api_send_overall_email():
    report_data = api_overall_attendance().get_json().get('report', [])
    if not report_data: return jsonify({'success': False, 'message': 'No attendance data to report.'})
    
    def content_generator(name):
        student_data = next((item for item in report_data if item['student'] == name), None)
        if not student_data: return None
        return (f"Hi {name},\n\nHere is your overall attendance summary:\n"
                f"Classes Present: {student_data['present_count']}\n"
                f"Total Classes Held: {student_data['total_classes']}\n"
                f"Percentage: {student_data['percentage']:.1f}%\n\nThank you.")
            
    return _send_email_logic("Overall Attendance Summary", content_generator)

# --- Main Entry Point ---
if __name__ == '__main__':
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(ATTENDANCE_RECORDS_PATH, exist_ok=True)
    print("Starting Flask server...")
    print(f"Admin login at: http://127.0.0.1:5000/admin (Password: {ADMIN_PASSWORD})")
    # This line should be removed or commented out for deployment
    app.run(host='0.0.0.0', port=5000, debug=False)