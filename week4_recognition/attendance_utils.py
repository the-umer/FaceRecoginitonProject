import csv
import os
from datetime import datetime

ATTENDANCE_FILE = "attendance/attendance.csv"


def mark_attendance(name):
    # Create file if not exists
    file_exists = os.path.isfile(ATTENDANCE_FILE)

    # Get today's date
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    # Read existing records to prevent duplicates
    already_marked = False
    if file_exists:
        with open(ATTENDANCE_FILE, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2 and row[0] == name and row[1] == today:
                    already_marked = True
                    break

    # If already marked today → skip
    if already_marked:
        print(f"✔️ {name} already marked today")
        return

    # Append new attendance entry
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        
        # Write header if file didn't exist
        if not file_exists:
            writer.writerow(["Name", "Date", "Time"])
        
        writer.writerow([name, today, time_now])

    print(f"✅ Attendance marked for {name} at {time_now}")
