import tkinter as tk
from tkinter import messagebox
import subprocess
import os

# ------------------------------
# Helper to run Python scripts
# ------------------------------
def run_script(path):
    try:
        subprocess.run(["python3", path])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run script:\n{path}\n\n{e}")

# ------------------------------
# Button Functions
# ------------------------------
def register_user():
    run_script("week4_recognition/register_user.py")

def train_model():
    run_script("week4_recognition/train_knn.py")

def start_recognition():
    run_script("week4_recognition/real_time_recognition.py")

def view_attendance():
    file = "attendance/attendance.csv"
    if os.path.exists(file):
        os.system(f"open {file}")
    else:
        messagebox.showerror("Error", "attendance.csv not found!")

# ------------------------------
# GUI Window
# ------------------------------
root = tk.Tk()
root.title("Face Recognition Attendance System")
root.geometry("450x400")
root.configure(bg="#1e1e1e")

title_label = tk.Label(root, text="Face Recognition Attendance System",
                       bg="#1e1e1e", fg="white", font=("Arial", 16, "bold"))
title_label.pack(pady=20)

button_style = {
    "width": 20,
    "height": 2,
    "background": "#4CAF50",
    "fg": "white",
    "font": ("Arial", 12, "bold")
}

tk.Button(root, text="Register User", command=register_user, **button_style).pack(pady=10)
tk.Button(root, text="Train Model", command=train_model, **button_style).pack(pady=10)
tk.Button(root, text="Start Recognition", command=start_recognition, **button_style).pack(pady=10)
tk.Button(root, text="View Attendance", command=view_attendance, **button_style).pack(pady=10)

root.mainloop()
