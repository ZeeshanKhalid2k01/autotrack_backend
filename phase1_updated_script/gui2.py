import tkinter as tk
from tkinter import filedialog
import subprocess
from PIL import ImageTk, Image

def run_script():
    folder_path = entry_folder_path.get()
    output_folder = entry_output_folder.get()
    weights = entry_weights.get()
    confidence = entry_confidence.get()

    command = f"python process_videos2.py --folder_path {folder_path} --output_folder {output_folder} --weights {weights} --confidence {confidence}"
    subprocess.call(command, shell=True)
    print("Script completed.")

def run_results():
    command = f"python gui_code2.py"
    subprocess.call(command, shell=True)
    print("Script completed.")

def quit_program():
    root.destroy()

def adjust_confidence(event):
    current_confidence = float(entry_confidence.get())
    key = event.keysym

    if key == 'Up':
        current_confidence += 0.1
    elif key == 'Down':
        current_confidence -= 0.1

    current_confidence = max(0.1, min(1.0, current_confidence))

    entry_confidence.delete(0, tk.END)
    entry_confidence.insert(0, str(current_confidence))

def browse_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        entry_folder_path.delete(0, tk.END)
        entry_folder_path.insert(0, folder_path)

def browse_output_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        entry_output_folder.delete(0, tk.END)
        entry_output_folder.insert(0, folder_path)

root = tk.Tk()
root.title("Video Processing GUI")

frame_left = tk.Frame(root)
frame_left.pack(side=tk.LEFT, padx=20, pady=20)

# image_path = "E:/FYP/yolo/yolo checking/yolov5/pytorch_agroAI.png"
image_path = "Screenshot.png"
image = Image.open(image_path)
image = image.resize((300, 300))
image_tk = ImageTk.PhotoImage(image)
label_image = tk.Label(frame_left, image=image_tk)
label_image.pack()

frame_right = tk.Frame(root)
frame_right.pack(side=tk.LEFT, padx=20, pady=20)

label_folder_path = tk.Label(frame_right, text="Folder Path:")
label_folder_path.pack()
entry_folder_path = tk.Entry(frame_right, width=50)
entry_folder_path.pack()
entry_folder_path.insert(0, "unprocessed")  # Set default output folder value

btn_browse_folder = tk.Button(frame_right, text="Browse Folder", command=browse_folder, height=2, width=30)
btn_browse_folder.pack(pady=10)

label_output_folder = tk.Label(frame_right, text="Output Folder Path:")
label_output_folder.pack()
entry_output_folder = tk.Entry(frame_right, width=50)
entry_output_folder.pack()
entry_output_folder.insert(0, "processed")  # Set default output folder value

btn_browse_output_folder = tk.Button(frame_right, text="Browse Output Folder", command=browse_output_folder, height=2, width=30)
btn_browse_output_folder.pack(pady=10)

label_weights = tk.Label(frame_right, text="Weights:")
label_weights.pack()
entry_weights = tk.Entry(frame_right, width=50)
entry_weights.pack()
entry_weights.insert(0, r"D:\RaceProject\project\yolov8x.pt")

label_confidence = tk.Label(frame_right, text="Confidence:")
label_confidence.pack()
entry_confidence = tk.Entry(frame_right, width=50)
entry_confidence.pack()
entry_confidence.insert(0, "0.3")

btn_run = tk.Button(frame_right, text="Run Script", command=run_script, height=2, width=30)
btn_run.pack(pady=10)

btn_quit = tk.Button(frame_right, text="Quit", command=quit_program, height=2, width=30)
btn_quit.pack(pady=10)

root.bind("<Up>", adjust_confidence)
root.bind("<Down>", adjust_confidence)

root.mainloop()
