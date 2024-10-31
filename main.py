import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import sys

def select_input_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    input_file_entry.delete(0, tk.END)
    input_file_entry.insert(0, file_path)

def select_output_folder():
    folder_path = filedialog.askdirectory()
    output_folder_entry.delete(0, tk.END)
    output_folder_entry.insert(0, folder_path)

def run_kmeans2():
    input_file = input_file_entry.get()
    output_folder = output_folder_entry.get()
    k_value = k_entry.get()

    if not input_file or not output_folder or not k_value.isdigit():
        messagebox.showerror("Error", "Please fill all fields with valid values.")
        return

    try:
        result = subprocess.check_output(
            [sys.executable, "kmeans2.py", input_file, output_folder, k_value],
            text=True
        )
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, result)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run kmeans2.py: {e}")

def run_silhouette_test():
    input_file = input_file_entry.get()
    k_value = k_entry.get()

    if not input_file or not k_value.isdigit():
        messagebox.showerror("Error", "Please fill all fields with valid values.")
        return
    try:
        # Truyền đường dẫn file đầu vào và số lượng cụm (k) vào silhouette_test.py
        subprocess.run([sys.executable, "silhouse_test.py", input_file, k_value], check=True)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run silhouse_test.py: {e}")

# Set up the main window
window = tk.Tk()
window.title("K-means Clustering Interface")

# Input file selection
tk.Label(window, text="Select input file:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
input_file_entry = tk.Entry(window, width=40)
input_file_entry.grid(row=0, column=1, padx=5, pady=5)
tk.Button(window, text="Browse", command=select_input_file).grid(row=0, column=2, padx=5, pady=5)

# Output folder selection
tk.Label(window, text="Select output folder:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
output_folder_entry = tk.Entry(window, width=40)
output_folder_entry.grid(row=1, column=1, padx=5, pady=5)
tk.Button(window, text="Browse", command=select_output_folder).grid(row=1, column=2, padx=5, pady=5)

# K value input
tk.Label(window, text="Number of clusters (k):").grid(row=2, column=0, padx=5, pady=5, sticky="e")
k_entry = tk.Entry(window)
k_entry.grid(row=2, column=1, padx=5, pady=5)

# Run buttons
tk.Button(window, text="Run kmeans2.py", command=run_kmeans2).grid(row=3, column=1, padx=5, pady=10)
tk.Button(window, text="Run silhouette_test.py (Show Plot)", command=run_silhouette_test).grid(row=3, column=2, padx=5, pady=10)

# Result display
tk.Label(window, text="Result:").grid(row=4, column=0, padx=5, pady=5, sticky="ne")
result_text = tk.Text(window, width=60, height=10)
result_text.grid(row=4, column=1, columnspan=2, padx=5, pady=5)

window.mainloop()
