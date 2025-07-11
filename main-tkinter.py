import tkinter as tk
from tkinter import scrolledtext
import threading
import torch
from transformers import pipeline

# Inisialisasi pipeline dengan model LLM
model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="cpu",
)

# Pesan sistem tetap untuk mengarahkan chatbot
system_message = "You are a pirate chatbot who always responds in pirate speak!"

def generate_text():
    # Nonaktifkan tombol generate agar tidak terjadi panggilan berulang
    generate_button.config(state=tk.DISABLED)
    
    # Ambil prompt dari kotak teks
    prompt = prompt_entry.get("1.0", tk.END).strip()
    
    # Gabungkan pesan sistem dan prompt pengguna menjadi satu string
    full_prompt = f"You are a pirate chatbot who always responds in pirate speak!\nUser: {prompt}\nPirate:"
    
    print("Generating...")  # Debug print
    try:
        outputs = pipe(full_prompt, max_new_tokens=64)
        print("Done generating!")  # Debug print
        generated = outputs[0]["generated_text"]
        # Get only the text after "Pirate:"
        pirate_reply = generated.split("Pirate:")[-1].strip()
        # If nothing after "Pirate:", show the whole generated text
        if not pirate_reply:
            pirate_reply = generated.strip()
    except Exception as e:
        pirate_reply = f"Error: {e}"
    
    # Tampilkan hasil pada area output
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, pirate_reply)
    
    # Aktifkan kembali tombol generate
    generate_button.config(state=tk.NORMAL)

def on_generate():
    # Jalankan fungsi generate_text pada thread terpisah agar GUI tidak freeze
    thread = threading.Thread(target=generate_text)
    thread.start()

# Setup antarmuka menggunakan tkinter
root = tk.Tk()
root.title("Pirate Chatbot Generator")

# Label untuk prompt input
prompt_label = tk.Label(root, text="Enter your prompt:")
prompt_label.pack(pady=(10, 0))

# Kotak teks untuk memasukkan prompt
prompt_entry = tk.Text(root, height=5, width=60)
prompt_entry.pack(padx=10, pady=(0, 10))

# Tombol untuk memicu generasi teks
generate_button = tk.Button(root, text="Generate", command=on_generate)
generate_button.pack(pady=5)

# Label untuk hasil output
output_label = tk.Label(root, text="Generated Text:")
output_label.pack(pady=(10, 0))

# Area teks yang dapat discroll untuk menampilkan output
output_text = scrolledtext.ScrolledText(root, height=10, width=60)
output_text.pack(padx=10, pady=(0, 10))

# Mulai loop utama GUI
root.mainloop()
