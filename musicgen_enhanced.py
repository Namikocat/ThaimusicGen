import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
import torch
from audiocraft.models import MusicGen
import torchaudio
import pygame
from datetime import datetime
import gc

class MusicGenGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MusicGen - AI Music Generator")
        self.root.geometry("900x700")
        self.root.configure(bg='#2c3e50')
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
        
        # Variables
        self.base_model = None
        self.finetuned_model = None
        self.current_model = None
        self.is_generating = False
        self.is_loading = False
        self.current_audio_file = None
        self.model_type = "base"  # "base" or "finetuned"
        
        self.setup_ui()
        # Load base model on startup
        self.load_base_model_startup()
    
    def setup_ui(self):
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50')
        title_frame.pack(pady=15)
        
        title_label = tk.Label(
            title_frame, 
            text="🎵 MusicGen AI Music Generator 🎵",
            font=("Arial", 24, "bold"),
            fg='#ecf0f1',
            bg='#2c3e50'
        )
        title_label.pack()
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#34495e', relief='raised', bd=2)
        main_frame.pack(padx=20, pady=10, fill='both', expand=True)
        
        # Model selection section
        model_frame = tk.Frame(main_frame, bg='#34495e')
        model_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(
            model_frame,
            text="Model Selection:",
            font=("Arial", 14, "bold"),
            fg='#ecf0f1',
            bg='#34495e'
        ).pack(anchor='w')
        
        # Model type selection
        model_type_frame = tk.Frame(model_frame, bg='#34495e')
        model_type_frame.pack(fill='x', pady=10)
        
        self.model_type_var = tk.StringVar(value="base")
        
        base_radio = tk.Radiobutton(
            model_type_frame,
            text="🎼 Base Model (musicgen-medium)",
            variable=self.model_type_var,
            value="base",
            command=self.on_model_type_change,
            font=("Arial", 11, "bold"),
            fg='#ecf0f1',
            bg='#34495e',
            selectcolor='#2c3e50',
            activebackground='#34495e',
            activeforeground='#ecf0f1'
        )
        base_radio.pack(anchor='w')
        
        thai_radio = tk.Radiobutton(
            model_type_frame,
            text="🇹🇭 Thai Music Model (finetuned)",
            variable=self.model_type_var,
            value="finetuned",
            command=self.on_model_type_change,
            font=("Arial", 11, "bold"),
            fg='#ecf0f1',
            bg='#34495e',
            selectcolor='#2c3e50',
            activebackground='#34495e',
            activeforeground='#ecf0f1'
        )
        thai_radio.pack(anchor='w', pady=(5, 0))
        
        # Finetuned model file selection (initially hidden)
        self.finetuned_frame = tk.Frame(model_frame, bg='#34495e')
        
        tk.Label(
            self.finetuned_frame,
            text="Thai Music Model File:",
            font=("Arial", 11, "bold"),
            fg='#ecf0f1',
            bg='#34495e'
        ).pack(anchor='w')
        
        model_path_frame = tk.Frame(self.finetuned_frame, bg='#34495e')
        model_path_frame.pack(fill='x', pady=5)
        
        self.model_path_var = tk.StringVar(value="Please select your Thai music model file...")
        self.model_path_entry = tk.Entry(
            model_path_frame,
            textvariable=self.model_path_var,
            font=("Arial", 10),
            width=60,
            state='readonly',
            fg='#7f8c8d'
        )
        self.model_path_entry.pack(side='left', fill='x', expand=True)
        
        browse_btn = tk.Button(
            model_path_frame,
            text="📁 Browse",
            command=self.browse_model_file,
            bg='#3498db',
            fg='white',
            font=("Arial", 10, "bold"),
            relief='flat'
        )
        browse_btn.pack(side='right', padx=(5, 0))
        
        # Model control buttons
        model_control_frame = tk.Frame(model_frame, bg='#34495e')
        model_control_frame.pack(fill='x', pady=10)
        
        self.load_model_btn = tk.Button(
            model_control_frame,
            text="🔄 Load Selected Model",
            command=self.load_selected_model,
            bg='#e67e22',
            fg='white',
            font=("Arial", 11, "bold"),
            relief='flat'
        )
        self.load_model_btn.pack(side='left', padx=(0, 10))
        
        self.unload_model_btn = tk.Button(
            model_control_frame,
            text="🗑️ Unload Model",
            command=self.unload_current_model,
            bg='#e74c3c',
            fg='white',
            font=("Arial", 11, "bold"),
            relief='flat',
            state='disabled'
        )
        self.unload_model_btn.pack(side='left')
        
        # Model status
        self.model_status_label = tk.Label(
            model_frame,
            text="🔄 Loading base model...",
            font=("Arial", 11, "bold"),
            fg='#f39c12',
            bg='#34495e'
        )
        self.model_status_label.pack(anchor='w', pady=(10, 0))
        
        # Description input section
        desc_frame = tk.Frame(main_frame, bg='#34495e')
        desc_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        tk.Label(
            desc_frame,
            text="Music Description:",
            font=("Arial", 12, "bold"),
            fg='#ecf0f1',
            bg='#34495e'
        ).pack(anchor='w')
        
        # Preset descriptions for Thai music
        preset_frame = tk.Frame(desc_frame, bg='#34495e')
        preset_frame.pack(fill='x', pady=5)
        
        tk.Label(
            preset_frame,
            text="Thai Music Presets:",
            font=("Arial", 10, "bold"),
            fg='#ecf0f1',
            bg='#34495e'
        ).pack(side='left')
        
        self.preset_var = tk.StringVar()
        preset_combo = ttk.Combobox(
            preset_frame,
            textvariable=self.preset_var,
            values=[
                "Thai song with kim",
            ],
            width=30,
            state="readonly"
        )
        preset_combo.pack(side='left', padx=(10, 0))
        
        use_preset_btn = tk.Button(
            preset_frame,
            text="Use",
            command=self.use_preset,
            bg='#27ae60',
            fg='white',
            font=("Arial", 9, "bold"),
            relief='flat'
        )
        use_preset_btn.pack(side='left', padx=(5, 0))
        
        self.description_text = scrolledtext.ScrolledText(
            desc_frame,
            height=5,
            font=("Arial", 11),
            wrap=tk.WORD
        )
        self.description_text.pack(fill='both', expand=True, pady=(5, 0))
        self.description_text.insert('1.0', "upbeat electronic music")
        
        # Settings frame
        settings_frame = tk.Frame(main_frame, bg='#34495e')
        settings_frame.pack(fill='x', padx=20, pady=10)
        
        # Duration setting
        duration_frame = tk.Frame(settings_frame, bg='#34495e')
        duration_frame.pack(fill='x')
        
        tk.Label(
            duration_frame,
            text="Duration (seconds):",
            font=("Arial", 11, "bold"),
            fg='#ecf0f1',
            bg='#34495e'
        ).pack(side='left')
        
        self.duration_var = tk.StringVar(value="10")
        duration_spinbox = tk.Spinbox(
            duration_frame,
            from_=5,
            to=30,
            textvariable=self.duration_var,
            width=10,
            font=("Arial", 10)
        )
        duration_spinbox.pack(side='left', padx=(10, 0))
        
        # Sample rate setting
        tk.Label(
            duration_frame,
            text="Sample Rate:",
            font=("Arial", 11, "bold"),
            fg='#ecf0f1',
            bg='#34495e'
        ).pack(side='left', padx=(30, 0))
        
        self.sample_rate_var = tk.StringVar(value="32000")
        sample_rate_combo = ttk.Combobox(
            duration_frame,
            textvariable=self.sample_rate_var,
            values=["16000", "22050", "32000", "44100"],
            width=8,
            state="readonly"
        )
        sample_rate_combo.pack(side='left', padx=(10, 0))
        
        # Buttons frame
        buttons_frame = tk.Frame(main_frame, bg='#34495e')
        buttons_frame.pack(fill='x', padx=20, pady=10)
        
        self.generate_btn = tk.Button(
            buttons_frame,
            text="🎵 Generate Music",
            command=self.generate_music,
            bg='#27ae60',
            fg='white',
            font=("Arial", 12, "bold"),
            height=2,
            relief='flat',
            state='disabled'
        )
        self.generate_btn.pack(side='left', padx=(0, 10))
        
        self.play_btn = tk.Button(
            buttons_frame,
            text="▶️ Play",
            command=self.play_audio,
            bg='#e74c3c',
            fg='white',
            font=("Arial", 12, "bold"),
            height=2,
            relief='flat',
            state='disabled'
        )
        self.play_btn.pack(side='left', padx=(0, 10))
        
        self.stop_btn = tk.Button(
            buttons_frame,
            text="⏹️ Stop",
            command=self.stop_audio,
            bg='#f39c12',
            fg='white',
            font=("Arial", 12, "bold"),
            height=2,
            relief='flat',
            state='disabled'
        )
        self.stop_btn.pack(side='left', padx=(0, 10))
        
        save_btn = tk.Button(
            buttons_frame,
            text="💾 Save As",
            command=self.save_audio,
            bg='#9b59b6',
            fg='white',
            font=("Arial", 12, "bold"),
            height=2,
            relief='flat'
        )
        save_btn.pack(side='right')
        
        # Progress bar
        self.progress = ttk.Progressbar(
            main_frame,
            mode='indeterminate',
            length=500
        )
        self.progress.pack(pady=10)
        
        # Status text
        self.status_text = scrolledtext.ScrolledText(
            main_frame,
            height=8,
            font=("Consolas", 9),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        self.status_text.pack(fill='x', padx=20, pady=10)
        
        self.log_message("🎵 Welcome to MusicGen AI Music Generator!")
        self.log_message("🔄 Loading base model, please wait...")
    
    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()
    
    def on_model_type_change(self):
        if self.model_type_var.get() == "finetuned":
            self.finetuned_frame.pack(fill='x', pady=10)
            if self.preset_var.get() == "":
                self.preset_var.set("Thai song with kim")
                self.use_preset()
        else:
            self.finetuned_frame.pack_forget()
            if "Thai" in self.description_text.get('1.0', tk.END):
                self.description_text.delete('1.0', tk.END)
                self.description_text.insert('1.0', "upbeat electronic music")
    
    def use_preset(self):
        preset = self.preset_var.get()
        if preset:
            self.description_text.delete('1.0', tk.END)
            self.description_text.insert('1.0', preset)
    
    def load_base_model_startup(self):
        def load_in_thread():
            try:
                self.log_message("📥 Loading MusicGen medium model...")
                self.base_model = MusicGen.get_pretrained("facebook/musicgen-medium")
                self.current_model = self.base_model
                self.model_type = "base"
                
                self.log_message("✅ Base model loaded successfully!")
                self.model_status_label.config(text="✅ Base Model Ready", fg='#27ae60')
                self.generate_btn.config(state='normal')
                self.unload_model_btn.config(state='normal')
                
            except Exception as e:
                self.log_message(f"❌ Error loading base model: {str(e)}")
                self.model_status_label.config(text="❌ Base Model Failed", fg='#e74c3c')
        
        threading.Thread(target=load_in_thread, daemon=True).start()
    
    def browse_model_file(self):
        filename = filedialog.askopenfilename(
            title="Select Your Thai Music Model",
            filetypes=[("PyTorch Model files", "*.pt"), ("PyTorch files", "*.pth"), ("All files", "*.*")],
            initialdir=os.getcwd()
        )
        if filename:
            self.model_path_var.set(filename)
            self.model_path_entry.config(fg='#2c3e50')
            self.log_message(f"📁 Thai model file selected: {os.path.basename(filename)}")
    
    def load_selected_model(self):
        if self.is_loading:
            return
        
        model_type = self.model_type_var.get()
        
        if model_type == "base":
            if self.base_model is not None:
                self.current_model = self.base_model
                self.model_type = "base"
                self.model_status_label.config(text="✅ Base Model Active", fg='#27ae60')
                self.generate_btn.config(state='normal')
                self.unload_model_btn.config(state='normal')
                self.log_message("✅ Switched to base model")
                return
            else:
                self.load_base_model_startup()
                return
        
        # For finetuned model
        model_path = self.model_path_var.get()
        if not model_path or model_path == "Please select your Thai music model file...":
            messagebox.showwarning("Warning", "Please select a Thai music model file first!")
            return
        
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model file not found:\n{model_path}")
            return
        
        self.is_loading = True
        self.load_model_btn.config(state='disabled', text='Loading...')
        self.generate_btn.config(state='disabled')
        self.progress.start()
        self.model_status_label.config(text="🔄 Loading Thai Model...", fg='#f39c12')
        
        def load_in_thread():
            try:
                if self.base_model is None:
                    self.log_message("📥 Loading base model first...")
                    self.base_model = MusicGen.get_pretrained("facebook/musicgen-medium")
                
                self.log_message("🇹🇭 Loading Thai music model...")
                self.log_message(f"📂 File: {os.path.basename(model_path)}")
                
                # Create finetuned model from base
                self.finetuned_model = MusicGen.get_pretrained("facebook/musicgen-medium")
                
                # Load the finetuned weights
                checkpoint = torch.load(model_path, map_location='cpu')
                self.finetuned_model.lm.load_state_dict(checkpoint)
                self.finetuned_model.lm.eval()
                
                # Set as current model
                self.current_model = self.finetuned_model
                self.model_type = "finetuned"
                
                self.log_message("✅ Thai music model loaded successfully!")
                self.model_status_label.config(text="✅ Thai Model Active", fg='#27ae60')
                self.generate_btn.config(state='normal')
                self.unload_model_btn.config(state='normal')
                self.load_model_btn.config(text='✅ Model Loaded', bg='#27ae60')
                
            except Exception as e:
                self.log_message(f"❌ Error loading Thai model: {str(e)}")
                self.model_status_label.config(text="❌ Thai Model Failed", fg='#e74c3c')
                messagebox.showerror("Model Loading Error", f"Failed to load Thai model:\n{str(e)}")
                self.load_model_btn.config(state='normal', text='🔄 Load Selected Model', bg='#e67e22')
            
            finally:
                self.is_loading = False
                self.progress.stop()
        
        threading.Thread(target=load_in_thread, daemon=True).start()
    
    def unload_current_model(self):
        try:
            self.log_message("🗑️ Unloading current model...")
            
            # Stop any ongoing generation
            self.is_generating = False
            
            # Clear current model
            if self.current_model is not None:
                del self.current_model
                self.current_model = None
            
            # Clear finetuned model if exists
            if self.finetuned_model is not None:
                del self.finetuned_model
                self.finetuned_model = None
            
            # Clear base model
            if self.base_model is not None:
                del self.base_model
                self.base_model = None
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Update UI
            self.model_status_label.config(text="🚫 No Model Loaded", fg='#7f8c8d')
            self.generate_btn.config(state='disabled')
            self.unload_model_btn.config(state='disabled')
            self.load_model_btn.config(state='normal', text='🔄 Load Selected Model', bg='#e67e22')
            self.model_type = None
            
            self.log_message("✅ Model unloaded successfully!")
            self.log_message("💾 Memory freed")
            
        except Exception as e:
            self.log_message(f"❌ Error unloading model: {str(e)}")
    
    def generate_music(self):
        if self.is_generating:
            return
        
        description = self.description_text.get('1.0', tk.END).strip()
        if not description:
            messagebox.showwarning("Warning", "Please enter a music description!")
            return
        
        if self.current_model is None:
            messagebox.showerror("Error", "No model loaded! Please load a model first.")
            return
        
        self.is_generating = True
        self.generate_btn.config(state='disabled')
        self.progress.start()
        
        def generate_in_thread():
            try:
                duration = int(self.duration_var.get())
                sample_rate = int(self.sample_rate_var.get())
                
                model_name = "Thai Music Model" if self.model_type == "finetuned" else "Base Model"
                self.log_message(f"🎵 Generating with {model_name}: '{description}'")
                self.log_message(f"Duration: {duration}s, Sample Rate: {sample_rate}Hz")
                
                # Set generation parameters
                self.current_model.set_generation_params(duration=duration)
                
                # Generate music
                descriptions = [description]
                waveforms = self.current_model.generate(descriptions)
                
                # Save generated audio
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_suffix = "_thai" if self.model_type == "finetuned" else "_base"
                filename = f"generated_music{model_suffix}_{timestamp}.wav"
                torchaudio.save(filename, waveforms[0].cpu(), sample_rate=sample_rate)
                
                self.current_audio_file = filename
                self.log_message(f"✅ Music generated and saved as: {filename}")
                
                # Enable play button
                self.play_btn.config(state='normal')
                
            except Exception as e:
                self.log_message(f"❌ Error generating music: {str(e)}")
                messagebox.showerror("Generation Error", str(e))
            
            finally:
                self.is_generating = False
                self.generate_btn.config(state='normal')
                self.progress.stop()
        
        threading.Thread(target=generate_in_thread, daemon=True).start()
    
    def play_audio(self):
        if self.current_audio_file and os.path.exists(self.current_audio_file):
            try:
                pygame.mixer.music.load(self.current_audio_file)
                pygame.mixer.music.play()
                self.log_message(f"▶️ Playing: {self.current_audio_file}")
                self.stop_btn.config(state='normal')
            except Exception as e:
                self.log_message(f"❌ Error playing audio: {str(e)}")
        else:
            messagebox.showwarning("Warning", "No audio file to play!")
    
    def stop_audio(self):
        pygame.mixer.music.stop()
        self.log_message("⏹️ Audio stopped")
        self.stop_btn.config(state='disabled')
    
    def save_audio(self):
        if self.current_audio_file and os.path.exists(self.current_audio_file):
            filename = filedialog.asksaveasfilename(
                defaultextension=".wav",
                filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
                title="Save Generated Music"
            )
            if filename:
                try:
                    import shutil
                    shutil.copy2(self.current_audio_file, filename)
                    self.log_message(f"💾 Audio saved as: {filename}")
                    messagebox.showinfo("Success", f"Audio saved successfully as:\n{filename}")
                except Exception as e:
                    self.log_message(f"❌ Error saving file: {str(e)}")
                    messagebox.showerror("Save Error", str(e))
        else:
            messagebox.showwarning("Warning", "No audio file to save!")

def main():
    root = tk.Tk()
    app = MusicGenGUI(root)
    
    # Handle window closing
    def on_closing():
        try:
            pygame.mixer.quit()
            # Clean up models
            if app.current_model is not None:
                del app.current_model
            if app.base_model is not None:
                del app.base_model
            if app.finetuned_model is not None:
                del app.finetuned_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()