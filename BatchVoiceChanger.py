import tkinter as tk
from tkinter import ttk, filedialog
import os
import sys
import torch
import librosa
import soundfile as sf
from scipy.io.wavfile import write
import FreeVC.utils as utils
from FreeVC.models import SynthesizerTrn
from FreeVC.mel_processing import mel_spectrogram_torch
from FreeVC.speaker_encoder.voice_encoder import SpeakerEncoder

class BatchVoiceChangerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Batch Voice Changer")
        self.root.geometry("600x400")
        
        # Initialize model variables
        self.initialize_models()
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Voice selection
        ttk.Label(self.main_frame, text="Voice:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.voice_var = tk.StringVar(value="fem 1")
        voices = ["fem " + str(i) for i in range(1, 11)] + ["male " + str(i) for i in range(1, 11)]
        voice_combo = ttk.Combobox(self.main_frame, textvariable=self.voice_var, values=voices)
        voice_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Quality/Latency selection
        ttk.Label(self.main_frame, text="Quality (Latency):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.quality_var = tk.StringVar(value="High (1.5s)")
        qualities = ["High (1.5s)", "Medium (1.0s)", "Low (0.5s)"]
        quality_combo = ttk.Combobox(self.main_frame, textvariable=self.quality_var, values=qualities)
        quality_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Input folder selection
        ttk.Label(self.main_frame, text="Input Folder:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.input_folder = tk.StringVar()
        ttk.Entry(self.main_frame, textvariable=self.input_folder).grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(self.main_frame, text="Browse", command=self.select_input_folder).grid(row=2, column=2, pady=5)
        
        # Progress
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self.main_frame, variable=self.progress_var, maximum=100)
        self.progress.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.main_frame, textvariable=self.status_var).grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # Convert button
        self.convert_btn = ttk.Button(self.main_frame, text="Convert", command=self.start_conversion)
        self.convert_btn.grid(row=5, column=0, columnspan=3, pady=10)
        
        # Configure grid weights
        self.main_frame.columnconfigure(1, weight=1)
        
    def initialize_models(self):
        """Initialize the voice conversion models"""
        self.status_var.set("Loading models...")
        
        # Load configs
        self.hps = utils.get_hparams_from_file("FreeVC/configs/freevc.json")
        
        # Initialize models
        self.net_g = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model).cuda()
        _ = self.net_g.eval()
        
        # Load checkpoint
        _ = utils.load_checkpoint("FreeVC/checkpoints/freevc.pth", self.net_g, None)
        
        # Load WavLM
        self.cmodel = utils.get_cmodel(0, "FreeVC/wavlm/WavLM-Large.pt")
        
        # Load speaker encoder
        if self.hps.model.use_spk:
            self.smodel = SpeakerEncoder("FreeVC/speaker_encoder/ckpt/pretrained_bak_5805000.pt")
        
        self.status_var.set("Models loaded")
        
    def select_input_folder(self):
        """Open folder selection dialog"""
        folder = filedialog.askdirectory()
        if folder:
            self.input_folder.set(folder)
            
    def convert_file(self, source_path, output_path, voice_path):
        """Convert a single audio file"""
        with torch.no_grad():
            # Load source audio
            wav_src, _ = librosa.load(source_path, sr=self.hps.data.sampling_rate)
            wav_src = torch.from_numpy(wav_src).unsqueeze(0).cuda()
            c = utils.get_content(self.cmodel, wav_src)
            
            # Load target voice
            wav_tgt, _ = librosa.load(voice_path, sr=self.hps.data.sampling_rate)
            wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
            
            if self.hps.model.use_spk:
                g_tgt = self.smodel.embed_utterance(wav_tgt)
                g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()
                audio = self.net_g.infer(c, g=g_tgt)
            else:
                wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
                mel_tgt = mel_spectrogram_torch(
                    wav_tgt,
                    self.hps.data.filter_length,
                    self.hps.data.n_mel_channels,
                    self.hps.data.sampling_rate,
                    self.hps.data.hop_length,
                    self.hps.data.win_length,
                    self.hps.data.mel_fmin,
                    self.hps.data.mel_fmax
                )
                audio = self.net_g.infer(c, mel=mel_tgt)
                
            audio = audio[0][0].data.cpu().float().numpy()
            write(output_path, self.hps.data.sampling_rate, audio)
            
    def start_conversion(self):
        """Start the batch conversion process"""
        input_dir = self.input_folder.get()
        if not input_dir:
            self.status_var.set("Please select an input folder")
            return
            
        # Create output folder
        output_dir = os.path.join(input_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get selected voice
        voice_category, voice_num = self.voice_var.get().split()
        voice_path = f"voices/{voice_category}/{voice_num}.wav"
        
        # Get quality setting
        quality = self.quality_var.get()
        if quality == "High (1.5s)":
            blocksize_seconds = 1.5
        elif quality == "Medium (1.0s)":
            blocksize_seconds = 1.0
        else:
            blocksize_seconds = 0.5
            
        # Get list of wav files
        wav_files = [f for f in os.listdir(input_dir) 
                    if f.lower().endswith('.wav') and os.path.isfile(os.path.join(input_dir, f))]
        
        if not wav_files:
            self.status_var.set("No WAV files found in input folder")
            return
            
        # Disable controls during conversion
        self.convert_btn.state(['disabled'])
        
        # Process each file
        for i, wav_file in enumerate(wav_files):
            self.status_var.set(f"Converting {wav_file}...")
            
            input_path = os.path.join(input_dir, wav_file)
            output_path = os.path.join(output_dir, wav_file)
            
            try:
                self.convert_file(input_path, output_path, voice_path)
                progress = ((i + 1) / len(wav_files)) * 100
                self.progress_var.set(progress)
                self.root.update()
            except Exception as e:
                self.status_var.set(f"Error converting {wav_file}: {str(e)}")
                continue
                
        self.status_var.set("Conversion complete!")
        self.convert_btn.state(['!disabled'])
        
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = BatchVoiceChangerGUI()
    app.run()