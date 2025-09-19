#!/usr/bin/env python3
"""
Niezawodny wzmacniacz wideo z histogram stretching
Eliminuje ostre strefy i zachowuje płynne przejścia
"""

import cv2
import numpy as np
import sys
import os
import subprocess
import tempfile
from pathlib import Path

class VideoEnhancer:
    """Niezawodny wzmacniacz wideo"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix='video_enhance_')
        print(f"📁 Katalog tymczasowy: {self.temp_dir}")
    
    def enhance_frame(self, frame):
        """Niezawodne wzmocnienie klatki metodą histogram stretching"""
        if frame is None:
            return None
        
        # Metoda bazująca na histogram stretching
        frame_float = frame.astype(np.float32)
        
        # Dla każdego kanału osobno
        enhanced = np.zeros_like(frame_float)
        for c in range(3):
            channel = frame_float[:,:,c]
            
            # Percentyle dla miękkiego clippingu
            p2, p98 = np.percentile(channel, (2, 98))
            
            # Soft contrast stretching
            if p98 > p2:
                stretched = (channel - p2) / (p98 - p2)
                stretched = np.clip(stretched, 0, 1)
            else:
                stretched = channel / 255.0
            
            enhanced[:,:,c] = stretched * 255.0
        
        # Łagodne wyostrzanie
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 0.8)
        enhanced = enhanced + 0.2 * (enhanced - blurred)
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def diagnose_opencv(self):
        """Diagnostyka OpenCV"""
        print("🔍 Diagnostyka OpenCV:")
        print(f"   Wersja: {cv2.__version__}")
        
        codecs = ['mp4v', 'XVID', 'MJPG', 'H264', 'X264']
        available_codecs = []
        
        for codec in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                test_writer = cv2.VideoWriter('test.avi', fourcc, 25, (100, 100))
                if test_writer.isOpened():
                    available_codecs.append(codec)
                    print(f"     ✓ {codec}")
                else:
                    print(f"     ✗ {codec}")
                test_writer.release()
                try:
                    os.remove('test.avi')
                except:
                    pass
            except Exception as e:
                print(f"     ✗ {codec} - błąd: {e}")
        
        return available_codecs
    
    def check_ffmpeg(self):
        """Sprawdź dostępność FFmpeg"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                print(f"✓ FFmpeg dostępny: {version_line}")
                return True
        except:
            pass
        
        print("✗ FFmpeg niedostępny")
        return False
    
    def process_video(self, input_path, output_path):
        """Przetwarzanie wideo"""
        print(f"🎬 Rozpoczynam przetwarzanie: {input_path}")
        
        # Diagnostyka
        available_codecs = self.diagnose_opencv()
        has_ffmpeg = self.check_ffmpeg()
        
        if not available_codecs and not has_ffmpeg:
            print("✗ Brak dostępnych metod zapisu wideo!")
            return False
        
        # Informacje o wideo
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("✗ Nie można otworzyć pliku wejściowego")
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Wideo: {width}x{height} @ {fps:.2f} FPS, {frame_count} klatek")
        print(f"🎨 Metoda: histogram stretching")
        cap.release()
        
        # Strategia zapisu - klatki + FFmpeg
        if has_ffmpeg:
            return self.process_via_frames_ffmpeg(input_path, output_path)
        
        return False
    
    def process_via_frames_ffmpeg(self, input_path, output_path):
        """Przetwarzanie przez klatki i FFmpeg"""
        print("🎯 Strategia: Klatki -> FFmpeg")
        
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames_dir = os.path.join(self.temp_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        frame_count = 0
        success_count = 0
        
        print("📸 Przetwarzanie klatek...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Wzmocnienie histogram stretching
                enhanced = self.enhance_frame(frame)
                
                if enhanced is not None:
                    # Zapis jako PNG dla lepszej jakości
                    frame_path = os.path.join(frames_dir, f'frame_{frame_count:06d}.png')
                    success = cv2.imwrite(frame_path, enhanced, 
                                        [cv2.IMWRITE_PNG_COMPRESSION, 1])  # Minimalna kompresja
                    
                    if success:
                        success_count += 1
                
                frame_count += 1
                
                if frame_count % 30 == 0:
                    print(f"   📊 Postęp: {success_count}/{frame_count} klatek")
                    
            except Exception as e:
                print(f"⚠️ Błąd przetwarzania klatki {frame_count}: {e}")
        
        cap.release()
        print(f"✅ Przetworzono {success_count}/{frame_count} klatek")
        
        # Konwersja do wideo przez FFmpeg
        if success_count > 0:
            return self.frames_to_video_ffmpeg(frames_dir, output_path, fps)
        
        return False
    
    def frames_to_video_ffmpeg(self, frames_dir, output_path, fps):
        """Konwersja klatek do wideo przez FFmpeg z wysoką jakością"""
        print("🎬 Tworzenie wideo z klatek przez FFmpeg...")
        
        input_pattern = os.path.join(frames_dir, 'frame_%06d.png')
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', input_pattern,
            '-c:v', 'libx264',
            '-preset', 'slow',      # Lepsza kompresja
            '-crf', '15',           # Bardzo wysoka jakość (0-51, niższa = lepsza)
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',  # Szybsze ładowanie
            output_path
        ]
        
        try:
            print(f"   📝 Komenda: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print(f"✅ FFmpeg sukces: {output_path}")
                return True
            else:
                print(f"✗ FFmpeg błąd: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("✗ FFmpeg timeout")
            return False
        except Exception as e:
            print(f"✗ FFmpeg exception: {e}")
            return False
    
    def cleanup(self):
        """Wyczyść pliki tymczasowe"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Wyczyszczono: {self.temp_dir}")
        except Exception as e:
            print(f"⚠️ Błąd czyszczenia: {e}")

def main():
    if len(sys.argv) < 2:
        print("🎬 Niezawodny Wzmacniacz Wideo")
        print("Użycie: python3 video_enhancer.py <input.mp4> [output.mp4]")
        print("")
        print("Metoda: Histogram stretching")
        print("")
        print("Funkcje:")
        print("- Niezawodne histogram stretching")
        print("- Łagodne wyostrzanie bez artefaktów") 
        print("- Zachowuje naturalne przejścia")
        print("- Wysoką jakość dzięki PNG + FFmpeg")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "enhanced_" + os.path.basename(input_file)
    
    if not os.path.exists(input_file):
        print(f"✗ Plik {input_file} nie istnieje")
        return
    
    enhancer = VideoEnhancer()
    
    try:
        success = enhancer.process_video(input_file, output_file)
        if success:
            print(f"\n🎉 Sukces! Sprawdź wyniki: {output_file}")
            print("📊 Wideo zostało przetworzone metodą histogram stretching")
        else:
            print(f"\n💥 Przetwarzanie nie powiodło się")
    finally:
        enhancer.cleanup()

if __name__ == "__main__":
    main()