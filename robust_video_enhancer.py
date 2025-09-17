#!/usr/bin/env python3
"""
Wzmacniacz wideo z zaawansowaną diagnostyką i alternatywnymi metodami zapisu
"""

import cv2
import numpy as np
import sys
import os
import subprocess
import tempfile
from pathlib import Path

class DiagnosticVideoEnhancer:
    """Wzmacniacz z pełną diagnostyką"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix='video_enhance_')
        print(f"📁 Katalog tymczasowy: {self.temp_dir}")
    
    def diagnose_opencv(self):
        """Diagnostyka OpenCV"""
        print("🔍 Diagnostyka OpenCV:")
        print(f"   Wersja: {cv2.__version__}")
        
        # Sprawdź dostępne kodeki
        print("   Dostępne kodeki:")
        codecs = ['mp4v', 'XVID', 'MJPG', 'H264', 'X264', 'DIVX', 'IYUV', 'I420', 'YV12']
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
                # Usuń plik testowy
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
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        print("✗ FFmpeg niedostępny")
        return False
    
    def enhance_frame(self, frame):
        """Wzmocnienie klatki - uproszczona wersja"""
        if frame is None:
            return None
            
        # CLAHE dla każdego kanału w przestrzeni LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE dla luminancji
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_clahe = clahe.apply(l)
        
        # Delikatne wzmocnienie chrominancji
        clahe_chroma = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        a_enhanced = clahe_chroma.apply(a)
        b_enhanced = clahe_chroma.apply(b)
        
        # Rekombinacja
        lab_enhanced = cv2.merge([l_clahe, a_enhanced, b_enhanced])
        bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return bgr_enhanced
    
    def save_frames_as_images(self, input_path, max_frames=None):
        """Zapisz klatki jako obrazy - plan B"""
        print(f"💾 Zapisywanie klatek jako obrazy (maksymalnie {max_frames})...")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False
        
        frames_dir = os.path.join(self.temp_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        frame_count = 0
        success_count = 0
        
        while max_frames is None or frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Wzmocnienie
                enhanced = self.enhance_frame(frame)
                
                # Zapis jako JPEG
                frame_path = os.path.join(frames_dir, f'frame_{frame_count:06d}.jpg')
                success = cv2.imwrite(frame_path, enhanced)
                
                if success:
                    success_count += 1
                else:
                    print(f"⚠️  Błąd zapisu obrazu {frame_count}")
                
                frame_count += 1
                
                if frame_count % 10 == 0:
                    print(f"   Zapisano {success_count}/{frame_count} klatek...")
                    
            except Exception as e:
                print(f"⚠️  Błąd przetwarzania klatki {frame_count}: {e}")
        
        cap.release()
        print(f"✅ Zapisano {success_count} klatek w {frames_dir}")
        return success_count > 0, frames_dir
    
    def frames_to_video_ffmpeg(self, frames_dir, output_path, fps):
        """Konwersja klatek do wideo przez FFmpeg"""
        if not self.check_ffmpeg():
            return False
        
        print(f"🎬 Tworzenie wideo z klatek przez FFmpeg...")
        
        # Komenda FFmpeg
        input_pattern = os.path.join(frames_dir, 'frame_%06d.jpg')
        cmd = [
            'ffmpeg', '-y',  # -y = nadpisz plik wyjściowy
            '-framerate', str(fps),
            '-i', input_pattern,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',  # Wysoka jakość
            output_path
        ]
        
        try:
            print(f"   Komenda: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
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
    
    def try_opencv_writer_detailed(self, output_path, width, height, fps, available_codecs):
        """Szczegółowe testowanie OpenCV VideoWriter"""
        print("🔧 Testowanie zapisywania OpenCV...")
        
        # Testa klatka
        test_frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        for codec in available_codecs:
            for ext in ['.mp4', '.avi', '.mov']:
                test_path = f"test_{codec}{ext}"
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    writer = cv2.VideoWriter(test_path, fourcc, fps, (width, height))
                    
                    if writer.isOpened():
                        # Test zapisu pojedynczej klatki
                        success = writer.write(test_frame)
                        writer.release()
                        
                        if success and os.path.exists(test_path) and os.path.getsize(test_path) > 0:
                            print(f"✓ Działa: {codec} + {ext}")
                            # Zmień rozszerzenie pliku wyjściowego
                            final_path = os.path.splitext(output_path)[0] + ext
                            os.rename(test_path, final_path)
                            
                            # Stwórz właściwy writer
                            final_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                            return final_writer, output_path
                        else:
                            print(f"✗ Nie działa: {codec} + {ext} (zapis nieudany)")
                    else:
                        print(f"✗ Nie działa: {codec} + {ext} (nie można otworzyć)")
                    
                    # Cleanup
                    try:
                        os.remove(test_path)
                    except:
                        pass
                        
                except Exception as e:
                    print(f"✗ Błąd: {codec} + {ext} - {e}")
        
        return None, None
    
    def process_video_robust(self, input_path, output_path):
        """Przetwarzanie z wieloma metodami zapasowymi"""
        print(f"🎬 Rozpoczynam zaawansowane przetwarzanie: {input_path}")
        
        # Diagnostyka systemu
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
        cap.release()
        
        # Strategia 1: OpenCV z testowaniem
        if available_codecs:
            print("\n🎯 Strategia 1: OpenCV VideoWriter")
            writer, final_output = self.try_opencv_writer_detailed(
                output_path, width, height, fps, available_codecs
            )
            
            if writer is not None:
                success = self.process_with_opencv(input_path, writer)
                writer.release()
                if success:
                    return True
        
        # Strategia 2: Klatki + FFmpeg
        if has_ffmpeg:
            print("\n🎯 Strategia 2: Klatki -> FFmpeg")
            success, frames_dir = self.save_frames_as_images(input_path)  # Bez limitu
            if success:
                return self.frames_to_video_ffmpeg(frames_dir, output_path, fps)
        
        # Strategia 3: Tylko klatki
        print("\n🎯 Strategia 3: Tylko klatki (zapasowa)")
        success, frames_dir = self.save_frames_as_images(input_path, max_frames=50)
        if success:
            print(f"✅ Klatki zapisane w: {frames_dir}")
            print("   Możesz je ręcznie przekonwertować na wideo")
            return True
        
        return False
    
    def process_with_opencv(self, input_path, writer):
        """Przetwarzanie przez OpenCV"""
        cap = cv2.VideoCapture(input_path)
        frame_count = 0
        success_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            enhanced = self.enhance_frame(frame)
            if enhanced is not None:
                success = writer.write(enhanced)
                if success:
                    success_count += 1
                else:
                    print(f"⚠️  Błąd zapisu klatki {frame_count}")
                    if frame_count > 10:  # Jeśli pierwsze 10 klatek to błędy, przerwij
                        break
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"   Postęp: {frame_count} klatek, {success_count} zapisanych")
        
        cap.release()
        print(f"✅ OpenCV: {success_count}/{frame_count} klatek")
        return success_count > frame_count * 0.8  # Sukces jeśli >80% klatek zapisano
    
    def cleanup(self):
        """Wyczyść pliki tymczasowe"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Wyczyszczono: {self.temp_dir}")
        except Exception as e:
            print(f"⚠️  Błąd czyszczenia: {e}")

def main():
    if len(sys.argv) < 2:
        print("🎬 Zaawansowany Wzmacniacz Wideo")
        print("Użycie: python3 robust_video_enhancer.py <input.mp4> [output.mp4]")
        print("")
        print("System automatycznie:")
        print("- Testuje dostępne kodeki")
        print("- Próbuje OpenCV VideoWriter")
        print("- Używa FFmpeg jako backup")
        print("- Zapisuje klatki jako obrazy w ostateczności")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "enhanced_" + os.path.basename(input_file)
    
    if not os.path.exists(input_file):
        print(f"✗ Plik {input_file} nie istnieje")
        return
    
    enhancer = DiagnosticVideoEnhancer()
    
    try:
        success = enhancer.process_video_robust(input_file, output_file)
        if success:
            print(f"\n🎉 Sukces! Sprawdź wyniki.")
        else:
            print(f"\n💥 Przetwarzanie nie powiodło się")
    finally:
        enhancer.cleanup()

if __name__ == "__main__":
    main()
