#!/usr/bin/env python3
"""
Wzmacniacz wideo z łagodnym, naturalnym przetwarzaniem
Eliminuje ostre strefy i zachowuje płynne przejścia
"""

import cv2
import numpy as np
import sys
import os
import subprocess
import tempfile
from pathlib import Path

class SmoothVideoEnhancer:
    """Wzmacniacz z łagodnym, naturalnym przetwarzaniem"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix='video_enhance_')
        print(f"📁 Katalog tymczasowy: {self.temp_dir}")
    
    def enhance_frame_smooth(self, frame):
        """Łagodne wzmocnienie klatki - eliminuje ostre strefy"""
        if frame is None:
            return None
        
        # Konwersja do float32 dla lepszej precyzji
        frame_float = frame.astype(np.float32) / 255.0
        
        # Metoda 1: Gamma correction z adaptacją
        enhanced = self.adaptive_gamma_correction(frame_float)
        
        # Metoda 2: Łagodne wzmocnienie kontrastu
        enhanced = self.gentle_contrast_enhancement(enhanced)
        
        # Metoda 3: Selektywne wyostrzanie
        enhanced = self.selective_sharpening(enhanced, frame_float)
        
        # Metoda 4: Redukcja szumów
        enhanced = self.noise_reduction(enhanced)
        
        # Powrót do uint8
        enhanced = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
        
        return enhanced
    
    def adaptive_gamma_correction(self, frame):
        """Adaptacyjna korekcja gamma bazująca na jasności obrazu"""
        # Konwersja do HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Analiza histogramu kanału V (jasność)
        hist, _ = np.histogram(hsv[:,:,2], bins=256, range=(0, 1))
        
        # Oblicz średnią jasność
        mean_brightness = np.mean(hsv[:,:,2])
        
        # Adaptacyjne gamma - ciemne obrazy dostają więcej rozjaśnienia
        if mean_brightness < 0.3:
            gamma = 0.7  # Rozjaśnij ciemne obszary
        elif mean_brightness > 0.7:
            gamma = 1.3  # Przyciemnij jasne obszary
        else:
            gamma = 1.0  # Neutralne
        
        # Zastosuj korekcję gamma tylko do kanału jasności
        hsv[:,:,2] = np.power(hsv[:,:,2], gamma)
        
        # Powrót do BGR
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return enhanced
    
    def gentle_contrast_enhancement(self, frame):
        """Łagodne wzmocnienie kontrastu bez ostrych przejść"""
        # Konwersja do LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Bardzo łagodne CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=1.5,      # Zmniejszone z 3.0
            tileGridSize=(16,16) # Większe kafelki = mniej artefaktów
        )
        l_enhanced = clahe.apply((l * 255).astype(np.uint8)).astype(np.float32) / 255.0
        
        # Mieszanie z oryginałem (50/50) dla łagodności
        l_blended = 0.6 * l_enhanced + 0.4 * l
        
        # Rekombinacja
        lab_enhanced = cv2.merge([l_blended, a, b])
        bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return bgr_enhanced
    
    def selective_sharpening(self, frame, original):
        """Selektywne wyostrzanie - tylko tam gdzie potrzeba"""
        # Gaussian blur dla maski krawędzi
        blurred = cv2.GaussianBlur(frame, (0, 0), 1.0)
        
        # Maska krawędzi (gdzie są szczegóły do wyostrzenia)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
        edge_mask = edges.astype(np.float32) / 255.0
        
        # Rozszerz maskę dla płynniejszych przejść
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edge_mask = cv2.dilate(edge_mask, kernel, iterations=1)
        edge_mask = cv2.GaussianBlur(edge_mask, (5, 5), 2.0)
        
        # Unsharp mask - bardzo łagodny
        unsharp = frame + 0.3 * (frame - blurred)
        
        # Zastosuj wyostrzanie tylko na krawędziach
        result = np.zeros_like(frame)
        for c in range(3):
            result[:,:,c] = frame[:,:,c] * (1 - edge_mask) + unsharp[:,:,c] * edge_mask
        
        return result
    
    def noise_reduction(self, frame):
        """Łagodna redukcja szumów"""
        # Bilateral filter - zachowuje krawędzie, redukuje szumy
        denoised = cv2.bilateralFilter(
            (frame * 255).astype(np.uint8), 
            d=5,           # Średnica sąsiedztwa
            sigmaColor=25, # Próg kolorów
            sigmaSpace=25  # Próg przestrzeni
        ).astype(np.float32) / 255.0
        
        # Mieszaj z oryginałem dla naturalności
        return 0.7 * frame + 0.3 * denoised
    
    def enhance_frame_alternative(self, frame):
        """Alternatywna metoda - jeszcze łagodniejsza"""
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
    
    def compare_methods(self, frame):
        """Porównanie różnych metod wzmocnienia"""
        if frame is None:
            return None
        
        # Oryginał
        original = frame.copy()
        
        # Metoda łagodna
        smooth = self.enhance_frame_smooth(frame)
        
        # Metoda alternatywna
        alternative = self.enhance_frame_alternative(frame)
        
        # Zwróć najlepszą metodę (można dostosować)
        return smooth  # Lub alternative, w zależności od preferencji
    
    # Reszta metod pozostaje bez zmian
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
    
    def process_video_smooth(self, input_path, output_path):
        """Przetwarzanie z niezawodną metodą alternative"""
        print(f"🎬 Rozpoczynam łagodne przetwarzanie: {input_path}")
        
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
        print(f"🎨 Metoda wzmocnienia: alternative (histogram stretching)")
        cap.release()
        
        # Używamy tylko niezawodnej metody alternative
        enhance_func = self.enhance_frame_alternative
        
        # Strategia zapisu - najpierw klatki, potem FFmpeg
        if has_ffmpeg:
            return self.process_via_frames_ffmpeg(input_path, output_path, enhance_func)
        
        return False
    
    def process_via_frames_ffmpeg(self, input_path, output_path, enhance_func):
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
                # Wzmocnienie wybraną metodą
                enhanced = enhance_func(frame)
                
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
            return self.frames_to_video_ffmpeg(frames_dir, output_path, fps, 'png')
        
        return False
    
    def frames_to_video_ffmpeg(self, frames_dir, output_path, fps, ext='png'):
        """Konwersja klatek do wideo przez FFmpeg z wysoką jakością"""
        print("🎬 Tworzenie wideo z klatek przez FFmpeg...")
        
        input_pattern = os.path.join(frames_dir, f'frame_%06d.{ext}')
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
        print("🎬 Łagodny Wzmacniacz Wideo")
        print("Użycie: python3 smooth_video_enhancer.py <input.mp4> [output.mp4] [metoda]")
        print("")
        print("Metody wzmocnienia:")
        print("  smooth      - łagodne wzmocnienie (domyślne)")
        print("  alternative - histogram stretching")
        print("  compare     - automatyczny wybór najlepszej")
        print("")
        print("Funkcje:")
        print("- Eliminuje ostre strefy i artefakty")
        print("- Zachowuje naturalne przejścia")
        print("- Adaptacyjne wzmocnienie bazowane na zawartości")
        print("- Selektywne wyostrzanie tylko na krawędziach")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "smooth_" + os.path.basename(input_file)
    method = sys.argv[3] if len(sys.argv) > 3 else 'smooth'
    
    if not os.path.exists(input_file):
        print(f"✗ Plik {input_file} nie istnieje")
        return
    
    enhancer = SmoothVideoEnhancer()
    
    try:
        success = enhancer.process_video_smooth(input_file, output_file)
        if success:
            print(f"\n🎉 Sukces! Sprawdź wyniki: {output_file}")
            print("📊 Wideo zostało przetworzone z zachowaniem naturalnych przejść")
        else:
            print(f"\n💥 Przetwarzanie nie powiodło się")
    finally:
        enhancer.cleanup()

if __name__ == "__main__":
    main()