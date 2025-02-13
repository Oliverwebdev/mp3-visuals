import numpy as np
import matplotlib.pyplot as plt
import librosa

# Pfad zur Audio-Datei (z.B. MP3 oder WAV)
audio_path = "./Oimara - Wackelkontakt [FVgQW3iv90M].mp3"

# Audio laden (librosa wandelt alles in Mono um)
y, sr = librosa.load(audio_path, sr=None)
duration = librosa.get_duration(y=y, sr=sr)

# Größe und weitere Parameter für die Visualisierung
WIDTH, HEIGHT = 800, 600
CHUNK_SIZE = 1024  # Anzahl Samples für die FFT-Berechnung


def make_frame(t):
    """
    Erzeugt für einen gegebenen Zeitpunkt t ein Bild (Frame)
    mit einer FFT-basierten Balkenvisualisierung.
    """
    # Bestimme den Startindex im Audiosignal
    start_sample = int(t * sr)
    # Wähle einen kleinen Abschnitt (Chunk) aus
    end_sample = min(start_sample + CHUNK_SIZE, len(y))
    window = y[start_sample:end_sample]

    # Falls der Chunk leer sein sollte (z.B. am Ende), fülle mit Nullen
    if len(window) == 0:
        window = np.zeros(CHUNK_SIZE)

    # FFT durchführen und nur den positiven Frequenzbereich verwenden
    fft = np.abs(np.fft.fft(window))
    fft = fft[:len(fft)//2]

    # Normalisieren (damit die Balken passend skaliert sind)
    if np.max(fft) > 0:
        fft = fft / np.max(fft)

    # Erstelle ein Plot mit matplotlib
    fig, ax = plt.subplots(figsize=(WIDTH/100, HEIGHT/100), dpi=100)
    ax.axis('off')

    # X-Achse: Frequenzbereiche (bis ca. Nyquist-Frequenz)
    freqs = np.linspace(0, sr/2, len(fft))

    # Zeichne Balken (hier kannst du noch mit Farben und weiteren Effekten experimentieren)
    ax.bar(freqs, fft, width=(sr/2)/len(fft), color='limegreen')

    # Optional: Achsenlimits setzen
    ax.set_xlim(0, sr/2)
    ax.set_ylim(0, 1)

    # Das Bild als NumPy-Array extrahieren
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    return img


# Audio-Clip laden (für die Synchronisation)
audio_clip = AudioFileClip(audio_path)

# Video-Clip erstellen: für jeden Zeitpunkt t wird make_frame(t) aufgerufen
video_clip = VideoClip(make_frame, duration=duration)

# Audio hinzufügen
video_clip = video_clip.set_audio(audio_clip)

# Video exportieren (hier auf MP4, 24 FPS; passe die Parameter bei Bedarf an)
video_clip.write_videofile("visualizer_video.mp4", fps=24)
