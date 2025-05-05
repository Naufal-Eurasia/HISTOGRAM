Analisis HISTOGRAM HSV,INDEX,REFERENSI pada biji kopi,tomat,dan anggur untuk melihat kematangannya
```python
# --- 1. Instalasi Library (Jika Diperlukan) ---
!pip install opencv-python-headless matplotlib

# --- 2. Import Library ---
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# --- 3. Upload Gambar ---
uploaded = files.upload()
filename = list(uploaded.keys())[0]

# --- 4. Load Gambar dan Konversi ke HSV ---
img = cv2.imread(filename)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Tampilkan gambar asli
plt.figure(figsize=(6, 6))
plt.imshow(img_rgb)
plt.title("Gambar Biji Kopi (RGB)")
plt.axis('off')
plt.show()

# --- 5. Ekstrak Kanal HSV dan Histogram ---
h, s, v = cv2.split(img_hsv)

plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
plt.hist(h.ravel(), 180, [0,180], color='r')
plt.title('Histogram Hue (H)')
plt.xlabel('Hue Value')
plt.ylabel('Frequency')

plt.subplot(1,3,2)
plt.hist(s.ravel(), 256, [0,256], color='g')
plt.title('Histogram Saturation (S)')
plt.xlabel('Saturation Value')

plt.subplot(1,3,3)
plt.hist(v.ravel(), 256, [0,256], color='b')
plt.title('Histogram Value (V)')
plt.xlabel('Value')

plt.tight_layout()
plt.show()

# --- 6. Penjelasan Otomatis Berdasarkan Histogram Hue ---
# Ambil rata-rata dan modus dari Hue
hue_mean = int(np.mean(h))
hue_mode = int(np.bincount(h.ravel()).argmax())

# Klasifikasi berdasarkan Hue dominan
if hue_mode >= 35 and hue_mode <= 85:
    status = "Biji kopi kemungkinan masih mentah (hijau)."
elif hue_mode >= 15 and hue_mode < 35:
    status = "Biji kopi kemungkinan setengah matang (kuning/cokelat muda)."
elif hue_mode < 15:
    status = "Biji kopi kemungkinan sudah matang atau sangat matang (cokelat tua/hitam)."
else:
    status = "Warna biji kopi tidak terdeteksi secara spesifik."

# Tampilkan hasil analisis
print("=== Analisis Histogram Hue ===")
print(f"- Rata-rata Hue       : {hue_mean}")
print(f"- Mode (dominasi Hue) : {hue_mode}")
print(f"- Keterangan           : {status}")
```
Hasil output HISTOGRAM HSV pada biji kopi :
![image](https://github.com/user-attachments/assets/f16497dd-999f-420d-baa4-b66ed9d3950c)
![image](https://github.com/user-attachments/assets/5c7abacf-f689-4f61-9ff0-b2f71a63b083)

```python
from google.colab import files
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Upload satu gambar biji kopi
uploaded = files.upload()
filename = list(uploaded.keys())[0]
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# 2. Fungsi untuk menghitung HQI dan deskripsi
def calculate_hqi(image):
    hist_actual = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_actual = hist_actual / hist_actual.sum()

    hist_ideal = np.ones((256, 1), dtype=np.float32)
    hist_ideal = hist_ideal / hist_ideal.sum()

    diff = np.abs(hist_actual - hist_ideal)
    hqi = 1 - (np.sum(diff) / 2)

    if hqi >= 0.90:
        desc = "Distribusi histogram sangat merata, menunjukkan kualitas pencahayaan dan kontras yang sangat baik."
    elif hqi >= 0.75:
        desc = "Distribusi histogram cukup merata. Gambar memiliki pencahayaan dan kontras yang baik."
    elif hqi >= 0.50:
        desc = "Histogram menunjukkan distribusi sedang. Mungkin ada ketidakseimbangan dalam pencahayaan atau kontras."
    else:
        desc = "Histogram sangat tidak merata. Gambar kemungkinan terlalu gelap, terlalu terang, atau memiliki kontras rendah."

    return hqi, desc

# 3. Tampilkan histogram dan penjelasan rinci untuk gambar biji kopi
hqi_score, explanation = calculate_hqi(img)

# Menampilkan histogram gambar biji kopi
plt.figure(figsize=(6, 3))
plt.hist(img.ravel(), bins=256, range=[0, 256], color='brown', alpha=0.8)
plt.title('Histogram - Gambar Biji Kopi')
plt.xlabel('Intensitas Piksel (0-255)')
plt.ylabel('Frekuensi')
plt.grid(True)
plt.tight_layout()
plt.show()

# Menampilkan hasil HQI dan penjelasan rinci
print(f"\033[1m[Gambar Biji Kopi]\033[0m")
print(f"HQI: {hqi_score:.4f}")
print(f"Analisis: {explanation}\n{'-'*60}")

```
Hasil output HISTOGRAM INDEX pada biji kopi :
![image](https://github.com/user-attachments/assets/f6d7624e-bbd8-4558-8da3-cf6a08b0631b)

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from google.colab import files

# Mengupload gambar
uploaded = files.upload()

# Memastikan file di-upload
for filename in uploaded.keys():
    print(f"File yang di-upload: {filename}")

    # Membaca gambar biji kopi
    image = Image.open(filename)

    # Mengonversi gambar ke grayscale
    gray_image = image.convert('L')

    # Menghitung histogram
    histogram = np.array(gray_image).flatten()
    hist, bins = np.histogram(histogram, bins=256, range=(0, 255))

    # Menampilkan histogram
    plt.figure(figsize=(10, 6))
    plt.plot(bins[:-1], hist, color='brown', lw=2)
    plt.title('Histogram Gambar Biji Kopi')
    plt.xlabel('Intensitas Piksel')
    plt.ylabel('Frekuensi')
    plt.grid(True)
    plt.show()

    # Menampilkan penjelasan rinci
    total_pixels = len(histogram)
    mean_intensity = np.mean(histogram)
    std_deviation = np.std(histogram)
    min_intensity = np.min(histogram)
    max_intensity = np.max(histogram)

    print("\nPenjelasan Rinci Histogram:")
    print(f"Jumlah total piksel: {total_pixels}")
    print(f"Rata-rata intensitas piksel: {mean_intensity:.2f}")
    print(f"Standar deviasi intensitas piksel: {std_deviation:.2f}")
    print(f"Intensitas piksel minimum: {min_intensity}")
    print(f"Intensitas piksel maksimum: {max_intensity}")
```
Hasil output HISTOGRAM REFERENSI pada biji kopi :
![image](https://github.com/user-attachments/assets/7865f695-e7b1-4fa8-8f96-96d8dbebcad7)

```python
# --- 1. Instalasi Library (Jika Diperlukan) ---
!pip install opencv-python-headless matplotlib

# --- 2. Import Library ---
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# --- 3. Upload Dua Gambar ---
uploaded = files.upload()
uploaded_files = list(uploaded.keys())

# Memastikan dua gambar di-upload
if len(uploaded_files) != 2:
    print("Harap upload dua gambar tomat (hijau dan merah).")
else:
    # --- 4. Proses dan Tampilkan Gambar ---
    for filename in uploaded_files:
        img = cv2.imread(filename)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Tampilkan gambar asli
        plt.figure(figsize=(6, 6))
        plt.imshow(img_rgb)
        plt.title(f"Gambar Tomat: {filename}")
        plt.axis('off')
        plt.show()

        # --- 5. Ekstrak Kanal HSV dan Histogram ---
        h, s, v = cv2.split(img_hsv)

        plt.figure(figsize=(15,4))

        plt.subplot(1,3,1)
        plt.hist(h.ravel(), 180, [0,180], color='r')
        plt.title('Histogram Hue (H)')
        plt.xlabel('Hue Value')
        plt.ylabel('Frequency')

        plt.subplot(1,3,2)
        plt.hist(s.ravel(), 256, [0,256], color='g')
        plt.title('Histogram Saturation (S)')
        plt.xlabel('Saturation Value')

        plt.subplot(1,3,3)
        plt.hist(v.ravel(), 256, [0,256], color='b')
        plt.title('Histogram Value (V)')
        plt.xlabel('Value')

        plt.tight_layout()
        plt.show()

        # --- 6. Penjelasan Otomatis Berdasarkan Histogram Hue ---
        hue_mean = int(np.mean(h))
        hue_mode = int(np.bincount(h.ravel()).argmax())

        # Klasifikasi berdasarkan Hue dominan
        if hue_mode >= 35 and hue_mode <= 85:
            status = "Tomat kemungkinan belum matang (hijau)."
        elif hue_mode >= 15 and hue_mode < 35:
            status = "Tomat kemungkinan setengah matang (kuning/cokelat muda)."
        elif hue_mode < 15:
            status = "Tomat kemungkinan sudah matang (merah)."
        else:
            status = "Warna tomat tidak terdeteksi secara spesifik."

        # Tampilkan hasil analisis
        print(f"=== Analisis Histogram Hue untuk {filename} ===")
        print(f"- Rata-rata Hue       : {hue_mean}")
        print(f"- Mode (dominasi Hue) : {hue_mode}")
        print(f"- Keterangan           : {status}")
        print("-" * 60)
```
Hasil output HISTOGRAM HSV pada tomat :
TOMAT HIJAU
![image](https://github.com/user-attachments/assets/67f26b4d-1cc9-4359-8af1-b98c8f24cda4)
![image](https://github.com/user-attachments/assets/3f7d27cb-9c7d-4225-bd4c-63503960e0e0)
TOMAT MERAH
![image](https://github.com/user-attachments/assets/bb5064ee-f1ea-4787-914c-850c5e522326)
![image](https://github.com/user-attachments/assets/06200d20-903e-43f2-ad39-43265e0b2d9b)

```python
# --- 1. Instalasi Library (Jika Diperlukan) ---
!pip install opencv-python-headless matplotlib

# --- 2. Import Library ---
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# --- 3. Upload Dua Gambar (pisang dan anggur) ---
uploaded = files.upload()
uploaded_files = list(uploaded.keys())

# Memastikan dua gambar di-upload
if len(uploaded_files) != 2:
    print("Harap upload dua gambar buah (pisang dan anggur).")
else:
    # --- 4. Proses dan Tampilkan Gambar ---
    for filename in uploaded_files:
        img = cv2.imread(filename)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Tampilkan gambar asli
        plt.figure(figsize=(6, 6))
        plt.imshow(img_rgb)
        plt.title(f"Gambar Buah: {filename}")
        plt.axis('off')
        plt.show()

        # --- 5. Ekstrak Kanal HSV dan Histogram ---
        h, s, v = cv2.split(img_hsv)

        plt.figure(figsize=(15,4))

        plt.subplot(1,3,1)
        plt.hist(h.ravel(), 180, [0,180], color='r')
        plt.title('Histogram Hue (H)')
        plt.xlabel('Hue Value')
        plt.ylabel('Frequency')

        plt.subplot(1,3,2)
        plt.hist(s.ravel(), 256, [0,256], color='g')
        plt.title('Histogram Saturation (S)')
        plt.xlabel('Saturation Value')

        plt.subplot(1,3,3)
        plt.hist(v.ravel(), 256, [0,256], color='b')
        plt.title('Histogram Value (V)')
        plt.xlabel('Value')

        plt.tight_layout()
        plt.show()

        # --- 6. Penjelasan Otomatis Berdasarkan Histogram Hue ---
        hue_mean = int(np.mean(h))
        hue_mode = int(np.bincount(h.ravel()).argmax())

        # --- 7. Deteksi Buah Berdasarkan Nama File dan Hue ---
        if "pisang" in filename.lower():
            if 35 <= hue_mode <= 85:
                status = "Pisang kemungkinan belum matang (hijau)."
            elif 20 <= hue_mode < 35:
                status = "Pisang kemungkinan matang (kuning)."
            else:
                status = "Warna pisang tidak terdeteksi secara spesifik."
        elif "anggur" in filename.lower():
            if 35 <= hue_mode <= 85:
                status = "Anggur kemungkinan belum matang (hijau)."
            elif hue_mode < 15 or hue_mode > 160:
                status = "Anggur kemungkinan matang (merah keunguan)."
            else:
                status = "Warna anggur tidak terdeteksi secara spesifik."
        else:
            status = "Jenis buah tidak dikenali dari nama file."

        # Tampilkan hasil analisis
        print(f"=== Analisis Histogram Hue untuk {filename} ===")
        print(f"- Rata-rata Hue       : {hue_mean}")
        print(f"- Mode (dominasi Hue) : {hue_mode}")
        print(f"- Keterangan           : {status}")
        print("-" * 60)
```
Hasil output HISTOGRAM HSV pada anggur :
ANGGUR HIJAU
![image](https://github.com/user-attachments/assets/7a5a8efa-ca5f-44ea-b9a4-2c3b6a417260)
![image](https://github.com/user-attachments/assets/ad658cd7-e103-4005-99b9-9567bbfe84de)

ANGGUR MERAH
![image](https://github.com/user-attachments/assets/e3b34c50-0c83-45e4-a00e-0c6dc9c1fe7a)
![image](https://github.com/user-attachments/assets/62e7e220-de1f-4acd-9da6-2eae2bd504ed)





