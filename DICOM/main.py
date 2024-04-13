import pydicom
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import cv2

# Modeli yükle
model = load_model('simple_unet_model.h5')


# DICOM dosyasını oku
dicom_name = '1.3.46.423632.3373142023221753020.22.dcm'
path ='C:\\Users\\egeen\\Downloads\\'
dicom_file_path = path + dicom_name
dicom_file = pydicom.dcmread(dicom_file_path,force=True)

if "TransferSyntaxUID" not in dicom_file.file_meta:
    dicom_file.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

# Görüntüyü numpy dizisine çevir
image = dicom_file.pixel_array

# Görüntüyü yeniden boyutlandır
resized_image = cv2.resize(dicom_file.pixel_array, (256, 256))

# Görüntüyü model için uygun hale getir
image = np.expand_dims(resized_image, axis=-1)  # Model için boyut ekleyin
image = np.expand_dims(image, axis=0)  # Batch boyutu ekleyin
image = image / 255.0  # Görüntüyü normalize et

# Tahmini yap
predicted_mask = model.predict(image)

plt.figure(figsize=(10, 4))

# Orijinal DICOM Görüntüsü
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image[0, :, :, 0], cmap='gray')

# Kontürlü Görüntü
plt.subplot(1, 2, 2)
plt.title('Contoured Image')
plt.imshow(predicted_mask[0, :, :, 0], cmap='gray')

# Görüntüyü kaydet
plt.savefig('contoured_image.png')
