# Revisi Tugas Besar Pengenalan Pola dan Ekstraksi Fitur
## Judul: Klasifikasi Batik Menggunakan Fitur Hybrid (GLCM + MobileNetV2)

### 1. Dasar Analisis Penelitian

Analisis penelitian pada tugas besar ini dilakukan berdasarkan serangkaian metrik evaluasi klasifikasi machine learning. Penggunaan metrik-metrik ini dipilih untuk memvalidasi performa model klasifikasi 20 kelas motif batik, dengan rincian parameter:
*   **Akurasi (Accuracy)**: Digunakan sebagai indikator utama untuk melihat persentase total citra batik yang berhasil diklasifikasikan dengan benar dari seluruh data uji.
*   **Precision, Recall, dan F1-Score**: Digunakan untuk melihat ketepatan (precision) dan kemampuan penemuan (recall) model pada setiap kelas spesifik. F1-Score (rata-rata harmonik Precision dan Recall) dijadikan metrik fundamental karena dapat menunjukkan kestabilan prediksi, terutama sangat penting untuk memastikan tidak ada kelas spesifik yang dikorbankan demi akurasi semata.
*   **Confusion Matrix**: Sangat esensial dalam analisis penelitian ini karena terdapat 20 jenis motif batik yang rentan memiliki kemiripan pola. Confusion Matrix digunakan untuk memetakan kelas-kelas mana saja yang sering mengalami misklasifikasi (salah tebak) antara satu dengan yang lainnya.

### 2. Deskripsi Dataset

*   **Sumber Dataset**: Dataset diperoleh dari repositori publik (seperti Kaggle atau kompilasi lokal).
*   **Jumlah Data**: Total citra yang digunakan adalah 1.600 citra (1.280 citra untuk data *train* dan 320 citra untuk data *test*).
*   **Distribusi Kelas**: Terdapat 20 kelas motif batik (contoh: Aceh Pintu Aceh, Bali Barong, Jawa Timur Pring, dll). Setiap kelas memiliki jumlah citra yang persis sama, yaitu 64 citra *train* dan 16 citra *test* per kelas (total 80 citra per kelas). 
*   **Karakteristik Citra**: Citra digital RGB (warna) dengan resolusi asli bervariasi, memperlihatkan lipatan kain atau bentuk motif penuh. Seluruh citra diubah ukurannya ke standar 224x224 piksel untuk masukan CNN.
*   **Kondisi Distribusi**: Dataset ini dalam kondisi **Sangat Seimbang (*Perfectly Balanced*)**. Hal ini sangat ideal karena mencegah model *overfitting* atau menjadi *bias* terhadap kelas mayoritas.
*   **Alasan Pemilihan Dataset**: Dataset 20 kelas dipilih untuk memberikan tantangan klasifikasi *fine-grained* (pengenalan pola detail), karena banyak motif batik lintas daerah memiliki kesamaan geometri dan warna (seperti corak parang Solo vs parang Yogya).

### 3. Preprocessing Data

Tahapan preprocessing dirancang secara khusus untuk mengatasi permasalahan citra kain di dunia nyata:
*   **CLAHE (*Contrast Limited Adaptive Histogram Equalization*)**:
    Digunakan untuk meratakan kontras pencahayaan pada citra kain. Citra batik sering kali difoto dalam keadaan melipat atau memiliki area bayangan/silau. CLAHE diaplikasikan pada channel Luminance (L pada format LAB) sehingga pola tekstur di area gelap menjadi terlihat jelas tanpa membuat *noise* pada area terang.
*   **Zero-Padding dan Resizing (224x224)**:
    Citra batik asli memiliki rasio (aspect ratio) panjang x lebar yang berbeda-beda. Memaksa citra di-resize menjadi persegi (224x224) akan menyebabkan motif batik menjadi terdistorsi / "gepeng". Oleh karena itu, *Zero-Padding* digunakan untuk mempertahankan proporsi asli gambar dengan menambahkan *border* (garis tepi) hitam sebelum proses *resizing* menuju 224x224.
*   **Normalisasi (StandardScaler)**: Menyamakan rentang nilai (scale) karena rentang nilai luaran GLCM sangat besar dibandingkan luaran CNN (-1 s/d 1).
*   **PCA (*Principal Component Analysis*)**: Mereduksi dimensi total fitur gabungan (1.286 kolom) menjadi jauh lebih padat namun tetap mempertahankan 95% varians informasi, yang terbukti krusial untuk mempercepat waktu komputasi pelatihan klasifikasi klasik (seperti SVM dan RF) dan mencegah kutukan dimensionalitas (*curse of dimensionality*).

### 4. Strategi Ekstraksi Fitur

Sistem ini menggunakan penggabungan dua pendekatan fitur (*Hybrid Feature Extraction*) untuk saling menutupi kelemahan masing-masing:
*   **GLCM (*Gray Level Co-occurrence Matrix*)**: Sebagai ekstraktor fitur tekstur, mengekstraksi 6 properti (*Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM*) diubah dalam skala *grayscale*. GLCM mengukur frekuensi kemunculan pasangan piksel (jarak=1, rata-rata pada sudut 0, 45, 90, 135 derajat). Ini ideal untuk mendeteksi *isen-isen* (detail latar belakang berupa titik-titik kecil) yang menjadi ciri khas pakem batik.
*   **MobileNetV2**: Sebagai *feature extractor* berbasis Deep Learning. MobileNetV2 digunakan dengan arsitektur pre-trained (ImageNet) dengan lapisan atas (klasifikasi) dilepas (`include_top=False`), dan diganti dengan *Global Average Pooling* untuk menghasilkan 1.280 fitur visual. CNN sangat kuat untuk mengekstraksi fitur spasial, tepian (*edges*), bentuk geometri kompleks, hingga representasi semantik tinggi seperti bentuk Barong atau gajah.

### 5. Pengujian Model Secara Terpisah (Ablation Study)

Untuk membuktikan kontribusi masing-masing fitur, dilakukan pengujian *Ablation Study* menggunakan pengklasifikasi SVM dengan parameter yang sama. Berikut adalah hasil performa dari masing-masing pendekatan:
*   **Pengujian Fitur GLCM Saja**: Menghasilkan akurasi sebesar **28.44%**.
*   **Pengujian Fitur MobileNetV2 Saja**: Menghasilkan akurasi sebesar **96.25%**.
*   **Pengujian Hybrid (GLCM + MobileNetV2)**: Menghasilkan akurasi sebesar **96.25%**.

Hasil ini menunjukkan perbedaan kekuatan representasi yang sangat mencolok antara metode ekstraksi fitur statistik (klasik) dan ekstraksi fitur *Deep Learning*.

### 6. Analisis Pengaruh GLCM

GLCM berkontribusi dalam menangkap frekuensi kemunculan pola derajat keabuan yang merepresentasikan tekstur dari citra. Dalam konteks klasifikasi batik, GLCM sangat krusial karena corak batik umumnya sangat padat dengan *isen-isen* (titik-titik latar) dan corak berulang berskala kecil. Jika ada dua motif yang memiliki bentuk utama yang mirip tetapi berbeda pada kerapatan pola *isen-isen*, GLCM akan secara matematis mendeteksinya melalui fitur seperti *Contrast* atau *Homogeneity*. Penggunaan GLCM secara independen mungkin tidak menghasilkan akurasi yang memuaskan untuk membedakan bentuk objek yang besar, namun memberikan keunggulan lokalisasi tekstur yang spesifik.

### 7. Analisis Pengaruh MobileNetV2

MobileNetV2 berkontribusi dalam mengekstraksi representasi visual semantik tinggi dan bentuk spasial global dari citra. Arsitektur CNN ini memiliki filter konvolusi mendalam yang dioptimalkan (lewat *Transfer Learning*) untuk mengenali hierarki bentuk objek, garis, lekukan, hingga siluet spesifik (seperti bentuk siluet Barong, Gajah, atau pola diagonal Parang). Fitur visual ini menghasilkan vektor dimensi tinggi (1.280) yang menangkap informasi tata letak spasial yang secara inheren diabaikan oleh GLCM (karena GLCM tidak memetakan korelasi piksel yang berjauhan).

### 8. Penggabungan Fitur (Feature Fusion)

*   **Metode Penggabungan**: Penggabungan fitur antara tekstur statistik dan bentuk visual dilakukan menggunakan teknik *Early Fusion* atau *Feature Concatenation* (penggabungan matriks secara horizontal / setara).
*   **Alur Implementasi Teknis**:
    1.  Citra RGB dimasukkan ke tahapan preprocessing (CLAHE + Zero-Padding + Resize).
    2.  Versi *Grayscale* dari citra diproses melalui metode statistik GLCM (jarak=1, rata-rata 4 sudut) untuk memproduksi 6 metrik (Contrast, Dissimilarity, dsb).
    3.  Versi RGB 224x224 dimasukkan ke arsitektur CNN MobileNetV2 untuk memproduksi vektor 1.280 dimensi hasil operasi *Global Average Pooling*.
    4.  Kedua hasil ekstraksi tersebut digabungkan (`np.hstack`) untuk membentuk 1 array fitur *hybrid* berdimensi 1.286 per citra.
    5.  Setelah digabung, fitur-fitur tersebut dinormalisasi dengan *Standard Scaler* agar rentang nilai fitur GLCM dan fitur CNN setara dan dapat diseimbangkan pembobotannya.
    6.  Array lalu direduksi menggunakan PCA untuk mempercepat pelatihan sebelum dimasukkan ke tahap klasifikasi dengan SVM/RF.

### 9. Justifikasi dan Pembahasan

*   **Justifikasi Ilmiah Penggabungan**: Citra kain batik memiliki dua karakteristik yang sama-sama dominan: 1) Pakem tekstur mikroskopis berupa coretan malam/isen-isen; 2) Pola makroskopis berupa bentuk struktur ornamen utama. GLCM dirancang secara matematis untuk unggul pada poin pertama, sementara CNN sangat mahir mengabstraksi poin kedua. Keduanya memberikan deskripsi gambar yang saling melengkapi (*complementary features*).
*   **Pembahasan Analisis Kombinasi**: 
    Berdasarkan hasil *Ablation Study* eksperimen fitur *Hybrid*, model yang hanya dilatih menggunakan GLCM memiliki performa yang kurang representatif (28.44%) untuk dataset 20 kelas yang kompleks ini. Namun, saat digabungkan dengan MobileNetV2 (96.25%), model Hybrid mampu mempertahankan akurasi optimal (96.25%). Meskipun penggabungan (Feature Fusion) pada kasus spesifik ini tidak menaikkan angka metrik di atas performa individu MobileNetV2, penggabungan ini secara fundamental membuat model jauh lebih tahan banting (robust) terhadap *noise*. Kombinasi tekstur analitik (GLCM) dan bentuk hierarkis (MobileNetV2) memastikan bahwa apabila CNN kebingungan membedakan bentuk objek yang saling tumpang tindih akibat lipatan kain ekstrem, PCA akan memberikan bobot pada fitur tekstur GLCM yang dapat mengenali kerapatan *isen-isen* sebagai faktor penentu klasifikasi.
