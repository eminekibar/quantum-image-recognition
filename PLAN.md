# Kuantum Makine Öğrenmesi ile Görüntü Tanıma Sistemi
## Proje Planı — Grup 3

---

## 1. Proje Özeti

Gömülü bir sistem üzerinde çalışan, kameradan görüntü toplayan ve **hibrit kuantum-klasik sinir ağı** kullanarak görüntüleri tanıyan bir sistem. Başlangıç görevi olarak **el yazısı rakam tanıma (MNIST)** seçilmiştir; gerçek kamera olmadan test edilebilmesi için `MockCamera` altyapısı mevcuttur.

### Temel Bileşenler
- **Hibrit QNN**: Klasik CNN encoder → Kuantum devre (PennyLane) → Klasik sınıflandırıcı
- **Kamera katmanı**: Gerçek webcam, mock (test), statik görüntü — tek satır config değişikliğiyle geçiş
- **Web arayüzü**: Flask tabanlı, tarayıcıdan canlı demo yapılabilir
- **Test altyapısı**: pytest ile otomatik testler

---

## 2. Teknoloji Yığını

| Araç | Versiyon | Amaç |
|------|----------|-------|
| Python | 3.10+ | Ana dil |
| PennyLane | 0.36+ | Kuantum devre & gradyan |
| PyTorch | 2.2+ | Klasik ML katmanları |
| OpenCV | 4.9+ | Kamera & görüntü işleme |
| Flask | 3.0+ | Web arayüzü |
| pytest | 8.0+ | Otomatik testler |
| torchvision | 0.17+ | MNIST veri seti |
| PyYAML | 6.0+ | Config dosyası okuma |
| matplotlib | 3.8+ | Grafik & görselleştirme |
| Jupyter | 7.0+ | Notebook'lar |

---

## 3. Klasör Yapısı

```
quantum_vision/
│
├── src/                         # Ana kaynak kodlar
│   ├── quantum_model.py         # Hibrit QNN modeli
│   ├── train.py                 # Eğitim scripti
│   ├── evaluate.py              # Değerlendirme & metrikler
│   └── predict.py               # Tahmin arayüzü
│
├── camera/                      # Kamera soyutlama katmanı
│   ├── base_camera.py           # Soyut temel sınıf
│   ├── live_camera.py           # Gerçek webcam (OpenCV)
│   ├── mock_camera.py           # Test kamerası (MNIST'ten frame)
│   └── image_camera.py          # Statik görüntüden okur
│
├── web/                         # Web arayüzü
│   ├── app.py                   # Flask sunucu
│   ├── templates/
│   │   └── index.html           # Ana sayfa
│   └── static/
│       ├── style.css
│       └── script.js
│
├── tests/                       # Test dosyaları
│   ├── test_model.py            # Model forward pass testleri
│   ├── test_camera.py           # Kamera testleri
│   ├── test_predict.py          # Tahmin doğruluğu testleri
│   └── test_pipeline.py         # Uçtan uca entegrasyon testi
│
├── notebooks/                   # Jupyter defterleri
│   ├── 01_quantum_basics.ipynb  # Kuantum devre görselleştirme
│   ├── 02_training.ipynb        # Eğitim süreci adım adım
│   └── 03_results.ipynb         # Sonuçlar & karışıklık matrisi
│
├── data/                        # Otomatik indirilir, git'e eklenmez
│   └── .gitkeep
│
├── checkpoints/                 # Kaydedilen model ağırlıkları
│   └── .gitkeep
│
├── assets/                      # Görseller, sunum materyalleri
│   └── .gitkeep
│
├── config.yaml                  # Tüm ayarlar tek dosyada
├── requirements.txt             # Üretim bağımlılıkları
├── requirements-dev.txt         # Geliştirme bağımlılıkları
├── setup.py                     # Paket kurulum dosyası
├── .gitignore
└── README.md
```

---

## 4. Model Mimarisi — Hibrit QNN

```
Giriş Görüntüsü (28x28 gri)
        │
        ▼
┌─────────────────────────┐
│   Klasik CNN Encoder    │  Conv2d → ReLU → MaxPool × 2
│   (28x28 → 4 özellik)  │  Fully Connected → 4 nöron
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Kuantum Devre         │  4 qubit, 3 katman
│   (PennyLane)           │  RY, CNOT kapıları
│                         │  Beklenti değerleri → 4 çıktı
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Klasik Sınıflandırıcı │  Linear(4 → 10)
│   (10 sınıf)            │  Softmax
└─────────────────────────┘
        │
        ▼
   Tahmin (0-9 rakam)
```

### Kuantum Devre Detayı
- **Qubit sayısı**: 4
- **Katman sayısı**: 3 (ayarlanabilir)
- **Ansatz**: StronglyEntanglingLayers
- **Ölçüm**: Her qubit için PauliZ beklentisi
- **Simülatör**: `default.qubit` (gerçek donanım gerektirmez)

---

## 5. Kamera Soyutlama Katmanı

```
BaseCamera (ABC)
    ├── read_frame() → np.ndarray   # Her subclass implement eder
    ├── is_open() → bool
    └── release()

LiveCamera(BaseCamera)
    └── OpenCV VideoCapture(0)

MockCamera(BaseCamera)
    └── MNIST test setinden rastgele frame döndürür
        (gerçek kamera olmadan tam test yapılabilir)

ImageCamera(BaseCamera)
    └── Belirli bir klasördeki görüntüleri sırayla döndürür
        (demo ve sunum için ideal)
```

**config.yaml'da tek satır değiştirme:**
```yaml
camera:
  mode: mock   # mock | live | image
```

---

## 6. Test Stratejisi

| Test Dosyası | Ne Test Eder | Koşullar |
|-------------|-------------|----------|
| `test_model.py` | Forward pass, çıktı boyutu, gradyan akışı | Her commit |
| `test_camera.py` | MockCamera frame boyutu, dtype, değer aralığı | Kamera kodu değişince |
| `test_predict.py` | Bilinen görüntülere doğru/yanlış tahmin oranı | Model checkpoint'i varsa |
| `test_pipeline.py` | Kamera → ön işleme → model → çıktı tam akış | Sunumdan önce |

**Çalıştırma:** `pytest tests/ -v`

---

## 7. Haftalık Geliştirme Takvimi

| Hafta | Hedef | Sorumlu | Tamamlandı mı? |
|-------|-------|---------|----------------|
| 1 | Proje kurulumu, `config.yaml`, `requirements.txt`, klasör yapısı | Herkes | [ ] |
| 2 | `quantum_model.py` + `test_model.py` | Kişi 1 | [ ] |
| 3 | `camera/` modülü + `train.py` + `test_camera.py` | Kişi 2 | [ ] |
| 4 | `evaluate.py` + `notebooks/01` ve `02` | Kişi 1 | [ ] |
| 5 | Web arayüzü (`web/`) + `test_pipeline.py` | Kişi 3 | [ ] |
| 6 | `LiveCamera` entegrasyonu + `predict.py` | Kişi 2 | [ ] |
| 7 | `notebooks/03` (sonuçlar) + README + sunum | Kişi 3 | [ ] |
| 8 | Son testler, demo video, Teknofest dosyası | Herkes | [ ] |

---

## 8. Görev Dağılımı (3 Kişi)

### Kişi 1 — Model & Eğitim
- `src/quantum_model.py`
- `src/train.py`
- `src/evaluate.py`
- `notebooks/01_quantum_basics.ipynb`
- `notebooks/02_training.ipynb`
- `tests/test_model.py`

### Kişi 2 — Kamera & Pipeline
- `camera/` tüm dosyalar
- `src/predict.py`
- `tests/test_camera.py`
- `tests/test_pipeline.py`

### Kişi 3 — Arayüz & Sunum
- `web/` tüm dosyalar
- `notebooks/03_results.ipynb`
- `assets/` görseller
- `README.md`

---

## 9. Başarı Kriterleri

| Kriter | Hedef |
|--------|-------|
| Test doğruluğu | > %85 (MNIST üzerinde) |
| Tahmin süresi (tek görüntü) | < 500ms |
| Çalışan demo | MockCamera ile laptop'ta çalışır |
| Gerçek kamera desteği | Hafta 6'da eklenir |
| Web arayüzü | Tarayıcıdan demo yapılabilir |
| Jupyter notebook'lar | Kuantum devre görsel olarak açıklanmış |

---

## 10. Riskler ve Önlemler

| Risk | Önlem |
|------|-------|
| Gerçek kamera yok | MockCamera ile geliştirme ve test yapılır |
| Kuantum simülasyon yavaş | Küçük model (4 qubit, 3 katman), az veri (2000 örnek) |
| Ekip üyeleri farklı hızda öğrenir | Her modül bağımsız; arayüz grubu kamera gerektirmez |
| Teknofest belgeler eksik | README.md sunum-ready formatında, notebook'lar hazır |

---

## 11. Kurulum (İlk Çalıştırma)

```bash
# 1. Bağımlılıkları kur
pip install -r requirements.txt

# 2. Modeli eğit (MockCamera ile çalışır)
python src/train.py

# 3. Testleri çalıştır
pytest tests/ -v

# 4. Web arayüzünü başlat
python web/app.py
# Tarayıcıda: http://localhost:5000

# 5. Notebook'ları aç
jupyter notebook notebooks/
```

---

## 12. Git Workflow

```bash
main          → stabil, çalışan kod
dev           → aktif geliştirme
feature/model → model geliştirme dalı
feature/camera → kamera dalı
feature/web   → arayüz dalı
```

**Kural:** `main`'e doğrudan push yok. Her özellik `dev`'e merge edilir, test geçince `main`'e alınır.

---

*Son güncelleme: 2026-04-07*
