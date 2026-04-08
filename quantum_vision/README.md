# Kuantum Makine Öğrenmesi ile Görüntü Tanıma Sistemi
**Grup 3 — Gömülü Sistemler Projesi**

Hibrit kuantum-klasik sinir ağı (QNN) kullanan, kameradan gerçek zamanlı görüntü tanıma sistemi.

---

## Hızlı Başlangıç

```bash
# 1. Bağımlılıkları kur
pip install -r requirements-dev.txt

# 2. Modeli eğit (MNIST otomatik indirilir)
python src/train.py

# 3. Testleri çalıştır
pytest tests/ -v

# 4. Web arayüzünü başlat
python web/app.py
# → http://localhost:5000
```

## Kamera Modunu Değiştirme

`config.yaml` içinde tek satır:
```yaml
camera:
  mode: mock   # mock | live | image
```
- `mock`  — gerçek kamera gerekmez, MNIST'ten test görüntüsü üretir
- `live`  — bilgisayarın webcam'i
- `image` — klasördeki görüntü dosyalarını sırayla okur

## Model Mimarisi

```
Giriş (28×28) → CNN Encoder → Kuantum Devre (4 qubit) → Sınıflandırıcı → 10 sınıf
```

## Görev Dağılımı

| Kişi | Sorumluluk |
|------|-----------|
| Kişi 1 | `src/quantum_model.py`, `src/train.py`, `src/evaluate.py`, notebook'lar |
| Kişi 2 | `camera/`, `src/predict.py`, kamera testleri |
| Kişi 3 | `web/`, `assets/`, sunum |
