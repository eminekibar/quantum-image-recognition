const frameImg    = document.getElementById('frame-img');
const noFrame     = document.getElementById('no-frame');
const predNumber  = document.getElementById('pred-number');
const predConf    = document.getElementById('pred-confidence');
const predTime    = document.getElementById('pred-time');
const probBars    = document.getElementById('prob-bars');
const statusBar   = document.getElementById('status-bar');
const btnCapture  = document.getElementById('btn-capture');
const btnAuto     = document.getElementById('btn-auto');
const btnStop     = document.getElementById('btn-stop');

let autoInterval = null;

// Olasilik cubuk grafigini olustur
function initBars() {
  probBars.innerHTML = '';
  for (let i = 0; i < 10; i++) {
    const row = document.createElement('div');
    row.className = 'prob-row';
    row.innerHTML = `
      <span class="prob-label">${i}</span>
      <div class="prob-track"><div class="prob-fill" id="bar-${i}" style="width:0%"></div></div>
      <span class="prob-pct" id="pct-${i}">0%</span>`;
    probBars.appendChild(row);
  }
}

async function captureAndPredict() {
  setStatus('Frame aliniyor...');
  try {
    // 1. Frame al
    const fr = await fetch('/frame');
    if (!fr.ok) throw new Error('Frame alinamadi');
    const frData = await fr.json();

    frameImg.src = 'data:image/jpeg;base64,' + frData.image;
    frameImg.style.display = 'block';
    noFrame.style.display = 'none';

    // 2. Tahmin yap
    setStatus('Tahmin yapiliyor...');
    const pr = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: frData.image }),
    });
    if (!pr.ok) throw new Error('Tahmin hatasi');
    const result = await pr.json();

    // 3. Sonuclari goster
    predNumber.textContent = result.prediction;
    predConf.textContent = `Guven: %${result.confidence}`;
    predTime.textContent = `${result.time_ms} ms`;

    result.probabilities.forEach((p, i) => {
      document.getElementById(`bar-${i}`).style.width = p + '%';
      document.getElementById(`pct-${i}`).textContent = p + '%';
    });

    setStatus(`Tahmin: ${result.prediction} | %${result.confidence} guven | ${result.time_ms}ms`);
  } catch (e) {
    setStatus('Hata: ' + e.message);
  }
}

function startAuto() {
  btnAuto.style.display = 'none';
  btnStop.style.display = 'inline-block';
  captureAndPredict();
  autoInterval = setInterval(captureAndPredict, 2000);
}

function stopAuto() {
  clearInterval(autoInterval);
  autoInterval = null;
  btnAuto.style.display = 'inline-block';
  btnStop.style.display = 'none';
  setStatus('Durduruldu.');
}

function setStatus(msg) {
  statusBar.textContent = msg;
}

async function checkStatus() {
  try {
    const r = await fetch('/status');
    const d = await r.json();
    const modelTxt = d.model_loaded ? 'Model hazir' : 'Model henuz egitilmedi (train.py calistirin)';
    setStatus(`${modelTxt} | Kamera: ${d.camera_mode} | Cihaz: ${d.device}`);
  } catch {}
}

btnCapture.addEventListener('click', captureAndPredict);
btnAuto.addEventListener('click', startAuto);
btnStop.addEventListener('click', stopAuto);

initBars();
checkStatus();
