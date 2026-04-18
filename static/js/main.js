const tabBtns      = document.querySelectorAll('.tab');
const dropZone     = document.getElementById('dropZone');
const fileInput    = document.getElementById('fileInput');
const browseBtn    = document.getElementById('browseBtn');
const uploadHint   = document.getElementById('uploadHint');
const fileSelected = document.getElementById('fileSelected');
const fileName     = document.getElementById('fileName');
const clearFile    = document.getElementById('clearFile');
const analyzeBtn   = document.getElementById('analyzeBtn');
const resultsIdle      = document.getElementById('resultsIdle');
const resultsAnalyzing = document.getElementById('resultsAnalyzing');
const resultsOutput    = document.getElementById('resultsOutput');
const steps = [
    document.getElementById('step1'),
    document.getElementById('step2'),
    document.getElementById('step3'),
    document.getElementById('step4'),
];
const hintMap = {
    image: 'Supported: PNG, JPG, WEBP · Max 10MB',
    video: 'Supported: MP4, AVI, MOV · Max 200MB',
    audio: 'Supported: WAV, MP3, FLAC · Max 50MB',
};
let currentTab = 'image';
let selectedFile = null;

tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        tabBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentTab = btn.dataset.tab;
        uploadHint.textContent = hintMap[currentTab];
        clearSelection();
    });
});

browseBtn.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('click', (e) => {
    if (e.target !== browseBtn && e.target !== clearFile) fileInput.click();
});
fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
});
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragging');
});
['dragleave', 'dragend'].forEach(ev =>
    dropZone.addEventListener(ev, () => dropZone.classList.remove('dragging'))
);
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragging');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
    selectedFile = file;
    fileName.textContent = file.name;
    fileSelected.classList.remove('hidden');
    analyzeBtn.disabled = false;
}

clearFile.addEventListener('click', (e) => {
    e.stopPropagation();
    clearSelection();
});

function clearSelection() {
    selectedFile = null;
    fileInput.value = '';
    fileSelected.classList.add('hidden');
    analyzeBtn.disabled = true;
    showIdle();
}

analyzeBtn.addEventListener('click', () => {
    if (!selectedFile) return;
    startAnalysis();
});

function showIdle() {
    resultsIdle.classList.remove('hidden');
    resultsAnalyzing.classList.add('hidden');
    resultsOutput.classList.add('hidden');
}

function startAnalysis() {
    resultsIdle.classList.add('hidden');
    resultsAnalyzing.classList.remove('hidden');
    resultsOutput.classList.add('hidden');
    steps.forEach(s => s.classList.remove('active', 'done'));
    const delays = [0, 1200, 2400, 3400];
    const dones  = [1000, 2200, 3200, 4200];
    steps.forEach((step, i) => {
        setTimeout(() => step.classList.add('active'), delays[i]);
        setTimeout(() => {
            step.classList.remove('active');
            step.classList.add('done');
        }, dones[i]);
    });
    setTimeout(() => {
        resultsAnalyzing.classList.add('hidden');
        showResults({
            verdict: 'DEEPFAKE DETECTED',
            isFake: true,
            confidence: '94.7%',
            scores: [
                { val: 88, label: '88.2%' },
                { val: 96, label: '96.4%' },
                { val: 91, label: '91.3%' },
            ],
            unified: '94.7%'
        });
    }, 5000);
}

function showResults(data) {
    resultsOutput.classList.remove('hidden');
    const banner = document.getElementById('verdictBanner');
    banner.classList.remove('fake', 'real');
    banner.classList.add(data.isFake ? 'fake' : 'real');
    document.getElementById('verdictText').textContent = data.verdict;
    document.getElementById('verdictConf').textContent = `Confidence: ${data.confidence}`;
    data.scores.forEach((s, i) => {
        const bar = document.getElementById(`scoreBar${i+1}`);
        const val = document.getElementById(`scoreVal${i+1}`);
        setTimeout(() => {
            bar.style.setProperty('--w', `${s.val}%`);
            val.textContent = s.label;
        }, i * 150);
    });
    document.getElementById('unifiedVal').textContent = data.unified;
}

document.getElementById('resetBtn').addEventListener('click', () => {
    clearSelection();
});

showIdle();
