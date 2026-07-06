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
    // Reset Grad-CAM back to placeholder
    const gradcamImg  = document.getElementById('gradcamImg');
    const xaiPholder  = document.getElementById('xaiPlaceholder');
    if (gradcamImg) gradcamImg.classList.add('hidden');
    if (xaiPholder) xaiPholder.classList.remove('hidden');
}

function formatPercent(score) {
    return `${(score * 100).toFixed(1)}%`;
}

function mapAnalysisResponse(data) {
    const layers = [
        data.layer_scores.metadata,
        data.layer_scores.content,
        data.layer_scores.binary,
    ];
    return {
        verdict:  data.verdict,
        isFake:   data.is_fake,
        confidence: formatPercent(data.confidence),
        scores: layers.map(score => ({
            val:   Math.round(score * 100),
            label: formatPercent(score),
        })),
        unified:  formatPercent(data.confidence),
        gradcam:  data.gradcam_b64 || null,
    };
}

async function startAnalysis() {
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

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('currentTab', currentTab);

    // Send toggle states so backend only runs enabled layers
    const toggleInputs = document.querySelectorAll('.layer-toggle input[type="checkbox"]');
    formData.append('layer_metadata', toggleInputs[0]?.checked ? '1' : '0');
    formData.append('layer_content',  toggleInputs[1]?.checked ? '1' : '0');
    formData.append('layer_binary',   toggleInputs[2]?.checked ? '1' : '0');
    formData.append('layer_xai',      toggleInputs[3]?.checked ? '1' : '0');

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData,
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Analysis failed');
        }
        resultsAnalyzing.classList.add('hidden');
        showResults(mapAnalysisResponse(data));
    } catch (err) {
        resultsAnalyzing.classList.add('hidden');
        showIdle();
        alert(err.message || 'Analysis failed. Please try again.');
    }
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

    // Render Grad-CAM if available, otherwise show placeholder
    const gradcamImg = document.getElementById('gradcamImg');
    const xaiPholder = document.getElementById('xaiPlaceholder');
    if (data.gradcam) {
        gradcamImg.src = 'data:image/jpeg;base64,' + data.gradcam;
        gradcamImg.classList.remove('hidden');
        xaiPholder.classList.add('hidden');
    } else {
        gradcamImg.classList.add('hidden');
        xaiPholder.classList.remove('hidden');
    }
}

document.getElementById('resetBtn').addEventListener('click', () => {
    clearSelection();
});

showIdle();
