const tabBtns = document.querySelectorAll(".tab");
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browseBtn");
const uploadHint = document.getElementById("uploadHint");
const fileSelected = document.getElementById("fileSelected");
const fileName = document.getElementById("fileName");
const clearFile = document.getElementById("clearFile");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultsIdle = document.getElementById("resultsIdle");
const resultsAnalyzing = document.getElementById("resultsAnalyzing");
const resultsOutput = document.getElementById("resultsOutput");
const steps = [
  document.getElementById("step1"),
  document.getElementById("step2"),
  document.getElementById("step3"),
  document.getElementById("step4"),
];
const hintMap = {
  image: "Supported: PNG, JPG, WEBP · Max 10MB",
  video: "Supported: MP4, AVI, MOV · Max 200MB",
  audio: "Supported: WAV, MP3, FLAC · Max 50MB",
};
let currentTab = "image";
let selectedFile = null;

tabBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    tabBtns.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    currentTab = btn.dataset.tab;
    uploadHint.textContent = hintMap[currentTab];
    clearSelection();
  });
});

browseBtn.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("click", (e) => {
  if (e.target !== browseBtn && e.target !== clearFile) fileInput.click();
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragging");
});
["dragleave", "dragend"].forEach((ev) =>
  dropZone.addEventListener(ev, () => dropZone.classList.remove("dragging"))
);
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragging");
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  selectedFile = file;
  fileName.textContent = file.name;
  fileSelected.classList.remove("hidden");
  analyzeBtn.disabled = false;
}

clearFile.addEventListener("click", (e) => {
  e.stopPropagation();
  clearSelection();
});

function clearSelection() {
  selectedFile = null;
  fileInput.value = "";
  fileSelected.classList.add("hidden");
  analyzeBtn.disabled = true;
  showIdle();
}

analyzeBtn.addEventListener("click", () => {
  if (!selectedFile) return;
  startAnalysis();
});

function showIdle() {
  resultsIdle.classList.remove("hidden");
  resultsAnalyzing.classList.add("hidden");
  resultsOutput.classList.add("hidden");

  const gradcamImg = document.getElementById("gradcamImg");
  const xaiPholder = document.getElementById("xaiPlaceholder");
  const xaiBreakdown = document.getElementById("xaiBreakdown");
  if (gradcamImg) gradcamImg.classList.add("hidden");
  if (xaiPholder) xaiPholder.classList.remove("hidden");
  if (xaiBreakdown) xaiBreakdown.innerHTML = "";

  const computationText = document.getElementById("computationText");
  const copyConfirm = document.getElementById("copyConfirm");
  if (computationText) computationText.textContent = "—";
  if (copyConfirm) copyConfirm.classList.add("hidden");
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
    verdict: data.verdict,
    isFake: data.is_fake,
    confidence: formatPercent(data.confidence),
    scores: layers.map((score) => ({
      val: Math.round(score * 100),
      label: formatPercent(score),
    })),
    unified: formatPercent(data.confidence),
    gradcam: data.gradcam_b64 || null,
    xai: data.xai || null,
    // Raw values, unrounded-for-display, kept for the weighted-sum
    // computation block — sourced straight from fuse_scores() so this
    // can never drift out of sync with the backend math.
    rawLayerScores: data.layer_scores,
    rawWeights: data.weights,
    rawThreshold: data.threshold,
    rawConfidence: data.confidence,
    verdictRaw: data.verdict,
  };
}

// ── Weighted-sum verification block ──────────────────────────────────
function buildComputationText(data) {
  const { rawLayerScores: s, rawWeights: w, rawThreshold: t } = data;
  if (!s || !w || t === undefined) return "Computation data unavailable.";

  const m = s.metadata, c = s.content, b = s.binary;
  const wm = m * w.metadata, wc = c * w.content, wb = b * w.binary;
  const sum = wm + wc + wb;

  const fmt = (n) => n.toFixed(4);
  const pad = (label) => label.padEnd(15, " ");

  return (
    `${pad("L1 (Metadata)")}${fmt(m)} × ${w.metadata.toFixed(2)} = ${fmt(wm)}\n` +
    `${pad("L2 (Content)")}${fmt(c)} × ${w.content.toFixed(2)} = ${fmt(wc)}\n` +
    `${pad("L3 (Binary)")}${fmt(b)} × ${w.binary.toFixed(2)} = ${fmt(wb)}\n` +
    `${"-".repeat(38)}\n` +
    `${pad("Weighted Sum")}= ${fmt(sum)}\n` +
    `${pad("Threshold")}= ${t.toFixed(2)}  (confidence >= ${t.toFixed(2)} -> DEEPFAKE DETECTED, else LIKELY AUTHENTIC)\n\n` +
    `Verdict: ${data.verdictRaw} (${(sum * 100).toFixed(2)}%)`
  );
}

// Tab-separated single row — pastes directly into Excel / Sheets / a Word table.
function buildComputationRow(data) {
  const { rawLayerScores: s, rawWeights: w, rawThreshold: t } = data;
  if (!s || !w || t === undefined) return "";

  const m = s.metadata, c = s.content, b = s.binary;
  const wm = m * w.metadata, wc = c * w.content, wb = b * w.binary;
  const sum = wm + wc + wb;
  const fmt = (n) => n.toFixed(4);

  return [
    fmt(m), w.metadata.toFixed(2), fmt(wm),
    fmt(c), w.content.toFixed(2), fmt(wc),
    fmt(b), w.binary.toFixed(2), fmt(wb),
    fmt(sum), t.toFixed(2), data.verdictRaw,
  ].join("\t");
}

async function startAnalysis() {
  resultsIdle.classList.add("hidden");
  resultsAnalyzing.classList.remove("hidden");
  resultsOutput.classList.add("hidden");
  steps.forEach((s) => s.classList.remove("active", "done"));
  const delays = [0, 1200, 2400, 3400];
  const dones = [1000, 2200, 3200, 4200];
  steps.forEach((step, i) => {
    setTimeout(() => step.classList.add("active"), delays[i]);
    setTimeout(() => {
      step.classList.remove("active");
      step.classList.add("done");
    }, dones[i]);
  });

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("currentTab", currentTab);

  const toggleInputs = document.querySelectorAll(
    '.layer-toggle input[type="checkbox"]'
  );
  formData.append("layer_metadata", toggleInputs[0]?.checked ? "1" : "0");
  formData.append("layer_content", toggleInputs[1]?.checked ? "1" : "0");
  formData.append("layer_binary", toggleInputs[2]?.checked ? "1" : "0");
  formData.append("layer_xai", toggleInputs[3]?.checked ? "1" : "0");

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Analysis failed");
    }
    resultsAnalyzing.classList.add("hidden");
    showResults(mapAnalysisResponse(data));
  } catch (err) {
    resultsAnalyzing.classList.add("hidden");
    showIdle();
    alert(err.message || "Analysis failed. Please try again.");
  }
}

function renderFeatureBreakdown(xai) {
  const container = document.getElementById("xaiBreakdown");
  if (!container) return;
  container.innerHTML = "";

  // L1 / L3 — structured {label, contribution, direction} rows, scored
  // toward the verdict.
  const scoredSections = [
    { title: "L1 — Metadata Forensics", features: xai.metadata_features },
    { title: "L3 — Binary File Structure", features: xai.binary_features },
  ];

  scoredSections.forEach((section) => {
    if (!section.features || Object.keys(section.features).length === 0) return;

    const block = document.createElement("div");
    block.className = "xai-block";

    const heading = document.createElement("div");
    heading.className = "xai-block-title";
    heading.textContent = section.title;
    block.appendChild(heading);

    Object.values(section.features).forEach((feat) => {
      const row = document.createElement("div");
      row.className = `xai-row xai-${feat.direction}`;

      const label = document.createElement("span");
      label.className = "xai-row-label";
      label.textContent = feat.label;

      const contrib = document.createElement("span");
      contrib.className = "xai-row-contrib";
      const val = feat.contribution;
      const sign = val > 0 ? "+" : "";
      contrib.textContent = `${sign}${(val * 100).toFixed(0)}%`;

      row.appendChild(label);
      row.appendChild(contrib);
      block.appendChild(row);
    });

    container.appendChild(block);
  });

  // L2 — raw model diagnostics (currently only populated for audio).
  // Different shape than L1/L3 (no contribution/direction — these are
  // informational, not individually scored), so rendered as plain rows.
  // This exists specifically to answer "why did L2 say what it said" —
  // e.g. whether a short clip got zero-padded, or the model silently
  // fell back to neutral 0.5 on an error.
  const cf = xai.content_features;
  if (cf && Object.keys(cf).length > 0) {
    const block = document.createElement("div");
    block.className = "xai-block";

    const heading = document.createElement("div");
    heading.className = "xai-block-title";
    heading.textContent = "L2 — Content Analysis (raw model output)";
    block.appendChild(heading);

    if (cf.error) {
      const row = document.createElement("div");
      row.className = "xai-row xai-suspicious";
      row.textContent = `Model error — fell back to neutral 0.5: ${cf.error}`;
      block.appendChild(row);
    } else {
      const rows = [];
      if (cf.raw_fake_prob !== undefined)
        rows.push(`Raw fake probability: ${(cf.raw_fake_prob * 100).toFixed(1)}%`);
      if (cf.raw_real_prob !== undefined)
        rows.push(`Raw real probability: ${(cf.raw_real_prob * 100).toFixed(1)}%`);
      if (cf.original_duration_sec !== undefined)
        rows.push(`Clip duration: ${cf.original_duration_sec}s`);
      if (cf.analyzed_window_sec !== undefined)
        rows.push(`Model analysis window: ${cf.analyzed_window_sec}s`);
      if (cf.truncated === true) {
        rows.push(
          `Clip longer than the window — truncated to the first ${cf.analyzed_window_sec}s`
        );
      } else if (
        cf.truncated === false &&
        cf.original_duration_sec !== undefined &&
        cf.analyzed_window_sec !== undefined &&
        cf.original_duration_sec < cf.analyzed_window_sec
      ) {
        rows.push(
          `Clip shorter than the window — zero-padded to fill ${cf.analyzed_window_sec}s (short clips can skew the score)`
        );
      } else if (cf.truncated === false) {
        rows.push("Clip matched the analysis window — no padding or truncation");
      }
      rows.forEach((text) => {
        const row = document.createElement("div");
        row.className = "xai-row xai-plain";
        row.textContent = text;
        block.appendChild(row);
      });
    }

    container.appendChild(block);
  }
}

function showResults(data) {
  resultsOutput.classList.remove("hidden");
  const banner = document.getElementById("verdictBanner");
  banner.classList.remove("fake", "real");
  banner.classList.add(data.isFake ? "fake" : "real");
  document.getElementById("verdictText").textContent = data.verdict;
  document.getElementById(
    "verdictConf"
  ).textContent = `Confidence: ${data.confidence}`;
  data.scores.forEach((s, i) => {
    const bar = document.getElementById(`scoreBar${i + 1}`);
    const val = document.getElementById(`scoreVal${i + 1}`);
    setTimeout(() => {
      bar.style.setProperty("--w", `${s.val}%`);
      val.textContent = s.label;
    }, i * 150);
  });
  document.getElementById("unifiedVal").textContent = data.unified;

  // Grad-CAM
  const gradcamImg = document.getElementById("gradcamImg");
  const xaiPholder = document.getElementById("xaiPlaceholder");
  if (data.gradcam) {
    gradcamImg.src = "data:image/jpeg;base64," + data.gradcam;
    gradcamImg.classList.remove("hidden");
    xaiPholder.classList.add("hidden");
  } else {
    gradcamImg.classList.add("hidden");
    xaiPholder.classList.remove("hidden");
  }

  // Feature Contribution Breakdown
  if (data.xai) {
    renderFeatureBreakdown(data.xai);
  }

  // Weighted-sum verification block
  const computationText = document.getElementById("computationText");
  if (computationText) computationText.textContent = buildComputationText(data);
  const copyConfirm = document.getElementById("copyConfirm");
  if (copyConfirm) copyConfirm.classList.add("hidden");
  lastResultData = data;
}

// Kept outside showResults so the copy button always has the most recent
// result available, without re-wiring a listener on every analysis run.
let lastResultData = null;
document.getElementById("copyComputationBtn")?.addEventListener("click", async () => {
  if (!lastResultData) return;
  const row = buildComputationRow(lastResultData);
  const copyConfirm = document.getElementById("copyConfirm");
  try {
    await navigator.clipboard.writeText(row);
  } catch (err) {
    // Clipboard API can fail on non-HTTPS/non-localhost contexts — fall back.
    const textarea = document.createElement("textarea");
    textarea.value = row;
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    document.body.removeChild(textarea);
  }
  if (copyConfirm) {
    copyConfirm.classList.remove("hidden");
    setTimeout(() => copyConfirm.classList.add("hidden"), 2000);
  }
});

document.getElementById("resetBtn").addEventListener("click", () => {
  clearSelection();
});

showIdle();
