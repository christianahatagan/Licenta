/* ═══════════════════════════════════════════════════════════════════════════
   Mammography Detection App - Frontend Logic (REDESIGNED)
   ═══════════════════════════════════════════════════════════════════════════ */

// ─────────────────────────────────────────────
// GLOBAL VARIABLES
// ─────────────────────────────────────────────

let currentFilename = null;
let currentMethod = null;
let preprocessingSteps = {};
let comparisonChart = null;

// Fixed settings
const FIXED_VIEW = 'cc_mlo';
const FIXED_CONFIDENCE = 0.25;

// ─────────────────────────────────────────────
// DOM ELEMENTS
// ─────────────────────────────────────────────

const imageUpload = document.getElementById('imageUpload');
const methodSelect = document.getElementById('methodSelect');
const uploadBtn = document.getElementById('uploadBtn');
const detectBtn = document.getElementById('detectBtn');
const resetBtn = document.getElementById('resetBtn');
const imageContainer = document.getElementById('imageContainer');
const stepsCard = document.getElementById('stepsCard');
const stepsContainer = document.getElementById('stepsContainer');
const resultsCard = document.getElementById('resultsCard');
const detectionCount = document.getElementById('detectionCount');
const detectionList = document.getElementById('detectionList');
const modelInfo = document.getElementById('modelInfo');
const loadingSpinner = document.getElementById('loadingSpinner');

// ─────────────────────────────────────────────
// EVENT LISTENERS
// ─────────────────────────────────────────────

imageUpload.addEventListener('change', function() {
    if (this.files && this.files[0]) {
        uploadBtn.disabled = false;
        detectBtn.disabled = true;
        stepsCard.style.display = 'none';
        resultsCard.style.display = 'none';
    }
});

uploadBtn.addEventListener('click', uploadAndPreprocess);
detectBtn.addEventListener('click', runDetection);
resetBtn.addEventListener('click', resetApp);
methodSelect.addEventListener('change', updateModelInfo);

// ─────────────────────────────────────────────
// MAIN FUNCTIONS
// ─────────────────────────────────────────────

async function uploadAndPreprocess() {
    const file = imageUpload.files[0];
    if (!file) {
        showAlert('Please select an image first', 'warning');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('method', methodSelect.value);

    showLoading(true);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            currentFilename = data.filename;
            currentMethod = data.method;
            preprocessingSteps = data.steps;

            displayPreprocessingSteps(data.steps);
            detectBtn.disabled = false;
            updateModelInfo();

            showAlert('Image preprocessed successfully!', 'success');
        } else {
            showAlert(data.error || 'Upload failed', 'danger');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showAlert('Upload failed: ' + error.message, 'danger');
    } finally {
        showLoading(false);
    }
}

async function runDetection() {
    if (!currentFilename) {
        showAlert('Please upload an image first', 'warning');
        return;
    }

    showLoading(true);

    const requestData = {
        filename: currentFilename,
        method: methodSelect.value,
        view: FIXED_VIEW,
        confidence: FIXED_CONFIDENCE
    };

    try {
        const response = await fetch('/detect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });

        const data = await response.json();

        if (data.success) {
            displayDetectionResult(data.image_with_boxes);
            displayDetections(data.detections);
            showAlert(`Detection complete! Found ${data.detection_count} object(s)`, 'success');
        } else {
            showAlert(data.error || 'Detection failed', 'danger');
            if (data.available_models) {
                showAlert(data.available_models, 'info');
            }
        }
    } catch (error) {
        console.error('Detection error:', error);
        showAlert('Detection failed: ' + error.message, 'danger');
    } finally {
        showLoading(false);
    }
}

function resetApp() {
    imageUpload.value = '';
    methodSelect.selectedIndex = 0;
    currentFilename = null;
    currentMethod = null;
    preprocessingSteps = {};
    uploadBtn.disabled = true;
    detectBtn.disabled = true;
    stepsCard.style.display = 'none';
    resultsCard.style.display = 'none';
    modelInfo.style.display = 'none';

    imageContainer.innerHTML = `
        <div class="placeholder-image">
            <i class="fas fa-file-medical fa-5x text-muted"></i>
            <p class="text-muted mt-3">Upload an image to begin</p>
        </div>
    `;

    showAlert('Application reset', 'info');
}

// ─────────────────────────────────────────────
// DISPLAY FUNCTIONS
// ─────────────────────────────────────────────

function displayPreprocessingSteps(steps) {
    stepsContainer.innerHTML = '';

    const stepOrder = ['original', 'gaussian', 'nlm', 'artifacts_removed', 'otsu_mask', 
                       'cropped', 'scanlines_removed', 'clahe', 'resized', 'final'];
    
    const stepLabels = {
        'original': 'Original',
        'gaussian': 'Gaussian Blur',
        'nlm': 'NLM Denoising',
        'artifacts_removed': 'Artifacts Removed',
        'otsu_mask': 'Otsu Masking',
        'cropped': 'Cropped',
        'scanlines_removed': 'Scanlines Removed',
        'clahe': 'CLAHE',
        'resized': 'Resized',
        'final': 'Final Result'
    };

    stepOrder.forEach(stepKey => {
        if (steps[stepKey]) {
            const stepDiv = document.createElement('div');
            stepDiv.className = 'col-md-3 step-item';
            stepDiv.innerHTML = `
                <img src="${steps[stepKey]}" alt="${stepLabels[stepKey]}">
                <p>${stepLabels[stepKey]}</p>
            `;
            
            stepDiv.addEventListener('click', () => {
                displayImage(steps[stepKey], stepLabels[stepKey]);
            });
            
            stepsContainer.appendChild(stepDiv);
        }
    });

    stepsCard.style.display = 'block';

    if (steps.final) {
        displayImage(steps.final, 'Preprocessed Image');
    }
}

function displayImage(imageSrc, title) {
    imageContainer.innerHTML = `<img src="${imageSrc}" alt="${title}">`;
    document.getElementById('imageTitle').textContent = title;
}

function displayDetectionResult(imageSrc) {
    displayImage(imageSrc, 'Detection Result');
}

function displayDetections(detections) {
    detectionList.innerHTML = '';

    if (detections.length === 0) {
        detectionList.innerHTML = '<p class="text-muted">No detections found</p>';
    } else {
        detections.forEach((det, index) => {
            const detDiv = document.createElement('div');
            detDiv.className = `detection-item ${det.class_name.toLowerCase()}`;
            
            const [x1, y1, x2, y2] = det.bbox.map(v => Math.round(v));
            const width = x2 - x1;
            const height = y2 - y1;
            
            detDiv.innerHTML = `
                <strong>Detection ${index + 1}</strong>
                <span class="badge bg-${det.class_name === 'Malignant' ? 'danger' : 'success'} float-end">
                    ${(det.confidence * 100).toFixed(1)}%
                </span>
                <div class="mt-2 small">
                    <div><strong>Class:</strong> ${det.class_name}</div>
                    <div><strong>BBox:</strong> [${x1}, ${y1}, ${x2}, ${y2}]</div>
                    <div><strong>Size:</strong> ${width} × ${height} px</div>
                </div>
            `;
            
            detectionList.appendChild(detDiv);
        });
    }

    detectionCount.textContent = detections.length;
    resultsCard.style.display = 'block';
}

function updateModelInfo() {
    const method = methodSelect.value;
    const view = FIXED_VIEW;

    fetch('/metrics')
        .then(response => response.json())
        .then(data => {
            const metrics = data.performance[view]?.[method];
            
            if (metrics) {
                document.getElementById('modelName').textContent = `${view}_${method}.pt`;
                document.getElementById('modelPrecision').textContent = metrics.precision.toFixed(4);
                document.getElementById('modelRecall').textContent = metrics.recall.toFixed(4);
                document.getElementById('modelMap').textContent = metrics.mAP50.toFixed(4);
                modelInfo.style.display = 'block';
            } else {
                modelInfo.style.display = 'none';
            }
        })
        .catch(error => {
            console.error('Error fetching metrics:', error);
        });
}

// ─────────────────────────────────────────────
// COMPARISON PAGE FUNCTIONS
// ─────────────────────────────────────────────

async function loadComparisonData() {
    try {
        const response = await fetch('/metrics');
        const data = await response.json();
        
        populateComparisonTables(data);
        createComparisonChart(data);
    } catch (error) {
        console.error('Error loading comparison data:', error);
    }
}

function populateComparisonTables(data) {
    const views = ['cc', 'mlo', 'cc_mlo'];
    const tableIds = ['ccTable', 'mloTable', 'ccMloTable'];
    
    views.forEach((view, idx) => {
        const tableBody = document.getElementById(tableIds[idx]);
        tableBody.innerHTML = '';
        
        const performance = data.performance[view];
        if (!performance) return;
        
        const sorted = Object.entries(performance)
            .sort((a, b) => b[1].mAP50 - a[1].mAP50);
        
        const mAP50Values = sorted.map(([_, metrics]) => metrics.mAP50);
        const bestMAP = Math.max(...mAP50Values);
        
        sorted.forEach(([method, metrics]) => {
            const row = document.createElement('tr');
            const isBest = metrics.mAP50 === bestMAP;
            
            row.innerHTML = `
                <td>${data.methods[method]}</td>
                <td class="${isBest ? 'best-score' : ''}">${metrics.mAP50.toFixed(4)}</td>
                <td>${metrics.precision.toFixed(4)}</td>
                <td>${metrics.recall.toFixed(4)}</td>
            `;
            
            tableBody.appendChild(row);
        });
    });
}

function createComparisonChart(data) {
    const ctx = document.getElementById('comparisonChart').getContext('2d');
    
    const methods = Object.keys(data.performance.cc_mlo);
    const datasets = [
        {
            label: 'CC',
            data: methods.map(m => data.performance.cc[m]?.mAP50 || 0),
            backgroundColor: 'rgba(214, 51, 132, 0.7)',
            borderColor: 'rgba(214, 51, 132, 1)',
            borderWidth: 2
        },
        {
            label: 'MLO',
            data: methods.map(m => data.performance.mlo[m]?.mAP50 || 0),
            backgroundColor: 'rgba(230, 133, 181, 0.7)',
            borderColor: 'rgba(230, 133, 181, 1)',
            borderWidth: 2
        },
        {
            label: 'CC+MLO',
            data: methods.map(m => data.performance.cc_mlo[m]?.mAP50 || 0),
            backgroundColor: 'rgba(176, 42, 106, 0.7)',
            borderColor: 'rgba(176, 42, 106, 1)',
            borderWidth: 2
        }
    ];
    
    if (comparisonChart) {
        comparisonChart.destroy();
    }
    
    comparisonChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: methods.map(m => data.methods[m]),
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 0.6,
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + (context.parsed.y * 100).toFixed(2) + '%';
                        }
                    }
                }
            }
        }
    });
}

// ─────────────────────────────────────────────
// UTILITY FUNCTIONS
// ─────────────────────────────────────────────

function showLoading(show) {
    loadingSpinner.style.display = show ? 'flex' : 'none';
}

function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    alertDiv.style.top = '20px';
    alertDiv.style.right = '20px';
    alertDiv.style.zIndex = '10000';
    alertDiv.style.minWidth = '300px';
    
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

// ─────────────────────────────────────────────
// INITIALIZATION
// ─────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', function() {
    console.log('🏥 Mammography Detection App - Frontend Loaded (Redesigned)');
    
    updateModelInfo();
    loadComparisonData();
    
    // Reload comparison data when switching to comparison tab
    document.getElementById('comparison-tab').addEventListener('shown.bs.tab', function() {
        loadComparisonData();
    });
});
