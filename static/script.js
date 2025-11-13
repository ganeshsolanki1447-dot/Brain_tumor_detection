const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const imagePreview = document.getElementById('imagePreview');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const resultsSection = document.getElementById('resultsSection');
const loading = document.getElementById('loading');

let selectedFile = null;

// Upload area interactions
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#764ba2';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = '#667eea';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#667eea';
    handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) {
        alert('Please select a valid image file.');
        return;
    }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        previewSection.style.display = 'block';
        resultsSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

analyzeBtn.addEventListener('click', analyzeImage);
clearBtn.addEventListener('click', clearImage);

async function analyzeImage() {
    if (!selectedFile) {
        alert('Please select an image first.');
        return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    loading.style.display = 'block';
    resultsSection.style.display = 'none';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }

        displayResults(data);
    } catch (error) {
        alert('Analysis failed: ' + error.message);
    } finally {
        loading.style.display = 'none';
    }
}

function displayResults(data) {
    document.getElementById('tumorType').textContent = data.tumor_info.name;
    document.getElementById('tumorDescription').textContent = data.tumor_info.description;
    document.getElementById('severity').textContent = data.tumor_info.severity;
    document.getElementById('action').textContent = data.tumor_info.action;

    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceText = document.getElementById('confidenceText');

    confidenceBar.style.width = data.confidence + '%';
    confidenceText.textContent = data.confidence.toFixed(1) + '%';

    // Display all probabilities
    const probabilitiesDiv = document.getElementById('probabilities');
    probabilitiesDiv.innerHTML = '<h4>All Predictions:</h4>';

    Object.entries(data.all_probabilities).forEach(([tumor, prob]) => {
        const item = document.createElement('div');
        item.className = 'probability-item';
        item.innerHTML = `
            <span>${tumor}</span>
            <span>${prob.toFixed(1)}%</span>
        `;
        probabilitiesDiv.appendChild(item);
    });

    resultsSection.style.display = 'block';
}

function clearImage() {
    selectedFile = null;
    previewSection.style.display = 'none';
    resultsSection.style.display = 'none';
    fileInput.value = '';
}