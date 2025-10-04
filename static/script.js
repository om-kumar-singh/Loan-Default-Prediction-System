function switchTab(tabName) {
    const tabs = document.querySelectorAll('.tab-content');
    const buttons = document.querySelectorAll('.tab-button');

    tabs.forEach(tab => tab.classList.remove('active'));
    buttons.forEach(btn => btn.classList.remove('active'));

    document.getElementById(`${tabName}-tab`).classList.add('active');
    event.target.classList.add('active');

    if (tabName === 'comparison') {
        loadModelComparison();
    }
}

document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = {
        age: document.getElementById('age').value,
        income: document.getElementById('income').value,
        credit_score: document.getElementById('credit_score').value,
        loan_amount: document.getElementById('loan_amount').value,
        employment_type: document.getElementById('employment_type').value,
        marital_status: document.getElementById('marital_status').value,
        education: document.getElementById('education').value
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        const result = await response.json();

        if (response.ok) {
            displayResult(result);
        } else {
            showError(result.error || 'An error occurred');
        }
    } catch (error) {
        showError('Failed to connect to the server');
    }
});

function displayResult(result) {
    const resultContainer = document.getElementById('result-container');
    const errorContainer = document.getElementById('error-container');

    errorContainer.classList.add('hidden');

    document.getElementById('prediction-result').textContent = result.prediction;
    document.getElementById('risk-level').textContent = result.risk_level;
    document.getElementById('confidence').textContent = `${result.confidence}%`;

    const predictionElement = document.getElementById('prediction-result');
    const riskElement = document.getElementById('risk-level');

    if (result.prediction === 'Default') {
        predictionElement.style.color = '#ef4444';
        riskElement.style.color = '#ef4444';
    } else {
        predictionElement.style.color = '#10b981';
        riskElement.style.color = '#10b981';
    }

    const confidenceBar = document.getElementById('confidence-bar');
    confidenceBar.style.width = `${result.confidence}%`;

    resultContainer.classList.remove('hidden');
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showError(message) {
    const errorContainer = document.getElementById('error-container');
    const resultContainer = document.getElementById('result-container');

    resultContainer.classList.add('hidden');

    errorContainer.textContent = `Error: ${message}`;
    errorContainer.classList.remove('hidden');
}

document.getElementById('csv-file').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        document.querySelector('.upload-text').textContent = file.name;
        document.getElementById('upload-btn').classList.remove('hidden');
    }
});

document.getElementById('upload-btn').addEventListener('click', async () => {
    const fileInput = document.getElementById('csv-file');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select a CSV file');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict_batch', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            displayBatchResults(result);
        } else {
            alert(`Error: ${result.error}`);
        }
    } catch (error) {
        alert('Failed to process the file');
    }
});

function displayBatchResults(result) {
    document.getElementById('total-records').textContent = result.total;
    document.getElementById('default-count').textContent = result.defaults;
    document.getElementById('no-default-count').textContent = result.no_defaults;

    const tbody = document.getElementById('batch-tbody');
    tbody.innerHTML = '';

    result.results.forEach(row => {
        const tr = document.createElement('tr');

        const predictionColor = row.prediction === 'Default' ? '#ef4444' : '#10b981';

        tr.innerHTML = `
            <td>${row.age}</td>
            <td>$${parseInt(row.income).toLocaleString()}</td>
            <td>${row.credit_score}</td>
            <td>$${parseInt(row.loan_amount).toLocaleString()}</td>
            <td>${row.employment_type}</td>
            <td style="color: ${predictionColor}; font-weight: 600;">${row.prediction}</td>
            <td>${row.confidence}%</td>
        `;

        tbody.appendChild(tr);
    });

    document.getElementById('batch-result').classList.remove('hidden');
    document.getElementById('batch-result').scrollIntoView({ behavior: 'smooth' });
}

async function loadModelComparison() {
    try {
        const response = await fetch('/model_comparison');
        const models = await response.json();

        const chartContainer = document.getElementById('comparison-chart');
        chartContainer.innerHTML = '';

        Object.keys(models).forEach(modelName => {
            const modelData = models[modelName];

            const modelBar = document.createElement('div');
            modelBar.className = 'model-bar';

            modelBar.innerHTML = `
                <div class="model-name">${modelName}</div>
                <div class="bar-container">
                    <div class="bar-fill" style="width: ${modelData.test_accuracy}%">
                        ${modelData.test_accuracy}%
                    </div>
                </div>
                <div class="bar-label">
                    <span>Train: ${modelData.train_accuracy}%</span>
                    <span>Test: ${modelData.test_accuracy}%</span>
                </div>
            `;

            chartContainer.appendChild(modelBar);
        });
    } catch (error) {
        console.error('Failed to load model comparison:', error);
    }
}
