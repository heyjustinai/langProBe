// Global state
let evaluationData = [];
let currentTab = 'comparison';
let charts = {};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeFileUpload();
    loadSampleData();
});

// Tab management
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all nav tabs
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName).classList.add('active');
    
    // Find and activate the corresponding nav tab
    const navTabs = document.querySelectorAll('.nav-tab');
    const tabNames = ['comparison', 'individual', 'upload'];
    const tabIndex = tabNames.indexOf(tabName);
    if (tabIndex !== -1 && navTabs[tabIndex]) {
        navTabs[tabIndex].classList.add('active');
    }
    
    currentTab = tabName;
    
    // Refresh content based on tab
    if (tabName === 'comparison') {
        updateComparison();
    } else if (tabName === 'individual') {
        updateIndividualResults();
    }
}

// File upload functionality
function initializeFileUpload() {
    const fileUpload = document.getElementById('fileUpload');
    const fileInput = document.getElementById('fileInput');
    
    fileUpload.addEventListener('click', () => fileInput.click());
    
    fileUpload.addEventListener('dragover', (e) => {
        e.preventDefault();
        fileUpload.classList.add('dragover');
    });
    
    fileUpload.addEventListener('dragleave', () => {
        fileUpload.classList.remove('dragover');
    });
    
    fileUpload.addEventListener('drop', (e) => {
        e.preventDefault();
        fileUpload.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });
    
    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });
}

// Handle file uploads
function handleFiles(files) {
    const uploadStatus = document.getElementById('uploadStatus');
    uploadStatus.innerHTML = '<div class="loading">Processing files...</div>';
    
    Array.from(files).forEach(file => {
        if (file.type === 'application/json') {
            const reader = new FileReader();
            reader.onload = function(e) {
                try {
                    const data = JSON.parse(e.target.result);
                    processEvaluationData(data, file.name);
                    updateLoadedFilesList();
                    updateAllViews();
                    uploadStatus.innerHTML = '<div style="color: green;">Files loaded successfully!</div>';
                } catch (error) {
                    console.error('Error parsing JSON:', error);
                    uploadStatus.innerHTML = '<div class="error">Error parsing JSON file: ' + file.name + '</div>';
                }
            };
            reader.readAsText(file);
        }
    });
}

// Process evaluation data from different file types
function processEvaluationData(data, filename) {
    let processedData = {
        filename: filename,
        timestamp: new Date().toISOString(),
        type: 'unknown'
    };
    
    // Determine data type and extract relevant information
    if (data.metadata && data.individual_results) {
        // Individual results file
        processedData.type = 'individual_results';
        processedData.benchmark = data.metadata.benchmark;
        processedData.program = data.metadata.program;
        processedData.optimizer = data.metadata.optimizer;
        processedData.suffix = data.metadata.suffix;
        processedData.total_examples = data.metadata.total_examples;
        processedData.average_score = data.metadata.average_score;
        processedData.individual_results = data.individual_results;
        processedData.timestamp = data.metadata.timestamp;
    } else if (data.signature && data.signature.instructions) {
        // Prompt configuration file
        processedData.type = 'prompt_config';
        processedData.instructions = data.signature.instructions;
        processedData.fields = data.signature.fields;
    } else if (Array.isArray(data)) {
        // CSV-like data converted to JSON
        processedData.type = 'csv_data';
        processedData.data = data;
    }
    
    // Add to global data store
    evaluationData.push(processedData);
}

// Load sample data (the files provided by the user)
async function loadSampleData() {
    try {
        // Try to load data from the server API
        const response = await fetch('/api/scan-results');
        if (response.ok) {
            const scanResults = await response.json();
            await loadDataFromScanResults(scanResults);
        } else {
            throw new Error('Server API not available');
        }
    } catch (error) {
        console.log('Server API not available, using sample data:', error.message);
        // Fallback to sample data if server is not running
        createSampleData();
    }
    updateAllViews();
}

// Load data from scan results
async function loadDataFromScanResults(scanResults) {
    for (const versionDir of scanResults.directories.slice(0, 3)) { // Load latest 3 versions
        for (const benchmark of versionDir.benchmarks) {
            for (const file of benchmark.files) {
                if (file.type === 'individual_results') {
                    try {
                        const response = await fetch(`/api/load-file?path=${encodeURIComponent(file.path)}`);
                        if (response.ok) {
                            const data = await response.json();
                            processEvaluationData(data, file.name);
                        }
                    } catch (error) {
                        console.error(`Error loading file ${file.name}:`, error);
                    }
                }
            }
        }
    }
}

// Create sample data for demonstration
function createSampleData() {
    // Sample data based on the provided files
    const sampleData1 = {
        filename: 'HeartDiseaseBench_Predict_baseline_individual_results_extracted_default.json',
        type: 'individual_results',
        benchmark: 'HeartDiseaseBench',
        program: 'Predict',
        optimizer: 'baseline',
        suffix: 'extracted_default',
        total_examples: 100,
        average_score: 0.32,
        timestamp: '2025-08-22T17:48:44.712172',
        individual_results: generateSampleResults(100, 0.32, 'extracted_default', 'HeartDiseaseBench')
    };
    
    const sampleData2 = {
        filename: 'HeartDiseaseBench_Predict_baseline_individual_results_meta_quick_improve.json',
        type: 'individual_results',
        benchmark: 'HeartDiseaseBench',
        program: 'Predict',
        optimizer: 'baseline',
        suffix: 'meta_quick_improve',
        total_examples: 100,
        average_score: 0.38,
        timestamp: '2025-08-22T17:48:45.266871',
        individual_results: generateSampleResults(100, 0.38, 'meta_quick_improve', 'HeartDiseaseBench')
    };
    
    const sampleData3 = {
        filename: 'JudgeBench_Predict_baseline_individual_results_extracted_default.json',
        type: 'individual_results',
        benchmark: 'JudgeBench',
        program: 'Predict',
        optimizer: 'baseline',
        suffix: 'extracted_default',
        total_examples: 100,
        average_score: 0.47,
        timestamp: '2025-08-22T17:50:15.608034',
        individual_results: generateSampleResults(100, 0.47, 'extracted_default', 'JudgeBench')
    };
    
    const sampleData4 = {
        filename: 'JudgeBench_Predict_baseline_individual_results_meta_quick_improve.json',
        type: 'individual_results',
        benchmark: 'JudgeBench',
        program: 'Predict',
        optimizer: 'baseline',
        suffix: 'meta_quick_improve',
        total_examples: 100,
        average_score: 0.53,
        timestamp: '2025-08-22T17:50:16.608034',
        individual_results: generateSampleResults(100, 0.53, 'meta_quick_improve', 'JudgeBench')
    };
    
    evaluationData.push(sampleData1, sampleData2, sampleData3, sampleData4);
}

// Generate sample individual results
function generateSampleResults(count, averageScore, suffix, benchmark) {
    const results = [];
    const correctCount = Math.round(count * averageScore);
    
    for (let i = 0; i < count; i++) {
        const isCorrect = i < correctCount;
        const score = isCorrect ? 1.0 : 0.0;
        
        let sampleResult;
        
        if (benchmark === 'JudgeBench') {
            sampleResult = {
                example_id: i,
                inputs: generateSampleJudgeBenchInput(),
                golden_answer: { 
                    label: Math.random() > 0.5 ? "A>B" : "B>A",
                    pair_id: `sample-${i}`,
                    source: "sample-data"
                },
                prediction: { answer: Math.random() > 0.5 ? "A>B" : "B>A" },
                score: score,
                benchmark: 'JudgeBench',
                program: 'Predict',
                optimizer: 'baseline',
                timestamp: new Date().toISOString()
            };
        } else {
            sampleResult = {
                example_id: i,
                inputs: generateSampleHeartDiseaseInput(),
                golden_answer: { answer: Math.random() > 0.5 ? "yes" : "no" },
                prediction: { answer: Math.random() > 0.5 ? "yes" : "no" },
                score: score,
                benchmark: 'HeartDiseaseBench',
                program: 'Predict',
                optimizer: 'baseline',
                timestamp: new Date().toISOString()
            };
        }
        
        results.push(sampleResult);
    }
    
    return results;
}

// Generate sample HeartDisease input data
function generateSampleHeartDiseaseInput() {
    const ages = [40, 45, 50, 55, 60, 65, 70];
    const sexes = ['male', 'female'];
    const chestPains = ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'];
    
    const age = ages[Math.floor(Math.random() * ages.length)];
    const sex = sexes[Math.floor(Math.random() * sexes.length)];
    const cp = chestPains[Math.floor(Math.random() * chestPains.length)];
    
    return `Example({'age': '${age}', 'sex': '${sex}', 'cp': '${cp}', 'trestbps': '${120 + Math.floor(Math.random() * 60)}', 'chol': '${200 + Math.floor(Math.random() * 100)}', 'fbs': '${Math.floor(Math.random() * 2)}', 'restecg': 'normal', 'thalach': '${120 + Math.floor(Math.random() * 80)}', 'exang': '${Math.random() > 0.5 ? 'yes' : 'no'}', 'oldpeak': '${(Math.random() * 3).toFixed(1)}', 'slope': 'upsloping', 'ca': '${Math.floor(Math.random() * 4)}', 'thal': 'normal'})`;
}

// Generate sample JudgeBench input data
function generateSampleJudgeBenchInput() {
    const questionTypes = [
        "As of 2019, about what percentage of Americans say it is very important to have free media in our country without government/state censorship?",
        "In this question, assume each person either always tells the truth or always lies.",
        "What is the most likely explanation for the following observation?",
        "Which of the following statements is most accurate?"
    ];
    
    const question = questionTypes[Math.floor(Math.random() * questionTypes.length)];
    const responseA = "Response A: This is a sample response that provides one perspective on the question...";
    const responseB = "Response B: This is an alternative response that offers a different viewpoint or approach...";
    
    return `Example({'question': "${question}", 'response_A': "${responseA}", 'response_B': "${responseB}"}) (input_keys={'response_A', 'question', 'response_B'})`;
}

// Update all views
function updateAllViews() {
    updateComparison();
    updateIndividualResults();
    updateLoadedFilesList();
}



// Update individual results tab
function updateIndividualResults() {
    populateFilters();
    filterResults();
}

// Populate filter dropdowns
function populateFilters() {
    const individualResults = evaluationData.filter(d => d.type === 'individual_results');
    
    // Dataset filter
    const datasets = [...new Set(individualResults.map(d => d.benchmark))];
    const datasetSelect = document.getElementById('datasetSelect');
    datasetSelect.innerHTML = '<option value="">All Datasets</option>';
    datasets.forEach(dataset => {
        datasetSelect.innerHTML += `<option value="${dataset}">${dataset}</option>`;
    });
    
    // Optimizer filter
    const optimizers = [...new Set(individualResults.map(d => d.suffix))];
    const optimizerSelect = document.getElementById('optimizerSelect');
    optimizerSelect.innerHTML = '<option value="">All Optimizers</option>';
    optimizers.forEach(optimizer => {
        optimizerSelect.innerHTML += `<option value="${optimizer}">${optimizer}</option>`;
    });
}

// Filter and display results
function filterResults() {
    const datasetFilter = document.getElementById('datasetSelect').value;
    const optimizerFilter = document.getElementById('optimizerSelect').value;
    const scoreFilter = document.getElementById('scoreFilter').value;
    const searchTerm = document.getElementById('searchInput').value.toLowerCase();
    
    const individualResults = evaluationData.filter(d => d.type === 'individual_results');
    let allResults = [];
    
    individualResults.forEach(dataset => {
        if (datasetFilter && dataset.benchmark !== datasetFilter) return;
        if (optimizerFilter && dataset.suffix !== optimizerFilter) return;
        
        (dataset.individual_results || []).forEach(result => {
            if (scoreFilter && result.score.toString() !== scoreFilter) return;
            if (searchTerm && !result.inputs.toLowerCase().includes(searchTerm)) return;
            
            allResults.push({
                ...result,
                dataset: dataset.benchmark,
                optimizer: dataset.suffix
            });
        });
    });
    
    displayResults(allResults);
}

// Display filtered results in table
function displayResults(results) {
    const tbody = document.getElementById('resultsTableBody');
    
    if (results.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7">No results match the current filters</td></tr>';
        return;
    }
    
    tbody.innerHTML = results.slice(0, 100).map(result => {
        const scoreClass = result.score === 1.0 ? 'score-correct' : 'score-incorrect';
        const inputSummary = extractInputSummary(result.inputs);
        
        // Handle different golden answer formats
        const goldenAnswer = result.golden_answer.answer || result.golden_answer.label || 'N/A';
        const prediction = result.prediction.answer || 'N/A';
        
        return `
            <tr>
                <td>${result.example_id}</td>
                <td>${result.dataset}</td>
                <td>${result.optimizer}</td>
                <td class="${scoreClass}">${result.score}</td>
                <td>${goldenAnswer}</td>
                <td>${prediction}</td>
                <td>${inputSummary}</td>
            </tr>
        `;
    }).join('');
    
    if (results.length > 100) {
        tbody.innerHTML += `<tr><td colspan="7"><em>Showing first 100 of ${results.length} results</em></td></tr>`;
    }
}

// Extract summary from input string
function extractInputSummary(inputString) {
    try {
        // For HeartDisease - extract medical information
        const ageMatch = inputString.match(/'age':\s*'(\d+)'/);
        const sexMatch = inputString.match(/'sex':\s*'(\w+)'/);
        const cpMatch = inputString.match(/'cp':\s*'([^']+)'/);
        
        if (ageMatch && sexMatch) {
            const age = ageMatch[1];
            const sex = sexMatch[1];
            const cp = cpMatch ? cpMatch[1] : '?';
            return `Age: ${age}, Sex: ${sex}, CP: ${cp}`;
        }
        
        // For JudgeBench - extract question type or topic
        const questionMatch = inputString.match(/'question':\s*"([^"]{0,100})/);
        if (questionMatch) {
            let question = questionMatch[1];
            if (question.length > 80) {
                question = question.substring(0, 80) + '...';
            }
            return `Question: ${question}`;
        }
        
        // Fallback to truncated input
        return inputString.substring(0, 80) + '...';
    } catch (error) {
        return inputString.substring(0, 50) + '...';
    }
}

// Update comparison tab
function updateComparison() {
    updateComparisonGrid();
}

// Update comparison grid
function updateComparisonGrid() {
    const comparisonGrid = document.getElementById('comparisonGrid');
    const individualResults = evaluationData.filter(d => d.type === 'individual_results');
    
    if (individualResults.length === 0) {
        comparisonGrid.innerHTML = '<p>No evaluation data loaded for comparison</p>';
        return;
    }
    
    comparisonGrid.innerHTML = individualResults.map((result, index) => {
        const accuracy = (result.average_score * 100).toFixed(1);
        const correctCount = Math.round(result.total_examples * result.average_score);
        const incorrectCount = result.total_examples - correctCount;
        
        return `
            <div class="card comparison-card" onclick="viewIndividualResults('${result.benchmark}', '${result.suffix}')" 
                 style="cursor: pointer; transition: transform 0.2s ease, box-shadow 0.2s ease;"
                 onmouseover="this.style.transform='translateY(-4px)'; this.style.boxShadow='0 8px 25px rgba(0,0,0,0.2)'"
                 onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 2px 10px rgba(0,0,0,0.1)'">
                <h3>${result.benchmark}</h3>
                <p><strong>Optimizer:</strong> ${result.suffix}</p>
                <p><strong>Accuracy:</strong> ${accuracy}%</p>
                <p><strong>Examples:</strong> ${result.total_examples}</p>
                <p><strong>Correct:</strong> ${correctCount}</p>
                <p><strong>Incorrect:</strong> ${incorrectCount}</p>
                <p><strong>Timestamp:</strong> ${new Date(result.timestamp).toLocaleString()}</p>
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e0e0e0; color: #667eea; font-weight: 500;">
                    📊 Click to view individual results
                </div>
            </div>
        `;
    }).join('');
}

// Function to view individual results for a specific dataset/optimizer combination
function viewIndividualResults(benchmark, optimizer) {
    // Switch to the individual results tab
    showTab('individual');
    
    // Set the filters to match the selected dataset/optimizer
    document.getElementById('datasetSelect').value = benchmark;
    document.getElementById('optimizerSelect').value = optimizer;
    
    // Clear other filters
    document.getElementById('scoreFilter').value = '';
    document.getElementById('searchInput').value = '';
    
    // Apply the filters to show only the relevant results
    filterResults();
    
    // Scroll to the results table
    setTimeout(() => {
        const resultsContainer = document.getElementById('resultsContainer');
        if (resultsContainer) {
            resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }, 100);
}

// Update loaded files list
function updateLoadedFilesList() {
    const loadedFiles = document.getElementById('loadedFiles');
    
    if (evaluationData.length === 0) {
        loadedFiles.innerHTML = '<p>No files loaded yet</p>';
        return;
    }
    
    loadedFiles.innerHTML = evaluationData.map(data => {
        const typeLabel = data.type.replace('_', ' ').toUpperCase();
        const timestamp = new Date(data.timestamp).toLocaleString();
        
        return `
            <div class="card">
                <h4>${data.filename}</h4>
                <p><strong>Type:</strong> ${typeLabel}</p>
                <p><strong>Loaded:</strong> ${timestamp}</p>
                ${data.benchmark ? `<p><strong>Benchmark:</strong> ${data.benchmark}</p>` : ''}
                ${data.average_score ? `<p><strong>Average Score:</strong> ${(data.average_score * 100).toFixed(1)}%</p>` : ''}
                ${data.total_examples ? `<p><strong>Examples:</strong> ${data.total_examples}</p>` : ''}
            </div>
        `;
    }).join('');
}
