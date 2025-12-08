// API endpoint - use relative path to work on both local and production
const API_URL = window.location.origin;

// Form submission handler
document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Show loading state
    const predictBtn = document.getElementById('predictBtn');
    const btnText = predictBtn.querySelector('.btn-text');
    const btnLoading = predictBtn.querySelector('.btn-loading');
    
    btnText.style.display = 'none';
    btnLoading.style.display = 'inline';
    predictBtn.disabled = true;
    
    // Hide previous results/errors
    document.getElementById('resultsCard').style.display = 'none';
    document.getElementById('errorCard').style.display = 'none';
    
    // Collect form data
    const formData = new FormData(e.target);
    const patientData = {
        'cycle_number': parseInt(formData.get('cycle_number')),
        'Age': parseFloat(formData.get('Age')),
        'AMH': parseFloat(formData.get('AMH')),
        'n_Follicles': parseInt(formData.get('n_Follicles')),
        'E2_day5': parseFloat(formData.get('E2_day5')),
        'AFC': parseInt(formData.get('AFC')),
        'Protocol_agonist': formData.get('Protocol_agonist') === 'on',
        'Protocol_fixed antagonist': formData.get('Protocol_fixed_antagonist') === 'on',
        'Protocol_flexible antagonist': formData.get('Protocol_flexible_antagonist') === 'on'
    };
    
    try {
        // Make prediction request
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(patientData)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Prediction failed');
        }
        
        const result = await response.json();
        
        // Display results
        displayResults(result);
        
    } catch (error) {
        // Display error
        displayError(error.message);
    } finally {
        // Reset button state
        btnText.style.display = 'inline';
        btnLoading.style.display = 'none';
        predictBtn.disabled = false;
    }
});

function displayResults(result) {
    // Show results card
    const resultsCard = document.getElementById('resultsCard');
    resultsCard.style.display = 'block';
    
    // Scroll to results
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    
    // Update prediction badge
    const predictionValue = document.getElementById('predictionValue');
    predictionValue.textContent = result.prediction.toUpperCase();
    
    // Update badge color based on prediction
    const badge = document.getElementById('predictionBadge');
    badge.style.background = getPredictionGradient(result.prediction);
    
    // Update confidence
    const confidencePercent = (result.confidence * 100).toFixed(1);
    document.getElementById('confidencePercent').textContent = confidencePercent + '%';
    document.getElementById('confidenceFill').style.width = confidencePercent + '%';
    
    // Update probabilities
    updateProbability('Low', result.probabilities.low, 'probLow', 'probFillLow');
    updateProbability('Optimal', result.probabilities.optimal, 'probOptimal', 'probFillOptimal');
    updateProbability('High', result.probabilities.high, 'probHigh', 'probFillHigh');
    
    // Update clinical note
    document.getElementById('clinicalNoteText').textContent = result.clinical_note;
}

function updateProbability(label, value, valueId, fillId) {
    const percent = (value * 100).toFixed(1);
    document.getElementById(valueId).textContent = percent + '%';
    document.getElementById(fillId).style.width = percent + '%';
}

function getPredictionGradient(prediction) {
    const gradients = {
        'low': 'linear-gradient(135deg, #ef5350 0%, #c62828 100%)',
        'optimal': 'linear-gradient(135deg, #26a69a 0%, #00695c 100%)',
        'high': 'linear-gradient(135deg, #66bb6a 0%, #2e7d32 100%)'
    };
    return gradients[prediction.toLowerCase()] || 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
}

function displayError(message) {
    const errorCard = document.getElementById('errorCard');
    const errorMessage = document.getElementById('errorMessage');
    
    errorMessage.textContent = message;
    errorCard.style.display = 'block';
    
    // Scroll to error
    errorCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Check API connection on page load
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_URL}/`);
        if (!response.ok) {
            throw new Error('API connection failed');
        }
        console.log('✓ Connected to API');
    } catch (error) {
        displayError('Cannot connect to API server. Please make sure the API is running on http://localhost:8000');
    }
});
