# IVF Response Prediction - Web Interface

Simple web UI for making patient response predictions.

## Quick Start

### Step 1: Start the API Server

```bash
cd "c:\Users\ghass\OneDrive\Desktop\tanit ai"
python src\api\app.py
```

Wait for the message: `Uvicorn running on http://127.0.0.1:8000`

### Step 2: Open the UI

Open `index.html` in your web browser:
- Double-click `index.html` in File Explorer, OR
- Right-click `index.html` → Open with → Chrome/Firefox/Edge

### Step 3: Make a Prediction

1. **Review the pre-filled example patient data** (or modify values)
2. **Select the stimulation protocol** (checkboxes at bottom of form)
3. **Click "Predict Response"** button
4. **View results:**
   - Predicted response class (Low/Optimal/High)
   - Confidence percentage
   - Probability breakdown for each class
   - Clinical interpretation

## Input Fields Explained

All values should be **normalized** (already scaled by the preprocessing pipeline):

| Field | Description | Example Value |
|-------|-------------|---------------|
| Cycle Number | Treatment attempt number (normalized) | 0.0 |
| Age | Patient age (normalized) | -0.5 |
| AMH | Anti-Müllerian Hormone level (normalized) | 1.56 |
| Number of Follicles | Follicle count at monitoring (normalized) | 1.33 |
| E2 Day 5 | Estradiol level day 5 (normalized) | -0.12 |
| AFC | Antral Follicle Count (normalized) | 2.87 |
| Protocol | Stimulation protocol (select one) | Flexible Antagonist |

## Understanding Results

**Prediction Badge:**
- Shows the predicted response category (LOW/OPTIMAL/HIGH)
- Color coded: Red (Low), Teal (Optimal), Green (High)

**Confidence Level:**
- Shows model confidence in the prediction
- Higher confidence = more certain prediction

**Probabilities:**
- Shows likelihood for each response category
- All three probabilities sum to 100%

**Clinical Note:**
- Provides clinical interpretation
- Suggests monitoring or protocol adjustments

## Troubleshooting

**"Cannot connect to API server" error:**
- Make sure API is running: `python src\api\app.py`
- Check that port 8000 is not blocked
- Verify API at: http://localhost:8000/docs

**No prediction shown after clicking button:**
- Open browser console (F12) to check for errors
- Verify all fields are filled
- Ensure API is responding (green checkmark in console)

**Wrong predictions:**
- Verify input values are **normalized** (not raw values)
- Check protocol selection (only one should be checked)

## Files

- `index.html` - Main interface
- `style.css` - Visual styling
- `app.js` - API communication logic
