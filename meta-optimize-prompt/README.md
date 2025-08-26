# MetaPromptBench

A web-based visualization tool for analyzing and comparing prompt optimization evaluation results.

## Features

- **Overview Dashboard**: View key metrics, performance comparisons, and score distributions
- **Individual Results**: Browse and filter individual prediction results with detailed information
- **Comparison View**: Compare different optimizers and prompts side-by-side
- **File Upload**: Drag and drop JSON evaluation files for instant analysis
- **Automatic Discovery**: Automatically scans and loads evaluation results from the `generated/` directory
- **Responsive Design**: Works on desktop and mobile devices

## Quick Start

### Option 1: With Python Server (Recommended)

1. **Start the server:**
   ```bash
   cd meta-optimize-prompt
   python server.py
   ```

2. **Open your browser:**
   Navigate to `http://localhost:8000/evaluation_viewer.html`

3. **View results:**
   The viewer will automatically scan and load evaluation results from your `generated/` directory.

### Option 2: Static Files Only

1. **Open the HTML file:**
   Simply open `evaluation_viewer.html` in your web browser

2. **Upload files manually:**
   Use the "Upload Data" tab to drag and drop your JSON evaluation files

## Supported File Types

The viewer can process several types of evaluation files:

### Individual Results Files
Files containing detailed prediction results for each example:
```json
{
  "metadata": {
    "benchmark": "HeartDiseaseBench",
    "program": "Predict",
    "optimizer": "baseline",
    "total_examples": 100,
    "average_score": 0.32,
    "timestamp": "2025-08-22T17:48:44.712172",
    "suffix": "extracted_default"
  },
  "individual_results": [...]
}
```

### Prompt Configuration Files
Files containing prompt instructions and field definitions:
```json
{
  "signature": {
    "instructions": "Given patient information, predict...",
    "fields": [...]
  }
}
```

## Directory Structure

The viewer expects evaluation results to be organized as follows:

```
meta-optimize-prompt/
├── generated/
│   ├── v2025_08_22_1747/
│   │   └── evaluation_results/
│   │       ├── HeartDisease/
│   │       │   ├── *.json
│   │       │   └── *.csv
│   │       └── judgebench/
│   │           ├── *.json
│   │           └── *.csv
│   └── v2025_08_22_1733/
│       └── evaluation_results/
│           └── ...
├── evaluation_viewer.html
├── evaluation_viewer.js
└── server.py
```

## Usage Guide

### Overview Tab
- **Performance Metrics**: Key statistics across all loaded datasets
- **Performance Comparison**: Bar chart comparing accuracy across different optimizers
- **Score Distribution**: Pie chart showing correct vs incorrect predictions

### Individual Results Tab
- **Filters**: Filter by dataset, optimizer, score, or search terms
- **Results Table**: Detailed view of individual predictions
- **Export**: Copy or download filtered results

### Comparison Tab
- **Model Cards**: Side-by-side comparison of different models/optimizers
- **Performance Trends**: Line chart showing performance over time

### Upload Data Tab
- **Drag & Drop**: Simply drag JSON files onto the upload area
- **File Browser**: Click to select files from your computer
- **Loaded Files**: View all currently loaded data files

## API Endpoints

When using the Python server, the following API endpoints are available:

- `GET /api/scan-results`: Scan the `generated/` directory for evaluation files
- `GET /api/load-file?path=<file_path>`: Load a specific evaluation file

## Customization

### Adding New Visualizations

To add new charts or visualizations:

1. Add a new canvas element to the HTML
2. Create a chart update function in the JavaScript
3. Call the function from `updateAllViews()`

### Supporting New File Formats

To support additional file formats:

1. Update the `processEvaluationData()` function to handle the new format
2. Add appropriate type detection logic
3. Update the visualization functions to handle the new data structure

## Troubleshooting

### Server Won't Start
- Ensure Python 3.6+ is installed
- Check that port 8000 is not in use
- Try running with `python3 server.py` instead

### Files Not Loading
- Verify files are valid JSON
- Check browser console for error messages
- Ensure file paths are correct when using the API

### Charts Not Displaying
- Check that Chart.js is loading properly
- Verify data format matches expected structure
- Look for JavaScript errors in browser console

## Browser Compatibility

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## Dependencies

### Frontend
- Chart.js 3.x (loaded from CDN)
- Modern browser with ES6 support

### Backend (Optional)
- Python 3.6+
- Standard library only (no additional packages required)

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the Meta-Optimize Prompt framework. Please refer to the main project license for terms and conditions.
