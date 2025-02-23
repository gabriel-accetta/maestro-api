# Maestro - Piano Performance Analyzer (API)

Maestro API is the backend service that powers the piano performance analysis system. It processes video input to analyze various aspects of piano playing technique using computer vision and machine learning.

## Features

- Real-time posture analysis
- Hand position tracking
- Tempo detection
- Pedal usage analysis
- Dynamic level assessment

## Project Structure

This repository contains the backend API built with Python. The frontend application can be found at:
https://github.com/gabriel-accetta/maestro-front

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Getting Started

1. Clone the repository:
```bash
git clone [your-repository-url]
cd maestro-api
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the API server:
```bash
python run.py
```

5. The API will be available at http://127.0.0.1:5000

## API Endpoints

### Analysis Endpoints

- `POST /analyze`
  - 
