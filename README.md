# Maestro API - Piano performance analyzer

A comprehensive API system for piano performance analysis using computer vision and machine learning.

## Overview

This project consists of two main components:
- REST API for handling data and analysis results
- Real-time posture analyzer using computer vision

## Prerequisites

- Python 3.8+
- Webcam or video input device
- Required Python packages (install via pip):
  ```bash
  pip install -r requirements.txt
  ```

## Setup

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd maestro-api
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env file with your configurations
   ```

## Running the Application

### REST API Server

Start the API server:
```bash
python .\run.py
```

The API will be available at `http://localhost:5000`

### Real-time Posture Analyzer

Launch the real-time posture analysis:
```bash
python -m app.analysis.realtime_posture
```

This will start the webcam feed and begin analyzing posture in real-time.

## Features

- REST API endpoints for different technique analysis
- Real-time posture detection and analysis

## API Endpoints

- `GET /` - Check API health
- `POST /analyze` - Submit video for analysis
