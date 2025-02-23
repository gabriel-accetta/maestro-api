from flask import Blueprint, request, jsonify
from flask_cors import CORS
from app.services.analysis_service import analyze_video

api_bp = Blueprint('api', __name__)
CORS(api_bp)

@api_bp.route('/', methods=['GET'])
def health():
    return "API is running"
    
@api_bp.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video = request.files['video']
    result = analyze_video(video)
    print(f"/analyze Response: {result}")
    return jsonify(result)