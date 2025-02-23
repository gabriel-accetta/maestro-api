from app.analysis.posture import analyze_posture

def analyze_video(video):
    posture_result = analyze_posture(video)

    return {
        "posture": posture_result,
    }