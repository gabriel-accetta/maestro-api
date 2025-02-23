import tempfile
import os
from app.analysis.posture import analyze_posture
from app.analysis.tempo import analyze_tempo
from app.analysis.dynamics import analyze_dynamics

def analyze_video(video):
    # Create a temporary directory that will be automatically cleaned up
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save video to temporary file
        temp_video_path = os.path.join(temp_dir, "temp_video.mp4")
        video.save(temp_video_path)
        
        # Analyze posture and tempo using the same temporary file
        # posture_result = analyze_posture(temp_video_path)
        # tempo_result = analyze_tempo(temp_video_path)
        dynamics_result = analyze_dynamics(temp_video_path)

        return {
            # "posture": posture_result,
            # "tempo": tempo_result,
            "dynamics": dynamics_result
        }