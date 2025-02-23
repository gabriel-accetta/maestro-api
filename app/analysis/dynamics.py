import os
import json
import logging
import librosa
import numpy as np
from dataclasses import dataclass
from typing import Dict
from moviepy.editor import VideoFileClip
from scipy.ndimage import uniform_filter1d

@dataclass
class DynamicsConfig:
    # Thresholds for dynamics analysis - adjusted based on real performance data
    DYNAMICS_CV_THRESHOLD_EXCELLENT: float = 0.75  # Was 0.25 - now accommodates natural piano dynamics
    DYNAMICS_CV_THRESHOLD_GOOD: float = 1.00       # Was 0.40 - adjusted for realistic variation
    MIN_AUDIO_DURATION: float = 5.0                # Minimum duration in seconds
    SMOOTHING_WINDOW: int = 50                     # Window size for smoothing
    # Analysis parameters
    HOP_LENGTH: int = 512                          # Number of samples between frames
    FRAME_LENGTH: int = 2048                       # Length of the FFT window

class DynamicsAnalyzer:
    def __init__(self):
        self.config = DynamicsConfig()
        
        # Load recommendations
        recommendations_path = os.path.join(os.path.dirname(__file__), '../data/recommendations.json')
        with open(recommendations_path, 'r', encoding='utf-8') as f:
            recommendations_list = json.load(f)
            self.recommendations = {str(rec['id']): rec for rec in recommendations_list}
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def analyze_audio(self, audio_path: str) -> Dict:
        """Analyze dynamics in an audio file."""
        self.logger.info(f"Starting dynamics analysis for audio: {audio_path}")
        self.logger.info(f"Current thresholds - Excellent: {self.config.DYNAMICS_CV_THRESHOLD_EXCELLENT}, Good: {self.config.DYNAMICS_CV_THRESHOLD_GOOD}")
        
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)
            self.logger.info(f"Audio loaded successfully - Duration: {librosa.get_duration(y=y, sr=sr):.2f}s, Sample rate: {sr}Hz")
        except Exception as e:
            self.logger.error(f"Error loading audio: {str(e)}")
            return {
                "title": "Dynamics",
                "rating": "Not enough data",
                "feedback": ["Error loading audio file."],
                "recommendations": []
            }

        # Check audio duration
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < self.config.MIN_AUDIO_DURATION:
            self.logger.warning("Audio too short for reliable analysis")
            return {
                "title": "Dynamics",
                "rating": "Not enough data",
                "feedback": ["Audio sample too short for reliable analysis."],
                "recommendations": []
            }

        try:
            # Compute RMS energy
            S = librosa.stft(y, n_fft=self.config.FRAME_LENGTH, 
                           hop_length=self.config.HOP_LENGTH)
            rms = librosa.feature.rms(S=S)[0]
            
            # Smooth the RMS curve
            smoothed_rms = uniform_filter1d(rms, size=self.config.SMOOTHING_WINDOW)
            
            # Calculate dynamics variation with detailed logging
            cv = np.std(rms) / np.mean(rms) if np.mean(rms) > 0 else float('inf')
            
            self.logger.info("=== Dynamics Analysis Details ===")
            self.logger.info(f"Mean RMS: {np.mean(rms):.4f}")
            self.logger.info(f"Std RMS: {np.std(rms):.4f}")
            self.logger.info(f"Coefficient of Variation (CV): {cv:.4f}")
            self.logger.info(f"CV Thresholds - Excellent: {cv < self.config.DYNAMICS_CV_THRESHOLD_EXCELLENT}, Good: {cv < self.config.DYNAMICS_CV_THRESHOLD_GOOD}")
            
            # Calculate residuals with logging
            residuals = np.abs(rms - smoothed_rms)
            residual_cv = np.std(residuals) / np.mean(residuals) if np.mean(residuals) > 0 else float('inf')
            self.logger.info(f"Residual CV: {residual_cv:.4f}")
            
            recommendations_arr = []
            
            # Updated feedback messages to reflect more realistic expectations
            if cv < self.config.DYNAMICS_CV_THRESHOLD_EXCELLENT:
                rating = "Excellent"
                feedback = ["Your dynamic control demonstrates professional-level consistency while maintaining expressive variation."]
            elif cv < self.config.DYNAMICS_CV_THRESHOLD_GOOD:
                rating = "Good"
                feedback = ["Your dynamic control shows good musical expression with reasonable consistency."]
                recommendations_arr.append(self.recommendations["10"])
            else:
                rating = "Needs Improvement"
                feedback = [
                    "Your dynamics show more variation than typical for this style.",
                    "Focus on controlled expression while maintaining intended volume levels.",
                ]
                recommendations_arr.extend([
                    self.recommendations["6"],
                    self.recommendations["10"]
                ])

            return {
                "title": "Dynamics",
                "rating": rating,
                "feedback": feedback,
                "recommendations": recommendations_arr
            }

        except Exception as e:
            self.logger.error(f"Error during dynamics analysis: {str(e)}")
            return {
                "title": "Dynamics",
                "rating": "Not enough data",
                "feedback": ["Error during dynamics analysis."],
                "recommendations": []
            }

    def process_video(self, video_path: str) -> Dict:
        """Extract audio from video and analyze dynamics."""
        self.logger.info(f"Processing video for dynamics analysis: {video_path}")
        temp_audio_path = "temp_dynamics_audio.wav"
        clip = None
        
        try:
            clip = VideoFileClip(video_path)
            clip.audio.write_audiofile(temp_audio_path, logger=None)
            analysis = self.analyze_audio(temp_audio_path)
            return analysis
        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}")
            return {
                "title": "Dynamics",
                "rating": "Not enough data",
                "feedback": ["Error processing video file."],
                "recommendations": []
            }
        finally:
            # Clean up resources
            if clip is not None:
                clip.close()
            try:
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                    self.logger.info("Temporary audio file removed")
            except Exception as e:
                self.logger.warning(f"Could not remove temporary audio file: {str(e)}")

    def process_video_from_file(self, video_file) -> Dict:
        """Process video from either a file path or file object."""
        if isinstance(video_file, str):
            return self.process_video(video_file)
        return self.process_video(video_file)

def analyze_dynamics(video_file) -> Dict:
    """Main function to analyze dynamics."""
    analyzer = DynamicsAnalyzer()
    return analyzer.process_video_from_file(video_file)

if __name__ == "__main__":
    video_path = "video.mp4"
    analyzer = DynamicsAnalyzer()
    result = analyzer.process_video(video_path)
    logging.info("Dynamics analysis completed. Result:")
    print(result)
