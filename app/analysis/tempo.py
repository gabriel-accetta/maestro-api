import os
import json
import logging
import librosa
import numpy as np
from dataclasses import dataclass
from typing import Dict
from pydantic import BaseModel
from moviepy.editor import VideoFileClip

# Tempo configuration thresholds
@dataclass
class TempoConfig:
    # CV thresholds for consistency:
    CV_THRESHOLD_EXCELLENT: float = 0.05
    CV_THRESHOLD_GOOD: float = 0.10
    # Minimum audio duration for analysis (in seconds)
    MIN_AUDIO_DURATION: float = 5.0

class TempoAnalyzer:
    def __init__(self):
        self.config = TempoConfig()
        
        # Load recommendations from JSON file and transform into dictionary
        recommendations_path = os.path.join(os.path.dirname(__file__), '../data/recommendations.json')
        with open(recommendations_path, 'r', encoding='utf-8') as f:
            recommendations_list = json.load(f)
            self.recommendations = {str(rec['id']): rec for rec in recommendations_list}

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def analyze_audio(self, audio_path: str) -> Dict:
        """
        Analyze the tempo consistency of an audio file.
        Computes beat times, inter-beat intervals (IBI), and the coefficient of variation (CV).
        Returns a TechniqueAnalysis with title 'Tempo' and a rating based on the CV.
        """
        self.logger.info(f"Starting tempo analysis for audio: {audio_path}")
        try:
            y, sr = librosa.load(audio_path, sr=None)
        except Exception as e:
            self.logger.error(f"Error loading audio: {str(e)}")
            return {
                "title": "Tempo",
                "rating": "Not enough data",
                "feedback": ["Error loading audio."],
                "recommendations": []
            }
        
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < self.config.MIN_AUDIO_DURATION:
            self.logger.error("Audio too short for reliable tempo analysis")
            return {
                "title": "Tempo",
                "rating": "Not enough data",
                "feedback": ["Audio is too short for reliable tempo analysis."],
                "recommendations": []
            }
        
        # Compute onset envelope and track beats
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        if len(beat_times) < 2:
            self.logger.error("Not enough beats detected for analysis")
            return {
                "title":"Tempo",
                "rating":"Not enough data",
                "feedback":["Not enough beats detected for tempo analysis."],
                "recommendations":[]
            }
        
        # Calculate inter-beat intervals (IBI) and coefficient of variation (CV)
        ibis = np.diff(beat_times)
        mean_ibi = np.mean(ibis)
        std_ibi = np.std(ibis)
        cv = std_ibi / mean_ibi
        self.logger.info(f"Mean IBI: {mean_ibi:.3f}s, Std IBI: {std_ibi:.3f}s, CV: {cv:.3f}")
        recommendations_arr = []
        
        # Determine rating based on CV
        if cv < self.config.CV_THRESHOLD_EXCELLENT:
            rating = "Excellent"
            feedback = ["The tempo is very consistent throughout the performance."]
        elif cv < self.config.CV_THRESHOLD_GOOD:
            rating = "Good"
            feedback = ["The tempo is fairly consistent, though there are minor fluctuations."]
            recommendations_arr.append(self.recommendations["3"])
        else:
            rating = "Needs Improvement"
            feedback = ["The tempo is inconsistent. Consider practicing with a metronome to maintain a steady pace."]
            recommendations_arr.append(self.recommendations["3"])
        
        return {
            "title":"Tempo",
            "rating":rating,
            "feedback":feedback,
            "recommendations":recommendations_arr
        }

    def process_video(self, video_path: str) -> Dict:
        """
        Extract audio from a video file using MoviePy and perform tempo analysis.
        Cleans up temporary audio file after processing.
        """
        self.logger.info(f"Extracting audio from video: {video_path}")
        temp_audio_path = "temp_audio.wav"
        clip = None
        try:
            clip = VideoFileClip(video_path)
            # Write audio to temporary file; suppress MoviePy logger output by passing logger=None
            clip.audio.write_audiofile(temp_audio_path, logger=None)
            # Analyze the extracted audio
            analysis = self.analyze_audio(temp_audio_path)
            return analysis
        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}")
            return {
                "title":"Tempo",
                "rating":"Not enough data",
                "feedback":["Error processing video file."],
                "recommendations":[]
            }
        finally:
            # Clean up resources
            if clip is not None:
                clip.close()
            # Try to remove temporary files
            try:
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                    self.logger.info("Temporary audio file removed")
            except Exception as e:
                self.logger.warning(f"Could not remove temporary audio file: {str(e)}")

    def process_video_from_file(self, video_file) -> Dict:
        """
        Process video from either a file path or file object.
        """
        # If input is already a path, use it directly
        if isinstance(video_file, str):
            return self.process_video(video_file)
        
        # If input is a file object, assume it's already been saved
        # and the path is passed
        return self.process_video(video_file)

def analyze_tempo(video_file) -> Dict:
    """Main function to analyze tempo consistency."""
    analyzer = TempoAnalyzer()
    return analyzer.process_video_from_file(video_file)

if __name__ == "__main__":
    video_path = "video.mp4"
    analyzer = TempoAnalyzer()
    result = analyzer.process_video(video_path)
    logging.info("Tempo analysis completed. Result:")
    print(result)
