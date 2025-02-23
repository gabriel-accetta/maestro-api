import math
import cv2
import mediapipe as mp
import numpy as np
from openai import OpenAI
import json
import os
from dotenv import load_dotenv
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

# Configuration
@dataclass
class PostureConfig:
    FRAMES_TO_SAVE: int = 5
    SHOULDER_THRESHOLD_EXCELLENT: float = 0.03
    SHOULDER_THRESHOLD_GOOD: float = 0.07
    BACK_ANGLE_THRESHOLD_EXCELLENT: float = 5.0  # degrees
    BACK_ANGLE_THRESHOLD_GOOD: float = 10.0  # degrees
    FOREARM_ANGLE_THRESHOLD_EXCELLENT: float = 10.0  # degrees deviation from 90°
    FOREARM_ANGLE_THRESHOLD_GOOD: float = 20.0  # degrees deviation from 90°
    MIN_DETECTION_CONFIDENCE: float = 0.5
    MIN_TRACKING_CONFIDENCE: float = 0.5

class PostureAnalyzer:
    def __init__(self):
        # MediaPipe initialization
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.config = PostureConfig()
        
        # OpenAI initialization
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Configure logging with console output
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        
        # Create console handler and set level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add console handler to logger
        logger.addHandler(console_handler)

    def extract_landmarks(self, landmarks) -> Dict[str, Tuple[float, float]]:
        """Extract relevant landmarks for posture analysis."""
        return {
            'left_shoulder': (
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y
            ),
            'right_shoulder': (
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            ),
            'left_hip': (
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y
            ),
            'right_hip': (
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y
            ),
            'nose': (
                landmarks[self.mp_pose.PoseLandmark.NOSE].x,
                landmarks[self.mp_pose.PoseLandmark.NOSE].y
            ),
            'left_elbow': (
                landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].y
            ),
            'right_elbow': (
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y
            ),
            'left_wrist': (
                landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y
            ),
            'right_wrist': (
                landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y
            )
        }
    
    def calculate_shoulder_difference(self, landmarks: dict) -> float:
        """
        Calculate a normalized shoulder difference that is robust to different camera placements.

        This function rotates the shoulder coordinates based on the body orientation.
        It uses the midpoints of the shoulders and hips to compute a rotation angle that 
        aligns the torso vertically. Then it computes the vertical difference between the 
        rotated left and right shoulders and normalizes it by the horizontal shoulder distance.
        
        Parameters:
            landmarks: dict
                A dictionary with keys 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'
                where each value is a tuple (x, y).

        Returns:
            normalized_diff: float
                A measure of shoulder height asymmetry normalized by shoulder width.
        """
        # Convert landmarks to numpy arrays
        left_shoulder = np.array(landmarks['left_shoulder'])
        right_shoulder = np.array(landmarks['right_shoulder'])
        left_hip = np.array(landmarks['left_hip'])
        right_hip = np.array(landmarks['right_hip'])
        
        # Compute midpoints
        mid_shoulder = (left_shoulder + right_shoulder) / 2.0
        mid_hip = (left_hip + right_hip) / 2.0
        
        # Compute vector from mid_hip to mid_shoulder
        torso_vector = mid_shoulder - mid_hip
        # Angle between torso vector and vertical (0, 1)
        # Using arctan2(dx, dy) gives the rotation needed to align with vertical
        angle = math.atan2(torso_vector[0], torso_vector[1])
        
        # Create rotation matrix to align the torso vertically (rotate by -angle)
        cos_a = math.cos(-angle)
        sin_a = math.sin(-angle)
        rotation_matrix = np.array([[cos_a, -sin_a],
                                    [sin_a,  cos_a]])
        
        # Rotate the shoulder coordinates
        left_rotated = rotation_matrix.dot(left_shoulder)
        right_rotated = rotation_matrix.dot(right_shoulder)
        
        # Compute vertical difference (along y-axis in rotated frame)
        vertical_diff = abs(left_rotated[1] - right_rotated[1])
        
        # Compute horizontal distance (shoulder width) in the rotated frame
        horizontal_distance = abs(left_rotated[0] - right_rotated[0])
        
        # Normalize the vertical difference by the shoulder width
        normalized_diff = vertical_diff / horizontal_distance if horizontal_distance != 0 else vertical_diff
        return normalized_diff

    def calculate_back_angle(self, landmarks: Dict[str, Tuple[float, float]]) -> float:
        """Calculate the angle of the back relative to vertical.
        Returns the deviation from vertical in degrees (0° means perfectly straight)."""
        mid_shoulder = (
            (landmarks['left_shoulder'][0] + landmarks['right_shoulder'][0]) / 2,
            (landmarks['left_shoulder'][1] + landmarks['right_shoulder'][1]) / 2
        )
        mid_hip = (
            (landmarks['left_hip'][0] + landmarks['right_hip'][0]) / 2,
            (landmarks['left_hip'][1] + landmarks['right_hip'][1]) / 2
        )
        
        # Calculate angle with vertical
        import math
        dx = mid_shoulder[0] - mid_hip[0]
        dy = mid_shoulder[1] - mid_hip[1]
        angle = abs(90 - abs(math.degrees(math.atan2(dy, dx))))
        return angle

    def calculate_forearm_angle(self, landmarks: Dict[str, Tuple[float, float]]) -> float:
        """Calculate the average deviation of forearms from 90 degrees relative to the back."""
        # Calculate back vector (from mid_hip to mid_shoulder)
        mid_shoulder = (
            (landmarks['left_shoulder'][0] + landmarks['right_shoulder'][0]) / 2,
            (landmarks['left_shoulder'][1] + landmarks['right_shoulder'][1]) / 2
        )
        mid_hip = (
            (landmarks['left_hip'][0] + landmarks['right_hip'][0]) / 2,
            (landmarks['left_hip'][1] + landmarks['right_hip'][1]) / 2
        )
        
        back_vector = (
            mid_shoulder[0] - mid_hip[0],
            mid_shoulder[1] - mid_hip[1]
        )
        
        # Calculate left and right forearm vectors
        left_forearm_vector = (
            landmarks['left_wrist'][0] - landmarks['left_elbow'][0],
            landmarks['left_wrist'][1] - landmarks['left_elbow'][1]
        )
        right_forearm_vector = (
            landmarks['right_wrist'][0] - landmarks['right_elbow'][0],
            landmarks['right_wrist'][1] - landmarks['right_elbow'][1]
        )
        
        # Calculate angles between back and forearms
        def calculate_angle(v1, v2):
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            v1_mag = math.sqrt(v1[0]**2 + v1[1]**2)
            v2_mag = math.sqrt(v2[0]**2 + v2[1]**2)
            angle_rad = math.acos(dot_product / (v1_mag * v2_mag))
            return math.degrees(angle_rad)
        
        left_angle = abs(90 - calculate_angle(back_vector, left_forearm_vector))
        right_angle = abs(90 - calculate_angle(back_vector, right_forearm_vector))
        
        # Return average deviation from 90 degrees
        return (left_angle + right_angle) / 2

    def save_annotated_frame(self, frame: Any, results: Any, frame_data: Dict, 
                            current_second: int, total_seconds: int, output_path: str) -> None:
        """Save frame with annotations and measurements."""
        self.mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )

        shoulder_diff = self.calculate_shoulder_difference(frame_data)
        back_angle = self.calculate_back_angle(frame_data)
        forearm_angle = self.calculate_forearm_angle(frame_data)
        rating = self.get_posture_rating(shoulder_diff, back_angle, forearm_angle)

        text_lines = [
            f"Time: {current_second}s / {total_seconds}s",
            f"Shoulder difference: {shoulder_diff:.3f}",
            f"Back angle: {back_angle:.1f}",
            f"Forearm angle dev: {forearm_angle:.1f}°",
            f"Rating: {rating}"
        ]

        for i, text in enumerate(text_lines):
            cv2.putText(frame, text, (10, 30 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imwrite(output_path, frame)
        logging.info(f"Saved frame at {current_second}s to {output_path}")

    def get_posture_rating(self, shoulder_diff: float, back_angle: float, forearm_angle: float) -> str:
        """Determine posture rating based on shoulder difference, back angle, and forearm angle."""
        if (shoulder_diff < self.config.SHOULDER_THRESHOLD_EXCELLENT and 
            back_angle < self.config.BACK_ANGLE_THRESHOLD_EXCELLENT and
            forearm_angle < self.config.FOREARM_ANGLE_THRESHOLD_EXCELLENT):
            return "Excellent"
        elif (shoulder_diff < self.config.SHOULDER_THRESHOLD_GOOD and 
              back_angle < self.config.BACK_ANGLE_THRESHOLD_GOOD and
              forearm_angle < self.config.FOREARM_ANGLE_THRESHOLD_GOOD):
            return "Good"
        return "Needs Improvement"

    def get_ai_feedback(self, avg_shoulder_diff: float, avg_back_angle: float, 
                       avg_forearm_angle: float, rating: str) -> List[str]:
        """Get AI-generated feedback for posture improvement."""
        system_prompt = (
            "You are a helpful assistant that provides feedback on a pianist's posture. "
            "The output format should be a list of at most 3 suggestions, each starting with a dash (-) "
            "and followed by a space. Ensure the response is at most 3 suggestions."
        )

        prompt_text = (
            f"You are analyzing a pianist's posture. The average vertical difference "
            f"between the left and right shoulders is {avg_shoulder_diff:.3f}. "
            f"The average back angle deviation from vertical is {avg_back_angle:.1f} degrees. "
            f"The average forearm angle deviation from 90 degrees is {avg_forearm_angle:.1f} degrees. "
            f"The current rating is {rating}. "
            "Provide at most 3 specific suggestions to improve or maintain posture."
        )

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=100,
            temperature=0.7
        )

        raw_feedback = response.choices[0].message.content.strip()
        return [line.strip("- ") for line in raw_feedback.split("\n") if line.strip()]

    def process_video(self, video_path: str) -> str:
        """Process video and analyze posture."""
        logging.info(f"Starting posture analysis for video: {video_path}")
        
        if not os.path.exists(video_path):
            logging.error(f"Video file not found: {video_path}")
            return self.generate_error_response()

        pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=self.config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.config.MIN_TRACKING_CONFIDENCE
        )
        logging.debug("MediaPipe Pose model initialized")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error("Error opening video file")
            return self.generate_error_response()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_seconds = int(total_frames / fps)
        logging.info(f"Video duration: {total_seconds} seconds, FPS: {fps}")
        
        # Calculate seconds at which to save frames, evenly distributed
        seconds_between_saves = total_seconds / (self.config.FRAMES_TO_SAVE - 1)
        save_at_seconds = [int(i * seconds_between_saves) for i in range(self.config.FRAMES_TO_SAVE)]
        
        posture_data = []
        frame_count = 0
        saved_frames = 0
        current_second = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only process one frame per second
            if frame_count % fps == 0:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    current_posture = self.extract_landmarks(results.pose_landmarks.landmark)
                    posture_data.append(current_posture)
                    
                    if current_second in save_at_seconds:
                        self.save_annotated_frame(
                            frame, results, current_posture,
                            current_second, total_seconds,
                            f"frame_{saved_frames:02d}.jpg"
                        )
                        saved_frames += 1
                else:
                    logging.warning(f"Second {current_second}: No pose landmarks detected")

                current_second += 1

            frame_count += 1

        cap.release()
        logging.info(f"Video processing completed. Analyzed {len(posture_data)} frames")
        
        if len(posture_data) < 10:
            logging.error("Insufficient pose data collected")
            return self.generate_error_response()

        avg_shoulder_diff = self.calculate_average_shoulder_difference(posture_data)
        avg_back_angle = self.calculate_average_back_angle(posture_data)
        avg_forearm_angle = self.calculate_average_forearm_angle(posture_data)
        logging.info(f"Average shoulder difference: {avg_shoulder_diff:.3f}")
        logging.info(f"Average back angle: {avg_back_angle:.1f}°")
        logging.info(f"Average forearm angle deviation: {avg_forearm_angle:.1f}°")
        
        rating = self.get_posture_rating(avg_shoulder_diff, avg_back_angle, avg_forearm_angle)
        logging.info(f"Final posture rating: {rating}")
        
        try:
            feedback = self.get_ai_feedback(avg_shoulder_diff, avg_back_angle, avg_forearm_angle, rating)
            logging.info("AI feedback generated successfully")
        except Exception as e:
            logging.error(f"Error getting AI feedback: {str(e)}")
            feedback = ["Error generating AI feedback"]

        return self.generate_analysis_response(rating, feedback)

    def calculate_average_shoulder_difference(self, posture_data: List[Dict]) -> float:
        """Calculate average shoulder height difference."""
        total_diff = sum(self.calculate_shoulder_difference(pd) for pd in posture_data)
        return total_diff / len(posture_data)

    def calculate_average_back_angle(self, posture_data: List[Dict]) -> float:
        """Calculate average back angle."""
        total_angle = sum(self.calculate_back_angle(pd) for pd in posture_data)
        return total_angle / len(posture_data)

    def calculate_average_forearm_angle(self, posture_data: List[Dict]) -> float:
        """Calculate average forearm angle."""
        total_angle = sum(self.calculate_forearm_angle(pd) for pd in posture_data)
        return total_angle / len(posture_data)

    def generate_error_response(self) -> str:
        """Generate error response when not enough data is collected."""
        return {
            "title": "Posture",
            "rating": "Not enough data",
            "feedback": ["Not enough posture data was collected to analyze. Make sure the video has a clear view of the pianist."]
        }

    def generate_analysis_response(self, rating: str, feedback: List[str]) -> str:
        """Generate final analysis response."""
        return {
            "title": "Posture",
            "rating": rating,
            "feedback": feedback
        }
    
    def process_video_from_file(self, video_file) -> str:
        """Process video from a file object and analyze posture."""
        # Save temporary file
        temp_path = "temp_video.mp4"
        video_file.save(temp_path)
        
        try:
            return self.process_video(temp_path)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logging.info("Temporary video file removed")


def analyze_posture(video_file) -> str:
    """Main function to analyze pianist's posture from a file object."""
    analyzer = PostureAnalyzer()
    return analyzer.process_video_from_file(video_file)


def analyze_posture_from_path(video_path: str) -> str:
    """Main function to analyze pianist's posture from a file path."""
    analyzer = PostureAnalyzer()
    return analyzer.process_video(video_path)


if __name__ == "__main__":
    video_path = "video.mp4"
    logging.info("Script started.")
    analysis_json = analyze_posture_from_path(video_path)
    logging.info("Analysis completed. Result: %s", analysis_json)
    print(analysis_json)