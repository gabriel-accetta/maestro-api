import cv2
import time
from .posture import PostureAnalyzer
import logging

class RealTimePostureAnalyzer(PostureAnalyzer):
    def __init__(self):
        super().__init__()
        self.is_running = False

    def start_realtime_analysis(self, camera_id=0):
        """Start realtime posture analysis using webcam."""
        cap = cv2.VideoCapture(camera_id)
        self.is_running = True
        
        pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=self.config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.config.MIN_TRACKING_CONFIDENCE
        )

        frame_count = 0
        start_time = time.time()
        fps = 0

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - start_time)
                start_time = time.time()

            # Process frame
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                # Extract landmarks and calculate metrics
                current_posture = self.extract_landmarks(results.pose_landmarks.landmark)
                shoulder_diff = self.calculate_shoulder_difference(current_posture)
                back_angle = self.calculate_back_angle(current_posture)
                forearm_angle = self.calculate_forearm_angle(current_posture)
                rating = self.get_posture_rating(shoulder_diff, back_angle, forearm_angle)

                # Draw pose landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )

                # Display metrics
                text_lines = [
                    f"FPS: {fps:.1f}",
                    f"Shoulder difference: {shoulder_diff:.3f}",
                    f"Back angle: {back_angle:.1f}",
                    f"Forearm angle dev: {forearm_angle:.1f}",
                    f"Rating: {rating}"
                ]

                for i, text in enumerate(text_lines):
                    cv2.putText(frame, text, (10, 30 + i * 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Color-coded rating indicator
                rating_colors = {
                    "Excellent": (0, 255, 0),  # Green
                    "Good": (0, 255, 255),     # Yellow
                    "Needs Improvement": (0, 0, 255)  # Red
                }
                cv2.circle(frame, (frame.shape[1] - 30, 30), 20, 
                          rating_colors.get(rating, (0, 0, 255)), -1)

            # Display frame
            cv2.imshow('Realtime Posture Analysis', frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.is_running = False
            elif key == ord('s'):
                cv2.imwrite(f'posture_snapshot_{time.time()}.jpg', frame)
                logging.info("Snapshot saved")

        cap.release()
        cv2.destroyAllWindows()

def start_realtime_analysis(camera_id=0):
    """Convenience function to start realtime analysis."""
    analyzer = RealTimePostureAnalyzer()
    analyzer.start_realtime_analysis(camera_id)

if __name__ == "__main__":
    start_realtime_analysis()
