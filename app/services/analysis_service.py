import tempfile
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from app.analysis.posture import analyze_posture
from app.analysis.tempo import analyze_tempo
from app.analysis.dynamics import analyze_dynamics

async def run_analysis(fn, video_path):
    """Run analysis function in a thread pool to avoid blocking."""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, fn, video_path)

async def analyze_video_async(video_path):
    """Run all analyses concurrently."""
    tasks = [
        run_analysis(analyze_posture, video_path),
        run_analysis(analyze_tempo, video_path),
        run_analysis(analyze_dynamics, video_path)
    ]
    posture_result, tempo_result, dynamics_result = await asyncio.gather(*tasks)
    
    return {
        "posture": posture_result,
        "tempo": tempo_result,
        "dynamics": dynamics_result
    }

def analyze_video(video):
    """Main entry point that handles the temporary file and runs async analysis."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_video_path = os.path.join(temp_dir, "temp_video.mp4")
        video.save(temp_video_path)
        
        # Run analyses concurrently using asyncio
        return asyncio.run(analyze_video_async(temp_video_path))