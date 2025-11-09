"""
Main integration script for YouTube Shorts automation.
Gets user preferences via GUI and continuously monitors YouTube Shorts URLs.
"""

import sys
import os
import asyncio
import subprocess
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import time
import cv2

# Import processing modules
from processing.metadata import YouTubeMetadataExtractor
from processing.embedding import VideoEmbeddingStore
from processing.llm import VideoDecisionSystem

# Import hand gesture controller
from cv.handGestureController import HandGestureController


def get_user_preference_from_gui():
    """
    Launch GUI in a separate process and get user input.
    This ensures the GUI is completely isolated and closes properly.
    """
    gui_script = Path(__file__).parent / "gui_begin.py"
    
    # Run GUI in a completely separate Python process
    result = subprocess.run(
        [sys.executable, str(gui_script)],
        capture_output=True,
        text=True
    )
    
    # Get the output (user's input)
    user_input = result.stdout.strip() if result.stdout else None
    return user_input if user_input else None


async def process_video(url: str, user_preference: str, metadata_extractor, embedding_store, decision_system):
    """
    Process a single video through the complete pipeline:
    1. Extract metadata
    2. Calculate similarity scores
    3. Get LLM decision
    4. Return decision and reasoning
    """
    try:
        metadata = metadata_extractor.get_processed_metadata(url, process_image=False)
        
        if not metadata:
            print(f"[Warning] Could not extract metadata for: {url}")
            return None, None, None
        
        stay_sim, scroll_sim = embedding_store.get_similarity_scores(metadata)
        stay_count, scroll_count = embedding_store.get_collection_sizes()
        
        # Print embedding scores
        print(f"\n--- Embedding Scores ---")
        print(f"Stay similarity: {stay_sim:.4f} (collection size: {stay_count})")
        print(f"Scroll similarity: {scroll_sim:.4f} (collection size: {scroll_count})")
        
        decision, reasoning = decision_system.make_decision(
            metadata=metadata,
            user_preferences=user_preference,
            stay_similarity=stay_sim,
            scroll_similarity=scroll_sim,
            stay_count=stay_count,
            scroll_count=scroll_count
        )
        
        return decision, reasoning, metadata
        
    except Exception as e:
        print(f"[Error] Failed to process video {url}: {e}")
        return None, None, None


def classify_and_add_to_collection(video_data: dict, watch_time: float, embedding_store):
    """
    Classify video as STAY or SCROLL based on actual watch time.
    Add to appropriate collection for learning.
    """
    duration = video_data['duration']
    metadata = video_data['metadata']
    llm_decision = video_data['decision']
    
    # More lenient classification:
    # If watched >= 30% of duration (was 50%), classify as STAY
    # OR if watched >= 3 seconds (was 1 second) for very short videos
    threshold_percentage = 0.30  # 30% instead of 50%
    min_watch_time = 3.0  # 3 seconds minimum
    
    watched_percentage = watch_time / duration if duration > 0 else 0
    
    if watch_time >= max(duration * threshold_percentage, min_watch_time):
        embedding_store.add_to_stay(metadata)
        print(f"Added to STAY collection (watched {watch_time:.1f}s / {duration}s = {watched_percentage*100:.1f}%)")
        
        # Show if this contradicts LLM prediction
        if llm_decision == "SCROLL":
            print(f"  ⚠️ LLM predicted SCROLL but user stayed - learning from override")
    else:
        embedding_store.add_to_scroll(metadata)
        print(f"Added to SCROLL collection (watched {watch_time:.1f}s / {duration}s = {watched_percentage*100:.1f}%)")
        
        # Show if this contradicts LLM prediction
        if llm_decision == "STAY":
            print(f"  ⚠️ LLM predicted STAY but user scrolled - learning from override")


async def watch_video_with_monitoring(driver, video_data: dict, is_revisit: bool = False):
    """
    Watch video for specified time while monitoring for manual scrolls.
    - If LLM said SCROLL: wait 1.0 seconds then scroll
    - If LLM said STAY: wait full duration then scroll to next
    - If is_revisit: wait full duration regardless, then scroll
    """
    decision = video_data['decision']
    duration = video_data['duration']
    current_url = video_data['url']
    
    # Determine wait time based on LLM decision and revisit status
    if is_revisit:
        wait_time = duration  # Always wait full duration on revisits
    elif decision == "SCROLL":
        wait_time = 1.0  # Minimum 1.0 seconds
    else:  # STAY
        wait_time = duration  # Wait full duration for STAY videos
    
    start_time = time.time()
    target_end_time = start_time + wait_time
    
    # Monitor for manual scrolls while waiting
    while time.time() < target_end_time:
        if driver.current_url != current_url:
            # User manually scrolled
            return
        await asyncio.sleep(0.1)
    
    # Time's up, scroll to next video
    try:
        body = driver.find_element("tag name", "body")
        body.send_keys(Keys.ARROW_DOWN)
    except Exception:
        pass  # If scroll fails, continue anyway
    
    await asyncio.sleep(0.3)


async def main():
    """
    Main execution flow:
    1. Show GUI and get user preference
    2. Initialize all processing components
    3. Open YouTube Shorts
    4. Monitor URLs and process each video through the pipeline
    """
    user_preference = get_user_preference_from_gui()
    
    if not user_preference:
        print("[Error] No user preference provided. Exiting.")
        return
    
    try:
        metadata_extractor = YouTubeMetadataExtractor()
        embedding_store = VideoEmbeddingStore()
        decision_system = VideoDecisionSystem()
    except Exception as e:
        print(f"[Error] Failed to initialize processing components: {e}")
        import traceback
        traceback.print_exc()
        return
    
    driver = None
    try:
        chrome_options = Options()
        chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.maximize_window()
        driver.get("https://www.youtube.com/shorts")
        time.sleep(3)
        
    except Exception as e:
        print(f"[Error] Failed to connect to Chrome browser: {e}")
        print("Make sure Chrome is running with remote debugging enabled:")
        print('  /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222')
        return
    
    # State tracking
    seen_shorts = set()
    current_video_data = None
    processed_videos = {}  # Track videos we've already processed with LLM
    classified_videos = set()  # Track videos we've already added to collections
    active_monitoring_task = None  # Track the current monitoring task
    
    # Initialize hand gesture controller
    gesture_controller = HandGestureController(show_video=True)
    gesture_controller.start()
    print("\n[Gesture Control Enabled]")
    print("Hand Gestures:")
    print("  - Swipe DOWN with hand to scroll to next video")
    print("  - Swipe UP with hand to scroll to previous video")
    print("Head Pose:")
    print("  - Nod DOWN (look down then return to neutral) to scroll to next video")
    print("  - Nod UP (look up then return to neutral) to scroll to previous video")
    print("- Press 'q' in the tracking window to quit")
    
    # Configure window size and position (small window in bottom right)
    window_name = "Gesture Tracking"
    window_width = 320  # Small width
    window_height = 240  # Small height
    
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, window_width, window_height)
        
        # Position window in bottom right corner with safe margins
        # Assuming common screen widths (adjust if needed for your screen)
        cv2.moveWindow(window_name, 1200, 500)  # Bottom right with margin
    except Exception as e:
        print(f"[Warning] Could not configure CV window position: {e}")
        print("  Window will appear in default position")
    
    try:
        while True:
            current_url = driver.current_url
            
            # Display camera feed (must be in main thread for macOS)
            try:
                with gesture_controller.frame_lock:
                    if gesture_controller.latest_frame is not None:
                        # Resize frame to fit small window
                        small_frame = cv2.resize(gesture_controller.latest_frame, (window_width, window_height))
                        cv2.imshow(window_name, small_frame)
            except Exception as e:
                # If display fails, just continue without it
                pass
            
            # Check for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[User pressed 'q' - quitting...]")
                break
            
            # Check for hand gestures
            gesture = gesture_controller.get_gesture()
            if gesture:
                try:
                    body = driver.find_element("tag name", "body")
                    if gesture == 'DOWN':
                        body.send_keys(Keys.ARROW_DOWN)
                        print("[Gesture] Scrolling DOWN")
                    elif gesture == 'UP':
                        body.send_keys(Keys.ARROW_UP)
                        print("[Gesture] Scrolling UP")
                except Exception as e:
                    print(f"[Gesture] Scroll failed: {e}")
            
            # Case 1: URL changed (previous video ended or user manually scrolled)
            if current_video_data and current_url != current_video_data['url']:
                video_url = current_video_data['url']
                
                # Only classify if we haven't classified this video before
                if video_url not in classified_videos:
                    # Calculate watch time and classify previous video
                    watch_time = time.time() - current_video_data['start_time']
                    classify_and_add_to_collection(current_video_data, watch_time, embedding_store)
                    classified_videos.add(video_url)  # Mark as classified
                else:
                    # Already classified - just report watch time
                    watch_time = time.time() - current_video_data['start_time']
                    print(f"[Revisit] Already classified - watched {watch_time:.1f}s this time")
                
                current_video_data = None
                
                # Cancel the previous monitoring task if it's still running
                if active_monitoring_task and not active_monitoring_task.done():
                    active_monitoring_task.cancel()
                    try:
                        await active_monitoring_task
                    except asyncio.CancelledError:
                        pass  # Expected when cancelling
                    active_monitoring_task = None
            
            # Case 2: Video detected (new or revisited)
            if '/shorts/' in current_url and (current_video_data is None or current_video_data['url'] != current_url):
                
                is_revisit = False
                
                # Check if we've already processed this video with LLM
                if current_url in processed_videos:
                    # Reuse previous LLM decision
                    decision, reasoning, metadata = processed_videos[current_url]
                    
                    # Check if this is a revisit (user went back)
                    if current_url in seen_shorts:
                        is_revisit = True
                        print(f"\n[Revisiting] Previous decision: {decision} - {reasoning}")
                        print(f"[Manual control - will not auto-scroll]")
                    else:
                        seen_shorts.add(current_url)
                        
                elif current_url not in seen_shorts:
                    # New video - get LLM decision
                    seen_shorts.add(current_url)
                    
                    decision, reasoning, metadata = await process_video(
                        current_url,
                        user_preference,
                        metadata_extractor,
                        embedding_store,
                        decision_system
                    )
                    
                    # Store for future reference
                    if decision and metadata:
                        processed_videos[current_url] = (decision, reasoning, metadata)
                else:
                    # Already seen but not in processed_videos (shouldn't happen but handle it)
                    decision, reasoning, metadata = None, None, None
                
                if decision and metadata:
                    # Start tracking this video
                    current_video_data = {
                        'url': current_url,
                        'metadata': metadata,
                        'start_time': time.time(),
                        'duration': metadata.get('duration', 10),
                        'decision': decision
                    }
                    
                    if not is_revisit:
                        print(f"Decision: {decision}, Reasoning: {reasoning}")
                        if decision == "STAY":
                            vid_duration = metadata.get('duration', 10)
                            print(f"  ⏸️  Will auto-scroll after {vid_duration} seconds (full duration)")
                        else:
                            print(f"  ⏩  Will auto-scroll after 1.0 second")
                    
                    # Always create monitoring task, but pass is_revisit flag
                    active_monitoring_task = asyncio.create_task(
                        watch_video_with_monitoring(driver, current_video_data, is_revisit)
                    )
            
            await asyncio.sleep(0.1)  # Check URL changes even more frequently
            
    except KeyboardInterrupt:
        print("\n[Shutting down...]")
    except Exception as e:
        print(f"\n[Error]: {e}")
        import traceback
        traceback.print_exc()
    finally:
        gesture_controller.stop()
        cv2.destroyAllWindows()
        if driver:
            try:
                driver.quit()
            except Exception as e:
                print(f"[Warning] Error closing driver: {e}")
        print("[Cleanup complete]")


if __name__ == "__main__":
    asyncio.run(main())
