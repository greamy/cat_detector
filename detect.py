import cv2
import subprocess
import time
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

# Stream URLs
RTSP_SOURCE_URL = 'rtsp://greamy:catstream@192.168.2.1:8554/cam-high'
RTSP_DEST_URL = 'rtsp://greamy:catstream@192.168.2.1:8554/cam-processed'

# Detection resolution (lower = faster processing)
# The stream will be downscaled to this resolution for detection only
# Output stream will maintain original resolution from source
DETECTION_WIDTH = 1920
DETECTION_HEIGHT = 1080

# YOLO model and detection settings
YOLO_MODEL = 'yolov8l.pt'
YOLO_IMGSZ = 640  # Image size for YOLO inference

# Detection classes to display (COCO dataset classes)
# Uncomment/comment classes you want to detect
DETECTED_CLASSES = [
    'person',
    # 'bicycle',
    # 'car',
    # 'motorcycle',
    # 'airplane',
    # 'bus',
    # 'train',
    # 'truck',
    # 'boat',
    # 'traffic light',
    # 'fire hydrant',
    # 'stop sign',
    # 'parking meter',
    # 'bench',
    #'bird',
    'cat',
    #'dog',
    # 'horse',
    # 'sheep',
    # 'cow',
    # 'elephant',
    # 'bear',
    # 'zebra',
    # 'giraffe',
      'backpack',
      'umbrella',
      'handbag',
    # 'tie',
    # 'suitcase',
    # 'frisbee',
    # 'skis',
    # 'snowboard',
    # 'sports ball',
    # 'kite',
    # 'baseball bat',
    # 'baseball glove',
    # 'skateboard',
    # 'surfboard',
    # 'tennis racket',
    # 'bottle',
    # 'wine glass',
      'cup',
      'fork',
      'knife',
      'spoon',
      'bowl',
      'banana',
      'apple',
      'sandwich',
      'orange',
    # 'broccoli',
    # 'carrot',
    # 'hot dog',
      'pizza',
    # 'donut',
    # 'cake',
    # 'chair',
    # 'couch',
    # 'potted plant',
    # 'bed',
    # 'dining table',
    # 'toilet',
    # 'tv',
      'laptop',
    # 'mouse',
      'remote',
    # 'keyboard',
      'cell phone',
    # 'microwave',
    # 'oven',
    # 'toaster',
    # 'sink',
    # 'refrigerator',
    # 'book',
    # 'clock',
    # 'vase',
    # 'scissors',
    # 'teddy bear',
    # 'hair drier',
    # 'toothbrush'
]

# Performance settings
PROCESS_EVERY_N_FRAMES = 3  # Process every Nth frame with YOLO (2 = every other frame)

MIN_CONFIDENCE = 0.5  # Minimum confidence threshold for detections (0.0 to 1.0)

# Color settings for each class (BGR format)
CLASS_COLORS = {
    'person': (255, 0, 0),      # Blue
    'cat': (0, 255, 0),         # Green
    'backpack': (10, 10, 10),
    'umbrella': (150, 150, 255),
    'handbag': (100, 255, 50),
    'cup': (255, 255, 255),
    'fork': (100, 100, 100),
    'knife': (100, 100, 100),
    'spoon': (100, 100, 100),
    'bowl': (255, 255, 255),
    'banana': (0, 255, 255),
    'apple': (0, 0, 255),
    'sandwich': (140, 180, 200),
    'orange': (0, 165, 255),
    'pizza': (50, 200, 255),
    'laptop': (200, 200, 200),
    'remote': (200, 200, 200),
    'cell phone': (200, 200, 200)
    # Add more colors as needed for other classes
}

# Reconnection settings
RETRY_DELAY = 3  # Seconds to wait before retry
MAX_CONSECUTIVE_FAILURES = 5  # Max failed frame reads before reconnecting

# Statistics settings
STATS_INTERVAL = 100  # Print stats every N frames

# ============================================================
# INITIALIZATION
# ============================================================

print("Loading YOLO model...")
model = YOLO(YOLO_MODEL)

# Build dictionary of class IDs for detected classes
DETECTED_CLASS_IDS = {}
for class_name in DETECTED_CLASSES:
    if class_name in model.names.values():
        class_id = list(model.names.keys())[list(model.names.values()).index(class_name)]
        DETECTED_CLASS_IDS[class_id] = class_name

print(f"Model loaded. Detecting classes: {', '.join(DETECTED_CLASSES)}")
print(f"Class IDs: {DETECTED_CLASS_IDS}")
print(f"Processing every {PROCESS_EVERY_N_FRAMES} frame(s) with YOLO\n")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_capture():
    """Create and return a video capture object"""
    print(f"Connecting to source stream: {RTSP_SOURCE_URL}")
    cap = cv2.VideoCapture(RTSP_SOURCE_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    
    if not cap.isOpened():
        raise Exception("Could not open source video stream")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 15  # Default to 15 if FPS not detected
    
    print(f"Source stream opened: {width}x{height} @ {fps}fps")
    print(f"Detection will run at: {DETECTION_WIDTH}x{DETECTION_HEIGHT}")
    
    return cap, width, height, fps

def create_ffmpeg_process(width, height, fps):
    """Create and return FFmpeg subprocess for streaming"""
    command = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-pix_fmt', 'yuv420p',
        '-profile:v', 'baseline',
        '-level', '4.0',
        '-g', str(fps * 2),
        '-bf', '0',
        '-refs', '1',
        '-sc_threshold', '0',
        '-f', 'rtsp',
        '-rtsp_transport', 'tcp',
        RTSP_DEST_URL
    ]
    
    print("Starting FFmpeg process...")
    # Redirect all FFmpeg output to devnull to keep console clean
    return subprocess.Popen(command, stdin=subprocess.PIPE, 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)

def draw_detections(frame, detections, scale_x, scale_y):
    """Draw detection boxes on frame"""
    for detection in detections:
        x1, y1, x2, y2, confidence, class_name = detection
        
        # Scale coordinates to frame size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

       # Get color for this class (default to green if not specified)
        color = CLASS_COLORS.get(class_name, (0, 255, 0))
        
        # Draw rectangle and label
        cv2.rectangle(frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), 
                    color, 2)
        cv2.putText(frame, f'{class_name} {confidence}', (x1_scaled, y1_scaled - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def cleanup(cap, proc):
    """Clean up resources"""
    if cap is not None:
        cap.release()
    if proc is not None:
        try:
            proc.stdin.close()
            proc.wait(timeout=5)
        except:
            proc.kill()

# ============================================================
# MAIN LOOP
# ============================================================

print("Starting multi-class detection stream processor...")
print("Press Ctrl+C to stop\n")

retry_count = 0

while True:
    cap = None
    proc = None
    frame_count = 0
    consecutive_failures = 0
    start_time = None
    last_stats_time = None
    last_stats_frame = 0
    last_detections = []  # Store detections from last processed frame
    
    try:
        # Initialize capture and FFmpeg
        cap, width, height, fps = create_capture()
        proc = create_ffmpeg_process(width, height, fps)
        
        # Calculate scaling factors
        scale_x = width / DETECTION_WIDTH
        scale_y = height / DETECTION_HEIGHT
        
        print(f"Processing started (attempt {retry_count + 1})\n")
        retry_count = 0  # Reset retry count on successful connection
        
        start_time = time.time()
        last_stats_time = start_time
        
        # Main processing loop
        while True:
            ret, frame = cap.read()
            
            if not ret:
                consecutive_failures += 1
                
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"\nToo many consecutive failures ({MAX_CONSECUTIVE_FAILURES}), reconnecting...")
                    break
                
                time.sleep(0.1)
                continue
            
            # Reset failure counter on successful read
            consecutive_failures = 0
            frame_count += 1
            
            # Process with YOLO every Nth frame
            if frame_count % PROCESS_EVERY_N_FRAMES == 1:
                # Downscale frame for detection
                detection_frame = cv2.resize(frame, (DETECTION_WIDTH, DETECTION_HEIGHT))
                
                # Run detection on downscaled frame
                results = model(detection_frame, stream=True, verbose=False, imgsz=YOLO_IMGSZ)
                
                # Clear previous detections and process new results
                last_detections = []
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        class_id = int(box.cls)
                        confidence = float(box.conf[0])
                        # Only store detections for configured classes
                        if class_id in DETECTED_CLASS_IDS and confidence >= MIN_CONFIDENCE:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            confidence_rounded = round(confidence, 2)
                            class_name = DETECTED_CLASS_IDS[class_id]
                            
                            # Store detection (coordinates are in detection frame space)
                            last_detections.append((x1, y1, x2, y2, confidence_rounded, class_name))
            
            # Draw detections on current frame (whether newly detected or carried over)
            draw_detections(frame, last_detections, scale_x, scale_y)
            
            # Send frame to FFmpeg
            try:
                proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                print(f"\nFFmpeg pipe error: {e}")
                break
            
            # Print statistics every STATS_INTERVAL frames
            if frame_count % STATS_INTERVAL == 0:
                current_time = time.time()
                elapsed_total = current_time - start_time
                elapsed_interval = current_time - last_stats_time
                
                # Calculate FPS for this interval
                frames_this_interval = frame_count - last_stats_frame
                current_fps = frames_this_interval / elapsed_interval if elapsed_interval > 0 else 0
                
                # Calculate overall average FPS
                avg_fps = frame_count / elapsed_total if elapsed_total > 0 else 0
                
                # Calculate speed multiplier (processing fps / source fps)
                speed_multiplier = avg_fps / fps if fps > 0 else 0
                
                # Calculate total time processed
                total_time = frame_count / fps if fps > 0 else 0
                
                print(f"frame={frame_count:5d} | fps={current_fps:4.1f} | "
                      f"avg_fps={avg_fps:4.1f} | speed={speed_multiplier:5.3f}x | "
                      f"time={int(total_time//60):02d}:{int(total_time%60):02d} | "
                      f"detections={len(last_detections)}")
                
                last_stats_time = current_time
                last_stats_frame = frame_count
    
    except KeyboardInterrupt:
        print("\n\nReceived keyboard interrupt, shutting down...")
        cleanup(cap, proc)
        break
    
    except Exception as e:
        print(f"\nError occurred: {e}")
        retry_count += 1
    
    finally:
        # Print final statistics
        if start_time and frame_count > 0:
            total_elapsed = time.time() - start_time
            final_avg_fps = frame_count / total_elapsed
            final_speed = final_avg_fps / fps if fps > 0 else 0
            print(f"\nSession ended: {frame_count} frames processed in {total_elapsed:.1f}s")
            print(f"Average FPS: {final_avg_fps:.2f} | Speed: {final_speed:.3f}x\n")
        
        cleanup(cap, proc)
    
    # Wait before retrying
    print(f"Waiting {RETRY_DELAY} seconds before reconnecting...")
    time.sleep(RETRY_DELAY)
    print("Attempting to reconnect...\n")

print("Script finished.")
