import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
import math
import time
import streamlit as st
import threading

# Streamlit setup
st.set_page_config(page_title="Fish Feeding Model", layout="wide")

# Initialize tracking history and model
track_history = defaultdict(list)
model = YOLO("best.pt")
names = model.model.names

# IP camera input and output settings
ip_camera_url = "rtsp://admin1:password@192.168.137.170/stream1"
cap = cv2.VideoCapture(ip_camera_url)
assert cap.isOpened(), "Error opening video stream"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
print(f"FPS: {fps}")
st.sidebar.write(f"FPS: {fps}")

result = cv2.VideoWriter("object_tracking.avi",
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         fps,
                         (w, h))

# Variables for fish counting and time tracking
fish_speeds = defaultdict(list)
predict = "feed"
start_time = time.time()
counting_period = 10  # seconds for test feeding
real_feeding_period = 10  # seconds for real feeding
Stop_count = 0  # Initialize Stop_count

# Variables for frame rate adjustment
prev_time = time.time()

seen_track_ids = set()

# Accumulated results
decision_history = []
count_history = []
speed_history = []

# Streamlit app interface
st.title('Fish Feeding Model')
start_button = st.button('Start Feeding')
stop_button = st.button('Stop Feeding')

frame_skip = 2  # Process every 2nd frame
frame_count = 0

# Capture frames in a separate thread
frame = None
frame_lock = threading.Lock()

def capture_frames():
    global frame
    while cap.isOpened():
        success, new_frame = cap.read()
        if success:
            with frame_lock:
                frame = new_frame

if start_button:
    # Start the frame capturing thread
    threading.Thread(target=capture_frames, daemon=True).start()

    # Create columns for layout with the video feed column larger
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    # Create a container for the video feed
    stframe = col1.empty()

    # Initialize the decision header and result containers
    with col2:
        st.subheader("Decision")
        decision_results = st.empty()
    
    # Initialize the decision header and result containers
    with col3:
        st.subheader("Accumulated Fish Count")
        count_results = st.empty()

    with col4:
        st.subheader("Average Speed")
        speed_results = st.empty()

    while cap.isOpened():
        if stop_button:
            st.write("Detection stopped.")
            break

        frame_count += 1
        with frame_lock:
            if frame is None or frame_count % frame_skip != 0:
                continue
            current_frame = frame.copy()

        # Create a copy of the frame for real-time display
        display_frame = current_frame.copy()

        results = model.track(current_frame, persist=True, verbose=False)
        boxes = results[0].boxes.xyxy.cpu()

        frame_fish_count = 0
        avg_speed = "N/A"  # Initialize avg_speed as "N/A"

        if results[0].boxes.id is not None:
            # Extract prediction results
            clss = results[0].boxes.cls.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.float().cpu().tolist()

            # Annotator Init with larger line width
            annotator = Annotator(display_frame, line_width=2)  # Increase the line width here

            unique_fish_in_frame = set(track_ids)  # Unique fish in the current frame
            frame_fish_count = len(unique_fish_in_frame)
            seen_track_ids.update(unique_fish_in_frame)  # Update the set of seen track IDs

            for box, cls, track_id in zip(boxes, clss, track_ids):
                # Draw the bounding box
                annotator.box_label(box, color=colors(int(cls), True), label="")

                # Create label with class name and tracking ID
                label = f"{names[int(cls)]} {track_id}"
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                label_ymin = max(int(box[1]) - label_height - 10, 0)
                label_rect = [(int(box[0]), label_ymin), (int(box[0]) + label_width, int(box[1]))]

                # Draw label background
                cv2.rectangle(display_frame, label_rect[0], label_rect[1], colors(int(cls), True), -1)

                # Draw label text
                cv2.putText(display_frame, label, (int(box[0]), label_ymin + label_height + baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

                # Store tracking history
                track = track_history[track_id]
                current_position = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
                track.append(current_position)
                if len(track) > 30:
                    track.pop(0)

                # Calculate speed (pixels per second)
                if len(track) > 1:
                    dx = track[-1][0] - track[-2][0]
                    dy = track[-1][1] - track[-2][1]
                    distance = math.sqrt(dx**2 + dy**2)
                    speed = (distance * fps)  # pixels per second

                    # Draw speed on the frame
                    speed_label = f"Speed: {speed:.2f} px/s"
                    cv2.putText(display_frame, speed_label, (int(box[0]), label_ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                    
                    # Record speed for averaging
                    fish_speeds[track_id].append(speed)
                    if len(fish_speeds[track_id]) > 30:
                        fish_speeds[track_id].pop(0)

                # Plot tracks
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.circle(display_frame, track[-1], 7, colors(int(cls), True), -1)
                cv2.polylines(display_frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)  # Increase thickness here

            if fish_speeds:
                avg_speed = sum(map(np.mean, fish_speeds.values())) / len(fish_speeds)
                avg_speed = f"{avg_speed:.2f} px/s"

        bottom_margin = 40
        line_spacing = 40

        cv2.putText(display_frame, f"Fish per frame: {frame_fish_count}", (10, h - bottom_margin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Accumulated fish: {len(seen_track_ids)}", (10, h - bottom_margin - line_spacing), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Average Speed: {avg_speed}", (10, h - bottom_margin - 2 * line_spacing), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Convert frame to RGB for displaying in Streamlit
        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        stframe.image(display_frame_rgb, channels="RGB", use_column_width=True)

        curr_time = time.time()
        processing_time = curr_time - prev_time

        sleep_duration = max(0, (1 / fps) - processing_time)
        time.sleep(sleep_duration)
        prev_time = curr_time

        result.write(display_frame)
        elapsed_time = time.time() - start_time
        if elapsed_time >= counting_period:
            avg_speed_value = float(avg_speed.split()[0]) if avg_speed != "N/A" else 0

            if len(seen_track_ids) > 8 and avg_speed_value > 300:
                real_feeding = "High"
                Stop_count = 0
                counting_period = 5
                decision_history.append("High\n")
                count_history.append(str(len(seen_track_ids)) + '\n')
                speed_history.append(str(avg_speed_value) + '\n')
            elif len(seen_track_ids) > 8 or avg_speed_value > 300:
                real_feeding = "Low"
                Stop_count = 0
                counting_period = 5
                decision_history.append("Low\n")
                count_history.append(str(len(seen_track_ids)) + '\n')
                speed_history.append(str(avg_speed_value) + '\n')
            elif len(seen_track_ids) < 8 and avg_speed_value < 300:
                real_feeding = "stop"
                Stop_count += 1
                count_history.append(str(len(seen_track_ids)) + '\n')
                speed_history.append(str(avg_speed_value) + '\n')

                if Stop_count == 1:
                    decision_history.append("First stop feeding decision\n")
                    counting_period = 10
                elif Stop_count == 2:
                    decision_history.append("Second stop feeding decision. STOP!\n")
                    #break  # uncomment this line to stop the detection
                else:
                    decision_history.append("Should stop feeding decision\n")
                    #break  # uncomment this line to stop the detection

            seen_track_ids = set()  # Reset accumulated fish count
            start_time = time.time()

            # Update Streamlit components
            latest_decisions = decision_history[-5:]  # Display only the latest 5 decisions
            latest_counts = count_history[-5:]  # Display only the latest 5 counts
            latest_speeds = speed_history[-5:]  # Display only the latest 5 speeds

            decision_results.markdown('\n'.join(latest_decisions))
            count_results.markdown('\n'.join(latest_counts))
            speed_results.markdown('\n'.join(latest_speeds))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    result.release()
    cap.release()
    cv2.destroyAllWindows()
else:
    st.write("Click the 'Start Feeding' button to begin detection.")
