import os
import argparse
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import sqlite3
from datetime import datetime

def process_videos(folder_path, output_folder, weights, confidence):
    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Define the SQLite database path
    db_path = os.path.join(output_folder, 'zulu_tracking_data_plot2.db')

    # Connect to the SQLite database (it will be created if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create a table to store tracking data with the new structure
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tracking_data (
            Frame INTEGER,
            Plot TEXT,
            Timestamps TEXT,
            Video_Index INTEGER
        )
    ''')

    # Load the YOLOv8 model
    model = YOLO(weights)

    # Get all video files from the folder
    video_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mkv'))]

    current_video_index = 0

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Loop through the video files
    while current_video_index < len(video_paths):
        # Reset tracking IDs
        track_history.clear()

        # Open the video file
        video_path = video_paths[current_video_index]
        cap = cv2.VideoCapture(video_path)

        # Check if the video file opened successfully
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            current_video_index += 1
            continue

        # Initialize variables for tracking frame information
        current_frame = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define the codec and create VideoWriter object with the correct frame size
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_video_path = os.path.join(output_folder, f'output_video_{current_video_index}.avi')
        output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        if current_video_index != 0:
            # Print the max car ID from the previous video
            print("Max Car ID from previous video: ", max(track_ids))
            last_video_last_car_id = max(track_ids)
        else:
            print("First video, no previous car ID")
            last_video_last_car_id = 0

        prev_time = datetime.now()

        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                try:
                    # Increment current frame count
                    current_frame += 1

                    # Run YOLOv8 tracking on the frame, persisting tracks between frames
                    start_time = datetime.now()
                    results = model.track(frame, persist=True, classes=2, iou=0.7, conf=confidence, half=True, device='cuda:0')
                    end_time = datetime.now()

                    # Check if detections are present
                    if results and results[0].boxes.id is not None:
                        # Get the boxes and track IDs
                        boxes = results[0].boxes.xywh.cpu()
                        track_ids = results[0].boxes.id.int().cpu().tolist()

                        # Initialize a list to store the bounding box coordinates
                        bounding_boxes = []

                        # Visualize the results on the frame
                        annotated_frame = results[0].plot(labels=False, conf=False)

                        # Plot the tracks and extract bounding box coordinates
                        for box, track_id in zip(boxes, track_ids):
                            x, y, w, h = box
                            x1 = int(x - w / 2)
                            y1 = int(y - h / 2)
                            x2 = int(x + w / 2)
                            y2 = int(y + h / 2)
                            bounding_boxes.append((x1, y1, x2, y2))

                            if current_video_index != 0:
                                # Subtract last_video_last_car_id from the current track_id
                                if track_id > last_video_last_car_id:
                                    track_id = track_id - last_video_last_car_id

                            track = track_history[track_id]
                            track.append((float(x), float(y)))  # x, y center point
                            if len(track) > 20:  # Retain 20 tracks for 20 frames
                                track.pop(0)

                            # Calculate elapsed time in Zulu (UTC) time
                            elapsed_time_zulu = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Including milliseconds

                            # Format the plot as Car_ID(X1,Y1,X2,Y2)
                            plot = f"{track_id}({x1},{y1},{x2},{y2})"

                            # Write the data to the SQLite database with the new structure
                            cursor.execute('''
                                INSERT INTO tracking_data (Frame, Plot, Timestamps, Video_Index)
                                VALUES (?, ?, ?, ?)
                            ''', (current_frame, plot, elapsed_time_zulu, current_video_index))

                            # Commit the changes after each frame
                            conn.commit()

                            # Draw the tracking lines
                            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                            cv2.polylines(
                                annotated_frame,
                                [points],
                                isClosed=False,
                                color=(120, 230, 230),
                                thickness=10,
                            )

                            # Display information on the bounding box
                            info_text = f'Coordinates: ({x1},{y1}),({x2},{y2})'
                            cv2.putText(annotated_frame, info_text, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                            info_text = f'Frame: {current_frame}/{total_frames}'
                            cv2.putText(annotated_frame, info_text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                            info_text = f'Car ID: {track_id}'
                            cv2.putText(annotated_frame, info_text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 192, 203), 2)

                            # Calculate current Zulu time
                            current_zulu_time = datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]  # Including milliseconds

                            # Display Zulu time on the annotated frame
                            info_text = f'Zulu Time: {current_zulu_time}'
                            cv2.putText(annotated_frame, info_text, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            # Calculate FPS
                            processing_time = (end_time - start_time).total_seconds()
                            if processing_time > 0:
                                fps = 1 / processing_time
                            else:
                                fps = 0

                            # Draw a rectangle for FPS background
                            cv2.rectangle(annotated_frame, (10, 10), (180, 50), (0, 0, 0), -1)

                            # Display FPS on the top left corner
                            fps_text = f'FPS: {fps:.2f}'
                            cv2.putText(annotated_frame, fps_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                        # Write the frame with bounding boxes to the output video
                        output_video.write(annotated_frame)

                        # Display the annotated frame
                        cv2.imshow("YOLOv8 Tracking", annotated_frame)

                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except Exception as e:
                    print("Error processing frame:", e)
                    continue
            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object
        cap.release()

        # Move to the next video
        current_video_index += 1

        # Release the output video object
        output_video.release()

    # Commit the changes and close the database connection
    conn.commit()
    conn.close()

    # Close the display window
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process videos with YOLOv8 tracking.')
    parser.add_argument('--folder_path', required=True, help='Path to the folder containing video files.')
    parser.add_argument('--output_folder', required=True, help='Path to the folder to save processed videos and database.')
    parser.add_argument('--weights', default=r"D:\RaceProject\project\yolov8x.pt", help='Path to the YOLOv8 weights file.')
    parser.add_argument('--confidence', type=float, default=0.3, help='Confidence threshold for YOLOv8.')

    args = parser.parse_args()

    process_videos(args.folder_path, args.output_folder, args.weights, args.confidence)
