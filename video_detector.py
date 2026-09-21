import cv2
import time
import argparse
from pathlib import Path
from ultralytics import YOLO


def process_video(video_path):
    video_path = Path(video_path)

    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return

    print(f"Opening video: {video_path}")

    # Load YOLO model
    model = YOLO("yolo11n.pt")

    # Open video with OpenCV
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Resolution: {width}x{height}")
    print(f"Source FPS: {source_fps:.2f}")
    print(f"Total frames: {total_frames}")

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "detected_video.mp4"

    # MP4 video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        source_fps,
        (width, height)
    )

    frame_number = 0
    total_detections = 0
    start_time = time.time()

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # Run YOLO directly on the OpenCV frame
        results = model(
            frame,
            conf=0.5,
            verbose=False
        )

        result = results[0]

        # YOLO draws the bounding boxes and labels
        annotated_frame = result.plot()

        detections_this_frame = len(result.boxes)
        total_detections += detections_this_frame

        # Calculate processing FPS
        elapsed = time.time() - start_time
        processing_fps = (
            frame_number / elapsed if elapsed > 0 else 0
        )

        # Add our own OpenCV text overlay
        cv2.putText(
            annotated_frame,
            f"Processing FPS: {processing_fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Frame: {frame_number}/{total_frames}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        # Write processed frame to our output video
        writer.write(annotated_frame)

        if frame_number % 30 == 0:
            print(
                f"Processed {frame_number}/{total_frames} frames "
                f"| FPS: {processing_fps:.1f}"
            )

    cap.release()
    writer.release()

    elapsed = time.time() - start_time
    average_fps = frame_number / elapsed if elapsed > 0 else 0

    print("\n--- VIDEO PROCESSING COMPLETE ---")
    print(f"Frames processed: {frame_number}")
    print(f"Frame-level detections: {total_detections}")
    print(f"Processing time: {elapsed:.2f} seconds")
    print(f"Average processing FPS: {average_fps:.2f}")
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="OpenCV + YOLO Video Object Detector"
    )

    parser.add_argument(
        "video",
        help="Path to the video file"
    )

    args = parser.parse_args()

    process_video(args.video)


if __name__ == "__main__":
    main()