import cv2
import argparse
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO


def track_objects(video_path):
    video_path = Path(video_path)

    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return

    # Load YOLO model
    model = YOLO("yolo11n.pt")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "tracked_video.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    frame_number = 0

    # Store unique tracking IDs by object class
    unique_objects = defaultdict(set)

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # persist=True keeps tracking IDs between frames
        results = model.track(
            frame,
            persist=True,
            conf=0.5,
            verbose=False
        )

        result = results[0]

        # Draw YOLO tracking boxes
        annotated_frame = result.plot()

        if result.boxes is not None and result.boxes.id is not None:

            track_ids = (
                result.boxes.id
                .int()
                .cpu()
                .tolist()
            )

            class_ids = (
                result.boxes.cls
                .int()
                .cpu()
                .tolist()
            )

            for track_id, class_id in zip(
                track_ids,
                class_ids
            ):
                object_name = model.names[class_id]

                unique_objects[object_name].add(
                    track_id
                )

        cv2.putText(
            annotated_frame,
            f"Frame: {frame_number}/{total_frames}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        writer.write(annotated_frame)

        if frame_number % 30 == 0:
            print(
                f"Tracking frame "
                f"{frame_number}/{total_frames}"
            )

    cap.release()
    writer.release()

    print("\n--- UNIQUE OBJECT SUMMARY ---")

    total_unique = 0

    for object_name, track_ids in unique_objects.items():
        count = len(track_ids)
        total_unique += count

        print(f"{object_name}: {count}")

    print(f"\nTotal unique tracked objects: {total_unique}")
    print(f"Tracked video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO Object Tracking System"
    )

    parser.add_argument(
        "video",
        help="Path to video file"
    )

    args = parser.parse_args()

    track_objects(args.video)


if __name__ == "__main__":
    main()