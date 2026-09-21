from ultralytics import YOLO
from pathlib import Path
import argparse


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def detect_objects(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    extension = file_path.suffix.lower()

    if extension in IMAGE_EXTENSIONS:
        media_type = "image"
    elif extension in VIDEO_EXTENSIONS:
        media_type = "video"
    else:
        print(f"Error: Unsupported file type: {extension}")
        return

    print(f"\nMedia type: {media_type}")
    print(f"Analyzing: {file_path}")

    # Load pretrained YOLO model
    model = YOLO("yolo11n.pt")

    # Run detection
    results = model(
        str(file_path),
        save=True,
        conf=0.5,
        stream=True
    )

    print("\n--- DETECTION SUMMARY ---")

    total_detections = 0
    object_counts = {}

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            object_name = model.names[class_id]

            total_detections += 1

            object_counts[object_name] = (
                object_counts.get(object_name, 0) + 1
            )

    for object_name, count in object_counts.items():
        print(f"{object_name}: {count}")

    print(f"\nTotal detections: {total_detections}")
    print("Detection complete.")


def main():
    parser = argparse.ArgumentParser(
        description="AI Computer Vision Object Detection System"
    )

    parser.add_argument(
        "file",
        help="Path to an image or video"
    )

    args = parser.parse_args()

    detect_objects(args.file)


if __name__ == "__main__":
    main()