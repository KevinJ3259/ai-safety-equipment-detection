from pathlib import Path
import shutil

SOURCE = Path("dataset_import")
DEST = Path("dataset")

SPLITS = {
    "train": "train",
    "valid": "val",
    "test": "test",
}

# Original Roboflow classes:
# 0 = unwanted blank class
# 1 = hard-hat
# 2 = safety-vest
#
# New classes:
# 0 = hard-hat
# 1 = safety-vest

CLASS_MAP = {
    1: 0,
    2: 1,
}

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]


def prepare_split(source_split, dest_split):
    source_images = SOURCE / source_split / "images"
    source_labels = SOURCE / source_split / "labels"

    dest_images = DEST / "images" / dest_split
    dest_labels = DEST / "labels" / dest_split

    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    images_copied = 0
    annotations_kept = 0
    annotations_removed = 0

    for label_file in source_labels.glob("*.txt"):
        new_lines = []

        for line in label_file.read_text().splitlines():
            parts = line.split()

            if not parts:
                continue

            old_class = int(parts[0])

            if old_class not in CLASS_MAP:
                annotations_removed += 1
                continue

            parts[0] = str(CLASS_MAP[old_class])
            new_lines.append(" ".join(parts))
            annotations_kept += 1

        # Only keep images that still have useful annotations
        if not new_lines:
            continue

        image_file = None

        for extension in IMAGE_EXTENSIONS:
            candidate = source_images / f"{label_file.stem}{extension}"

            if candidate.exists():
                image_file = candidate
                break

        if image_file is None:
            print(f"WARNING: No image found for {label_file.name}")
            continue

        shutil.copy2(image_file, dest_images / image_file.name)

        output_label = dest_labels / label_file.name
        output_label.write_text("\n".join(new_lines) + "\n")

        images_copied += 1

    print(f"\n{dest_split.upper()}")
    print(f"Images copied: {images_copied}")
    print(f"Annotations kept: {annotations_kept}")
    print(f"Class-0 annotations removed: {annotations_removed}")


def main():
    for source_split, dest_split in SPLITS.items():
        prepare_split(source_split, dest_split)

    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()