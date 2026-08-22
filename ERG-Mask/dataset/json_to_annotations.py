import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load_image(data, json_path):
    if data.get("imageData"):
        image_data = base64.b64decode(data["imageData"])
        return Image.open(io.BytesIO(image_data)).convert("RGB")

    image_path = data.get("imagePath")

    if image_path is None:
        raise ValueError(f"No imageData or imagePath found in {json_path}")

    image_path = Path(json_path).parent / image_path
    return Image.open(image_path).convert("RGB")


def create_label_image(image_size, shapes):
    width, height = image_size
    labels = ["_background_"]

    for shape in shapes:
        label = shape["label"]
        if label not in labels:
            labels.append(label)

    label_to_value = {
        label: index for index, label in enumerate(labels)
    }

    label_image = Image.new("I", (width, height), 0)
    draw = ImageDraw.Draw(label_image)

    for shape in shapes:
        label = shape["label"]
        points = shape["points"]
        shape_type = shape.get("shape_type", "polygon")
        value = label_to_value[label]

        if shape_type == "polygon":
            polygon = [(float(x), float(y)) for x, y in points]
            draw.polygon(polygon, fill=value)

        elif shape_type == "rectangle":
            (x1, y1), (x2, y2) = points
            draw.rectangle(
                [float(x1), float(y1), float(x2), float(y2)],
                fill=value
            )

        else:
            raise ValueError(
                f"Unsupported shape type: {shape_type}"
            )

    return np.array(label_image), labels


def create_edge_image(label_array):
    edge = np.zeros(label_array.shape, dtype=bool)

    edge[1:, :] |= (
        label_array[1:, :] != label_array[:-1, :]
    )

    edge[:-1, :] |= (
        label_array[:-1, :] != label_array[1:, :]
    )

    edge[:, 1:] |= (
        label_array[:, 1:] != label_array[:, :-1]
    )

    edge[:, :-1] |= (
        label_array[:, :-1] != label_array[:, 1:]
    )

    edge &= label_array > 0

    return (edge.astype(np.uint8) * 255)


def create_label_visualization(image, label_array, labels):
    image_array = np.array(image).astype(np.float32)
    image_array = (image_array * 0.45).astype(np.uint8)

    visualization = Image.fromarray(image_array)
    draw = ImageDraw.Draw(visualization)

    edge = create_edge_image(label_array)
    ys, xs = np.where(edge > 0)

    for x, y in zip(xs, ys):
        draw.point((int(x), int(y)), fill=(255, 255, 255))

    return visualization


def process_json(json_path, output_dir=None):
    json_path = Path(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image = load_image(data, json_path)
    shapes = data.get("shapes", [])

    if not shapes:
        raise ValueError(
            f"No annotation shapes found in {json_path}"
        )

    label_array, labels = create_label_image(
        image.size,
        shapes
    )

    if output_dir is None:
        save_dir = json_path.parent / json_path.stem
    else:
        save_dir = Path(output_dir) / json_path.stem

    save_dir.mkdir(parents=True, exist_ok=True)

    image.save(save_dir / "img.png")

    if label_array.max() <= 255:
        label_image = Image.fromarray(
            label_array.astype(np.uint8)
        )
    else:
        label_image = Image.fromarray(
            label_array.astype(np.uint16)
        )

    label_image.save(save_dir / "label.png")

    with open(
        save_dir / "label_names.txt",
        "w",
        encoding="utf-8"
    ) as f:
        for label in labels:
            f.write(label + "\n")

    label_viz = create_label_visualization(
        image,
        label_array,
        labels
    )

    label_viz.save(save_dir / "label_viz.png")

    edge = create_edge_image(label_array)

    Image.fromarray(edge).save(
        save_dir / "edge.png"
    )

    print(f"Converted: {json_path}")
    print(f"Saved to: {save_dir}")
    print(f"Labels: {labels}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input",
        help="LabelMe JSON file or directory"
    )

    parser.add_argument(
        "output",
        nargs="?",
        default=None
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        process_json(
            input_path,
            args.output
        )

    elif input_path.is_dir():
        json_files = sorted(
            input_path.glob("*.json")
        )

        for json_file in json_files:
            try:
                process_json(
                    json_file,
                    args.output
                )
            except Exception as e:
                print(f"Failed: {json_file}")
                print(f"Reason: {e}")

    else:
        raise FileNotFoundError(
            f"Input path does not exist: {input_path}"
        )


if __name__ == "__main__":
    main()
