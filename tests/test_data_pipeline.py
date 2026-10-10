from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.data.annotations.scan import scan_annotations
from src.data.discovery import discover_dataset
from src.data.images import load_rgb_image, validate_image


def _make_image(path: Path, size: tuple[int, int] = (100, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(120, 140, 160)).save(path)


def test_discovery_reports_missing_directory(tmp_path: Path) -> None:
    missing = discover_dataset("BDD100K", tmp_path / "missing")
    assert not missing.root_exists
    assert missing.image_count == 0
    assert missing.annotation_file_count == 0
    assert "does not exist" in missing.scan_errors[0]


def test_discovery_counts_images_and_candidate_annotation_formats(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _make_image(root / "images" / "road.jpg")
    (root / "labels").mkdir(parents=True)
    (root / "labels" / "labels.json").write_text("[]", encoding="utf-8")
    (root / "labels" / "road.xml").write_text("<annotation />", encoding="utf-8")
    discovery = discover_dataset("BDD100K", root)
    assert discovery.root_exists
    assert discovery.image_count == 1
    assert discovery.annotation_file_count == 2
    assert discovery.annotation_format_counts == {"JSON": 1, "XML": 1}


def test_image_validation_and_bad_image_error(tmp_path: Path) -> None:
    image_path = tmp_path / "valid.png"
    _make_image(image_path)
    valid, size, error = validate_image(image_path)
    assert valid and size == (100, 80) and error is None
    assert load_rgb_image(image_path).mode == "RGB"
    bad = tmp_path / "bad.jpg"
    bad.write_text("not an image", encoding="utf-8")
    valid, size, error = validate_image(bad)
    assert not valid and size is None and "Could not read" in (error or "")


def test_bdd_json_parses_categories_and_matches_image(tmp_path: Path) -> None:
    root = tmp_path / "bdd100k"
    image_path = root / "images" / "train" / "road.jpg"
    _make_image(image_path)
    label_path = root / "labels" / "det_train.json"
    label_path.parent.mkdir(parents=True)
    label_path.write_text(json.dumps([
        {
            "name": "images/train/road.jpg",
            "labels": [
                {"category": "car", "box2d": {"x1": 10, "y1": 12, "x2": 60, "y2": 55}},
                {"category": "motor", "box2d": {"x1": 61, "y1": 5, "x2": 90, "y2": 40}},
                {"category": "person", "box2d": {"x1": 2, "y1": 3, "x2": 8, "y2": 20}},
            ],
        },
        {"name": "empty.jpg", "labels": []},
    ]), encoding="utf-8")
    discovery = discover_dataset("BDD100K", root)
    scan = scan_annotations(discovery)
    assert scan.parsed_formats == {"BDD100K detection JSON": 1}
    assert scan.is_matched(image_path)
    assert len(scan.boxes_for(image_path)) == 2
    assert {box.vehicle_class for box in scan.boxes_for(image_path)} == {"car", "motorcycle"}
    assert scan.unmapped_categories["person"] == 1
    assert any("empty.jpg" in issue for issue in scan.issues)


def test_bdd_frames_objects_format_filters_nonvehicles_and_matches_image(tmp_path: Path) -> None:
    root = tmp_path / "bdd100k"
    image_path = root / "images" / "train" / "road.jpg"
    _make_image(image_path)
    labels = root / "labels" / "road.json"
    labels.parent.mkdir(parents=True)
    labels.write_text(json.dumps({
        "name": "images/train/road.jpg",
        "frames": [{
            "objects": [
                {"category": "car", "box2d": {"x1": 10, "y1": 12, "x2": 60, "y2": 55}},
                {"category": "traffic sign", "box2d": {"x1": 2, "y1": 3, "x2": 8, "y2": 20}},
            ],
        }],
    }), encoding="utf-8")

    scan = scan_annotations(discover_dataset("BDD100K", root))
    assert scan.parsed_formats == {"BDD100K detection JSON": 1}
    assert scan.is_matched(image_path)
    assert scan.has_valid_match(image_path)
    assert len(scan.boxes_for(image_path)) == 1
    assert scan.boxes_for(image_path)[0].vehicle_class == "car"
    assert scan.unmapped_categories["traffic sign"] == 1


def test_bdd_frame_missing_vehicle_box_and_multiframe_record_are_invalid(tmp_path: Path) -> None:
    root = tmp_path / "bdd100k"
    image_path = root / "images" / "road.jpg"
    _make_image(image_path)
    labels = root / "labels" / "road.json"
    labels.parent.mkdir(parents=True)
    labels.write_text(json.dumps({
        "name": "road.jpg",
        "frames": [{"objects": [{"category": "car"}]}],
    }), encoding="utf-8")
    scan = scan_annotations(discover_dataset("BDD100K", root))
    assert scan.is_invalid(image_path)
    assert any("has no box2d" in issue for issue in scan.issues)

    labels.write_text(json.dumps({
        "name": "road.jpg",
        "frames": [{"objects": []}, {"objects": []}],
    }), encoding="utf-8")
    scan = scan_annotations(discover_dataset("BDD100K", root))
    assert scan.is_invalid(image_path)
    assert any("exactly one frame" in issue for issue in scan.issues)


def test_annotation_xyxy_rejects_negative_origin() -> None:
    from src.data.annotations.common import parse_xyxy

    with pytest.raises(ValueError, match="nonnegative"):
        parse_xyxy((-1, 0, 10, 10))


def test_matched_image_with_malformed_label_is_withheld_from_evaluation(tmp_path: Path) -> None:
    root = tmp_path / "bdd"
    image_path = root / "images" / "bad.jpg"
    _make_image(image_path)
    labels = root / "labels" / "ann.json"
    labels.parent.mkdir(parents=True)
    labels.write_text(json.dumps([{
        "name": "bad.jpg",
        "labels": [{"category": "car", "box2d": {"x1": 30, "y1": 10, "x2": 5, "y2": 50}}],
    }]), encoding="utf-8")
    scan = scan_annotations(discover_dataset("BDD100K", root))
    assert scan.is_matched(image_path)
    assert scan.is_invalid(image_path)
    assert not scan.has_valid_match(image_path)
    assert scan.boxes_for(image_path) == []


def test_bdd_jsonl_is_discovered_and_streamed(tmp_path: Path) -> None:
    root = tmp_path / "bdd"
    image_path = root / "images" / "road.jpeg"
    _make_image(image_path)
    labels = root / "labels" / "ann.jsonl"
    labels.parent.mkdir(parents=True)
    records = [
        {"name": "road.jpeg", "labels": [{"category": "truck", "box2d": {"x1": 2, "y1": 2, "x2": 40, "y2": 30}}]},
        {"name": "empty.jpeg", "labels": []},
    ]
    labels.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
    discovery = discover_dataset("BDD100K", root)
    assert discovery.annotation_format_counts == {"JSONL": 1}
    scan = scan_annotations(discovery)
    assert scan.parsed_formats == {"BDD100K detection JSON": 1}
    assert scan.is_matched(image_path)
    assert scan.boxes_for(image_path)[0].vehicle_class == "truck"
    assert "empty.jpeg" in scan.unmatched_annotation_names


def test_coco_json_xywh_and_category_mapping(tmp_path: Path) -> None:
    root = tmp_path / "idd"
    image_path = root / "images" / "frame.png"
    _make_image(image_path)
    annotation_path = root / "annotations" / "instances.json"
    annotation_path.parent.mkdir(parents=True)
    annotation_path.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "frame.png", "width": 100, "height": 80}],
        "categories": [{"id": 7, "name": "truck"}, {"id": 8, "name": "person"}],
        "annotations": [
            {"image_id": 1, "category_id": 7, "bbox": [5, 10, 30, 20]},
            {"image_id": 1, "category_id": 8, "bbox": [1, 1, 4, 4]},
        ],
    }), encoding="utf-8")
    scan = scan_annotations(discover_dataset("IDD", root))
    boxes = scan.boxes_for(image_path)
    assert scan.parsed_formats == {"COCO JSON": 1}
    assert scan.is_matched(image_path)
    assert [item.box_xyxy for item in boxes if item.vehicle_class == "truck"] == [(5.0, 10.0, 35.0, 30.0)]
    assert scan.unmapped_categories["person"] == 1


def test_pascal_voc_and_yolo_txt_readers(tmp_path: Path) -> None:
    root = tmp_path / "idd"
    image_path = root / "images" / "train" / "car.jpg"
    _make_image(image_path, (200, 100))
    xml_path = root / "voc" / "car.xml"
    xml_path.parent.mkdir(parents=True)
    xml_path.write_text(
        "<annotation><filename>car.jpg</filename><object><name>bus</name><bndbox>"
        "<xmin>2</xmin><ymin>3</ymin><xmax>50</xmax><ymax>60</ymax>"
        "</bndbox></object></annotation>",
        encoding="utf-8",
    )
    yolo_dir = root / "labels" / "train"
    yolo_dir.mkdir(parents=True)
    (root / "classes.txt").write_text("car\nmotorcycle\n", encoding="utf-8")
    (yolo_dir / "car.txt").write_text("0 0.5 0.5 0.5 0.4\n1 0.25 0.5 0.2 0.2\n", encoding="utf-8")
    discovery = discover_dataset("IDD", root)
    scan = scan_annotations(discovery)
    assert "Pascal VOC XML" in scan.parsed_formats
    assert "YOLO TXT" in scan.parsed_formats
    boxes = scan.boxes_for(image_path)
    assert {item.vehicle_class for item in boxes} == {"bus", "car", "motorcycle"}
    assert any(item.box_xyxy == (50.0, 30.0, 150.0, 70.0) for item in boxes)


def test_malformed_annotation_is_reported_not_silently_dropped(tmp_path: Path) -> None:
    root = tmp_path / "idd"
    _make_image(root / "images" / "road.jpg")
    xml = root / "annotations" / "broken.xml"
    xml.parent.mkdir(parents=True)
    xml.write_text("<annotation><object>", encoding="utf-8")
    scan = scan_annotations(discover_dataset("IDD", root))
    assert str(xml) in scan.unsupported_files
    assert any("invalid XML" in issue for issue in scan.issues)


def test_full_image_validation_reports_unreadable_files(tmp_path: Path) -> None:
    from src.data.images import validate_image_paths

    valid = tmp_path / "valid.jpg"
    _make_image(valid)
    invalid = tmp_path / "invalid.jpg"
    invalid.write_text("not image data", encoding="utf-8")
    report = validate_image_paths((valid, invalid))
    assert report.checked_count == 2
    assert report.readable_count == 1
    assert len(report.unreadable) == 1
    assert report.unreadable[0][0] == str(invalid)
