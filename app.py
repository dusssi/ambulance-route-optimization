"""Streamlit dashboard for the local CV-to-route academic prototype."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data.annotations.common import AnnotationScan
from src.data.annotations.scan import scan_annotations
from src.data.discovery import DatasetDiscovery, discover_dataset
from src.data.images import load_rgb_image, validate_image_paths
from src.detection.models import DetectionRun
from src.detection.visualization import annotate_image
from src.detection.yolo import DEFAULT_CONFIDENCE, DEFAULT_MODEL, DetectionError, YOLOVehicleDetector
from src.evaluation.detection_metrics import DEFAULT_IOU_THRESHOLD, evaluate_image
from src.exports.csv_export import detections_to_csv, evaluation_to_csv
from src.routing.algorithms import compare_algorithms
from src.routing.graph import RoadEdge, WeightedGraph, build_weighted_graph
from src.routing.scenarios import SCENARIOS, TrafficScenario, apply_edge_load_override, scenario_edges
from src.traffic.density_estimator import estimate_density
from src.traffic.estimation import DEFAULT_SATURATION_COUNT, TrafficLoadEstimate, estimate_traffic_load

ROOT = Path(__file__).resolve().parent
DATASETS = ("BDD100K", "IDD")


@st.cache_data(show_spinner=False, ttl=60)
def cached_discovery(dataset_name: str, root_path: str) -> DatasetDiscovery:
    return discover_dataset(dataset_name, root_path)


@st.cache_data(show_spinner=False, ttl=300)
def cached_annotation_scan(dataset_name: str, root_path: str) -> AnnotationScan:
    return scan_annotations(discover_dataset(dataset_name, root_path))


@st.cache_resource(show_spinner=False)
def detector_for(model_source: str) -> YOLOVehicleDetector:
    # YOLOVehicleDetector loads weights only when inference is requested.
    return YOLOVehicleDetector(model_source)


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return path.name


def _scenario_frame(scenario: TrafficScenario) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "source": edge.source,
            "target": edge.target,
            "base_cost": edge.base_cost,
            "traffic_load": edge.traffic_load,
        }
        for edge in scenario_edges(scenario)
    ], columns=("source", "target", "base_cost", "traffic_load"))


def _frame_to_edges(frame: pd.DataFrame) -> list[RoadEdge]:
    edges: list[RoadEdge] = []
    for index, row in frame.iterrows():
        values = [row.get("source"), row.get("target"), row.get("base_cost"), row.get("traffic_load")]
        if all(pd.isna(value) or str(value).strip() == "" for value in values):
            continue
        if pd.isna(values[0]) or pd.isna(values[1]) or str(values[0]).strip() == "" or str(values[1]).strip() == "":
            raise ValueError(f"Edge row {index + 1} needs both a source and target node.")
        if pd.isna(values[2]) or pd.isna(values[3]):
            raise ValueError(f"Edge row {index + 1} needs a base cost and traffic load.")
        try:
            edges.append(RoadEdge(
                source=str(values[0]).strip(),
                target=str(values[1]).strip(),
                base_cost=float(values[2]),
                traffic_load=float(values[3]),
            ))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Edge row {index + 1} contains a non-numeric cost or load.") from exc
    if not edges:
        raise ValueError("Add at least one complete edge to the illustrative road graph.")
    return edges


def _graph_dot(graph: WeightedGraph, highlighted_path: tuple[str, ...] = ()) -> str:
    node_ids = {node: f"n{index}" for index, node in enumerate(graph.nodes)}
    path_edges = {
        tuple(sorted((left, right))) if not graph.directed else (left, right)
        for left, right in zip(highlighted_path, highlighted_path[1:])
    }
    lines = ["graph RoadNetwork {" if not graph.directed else "digraph RoadNetwork {", "  rankdir=LR;", "  node [shape=ellipse, style=filled, fillcolor=\"#eef4ff\", color=\"#456\"];", "  edge [fontname=\"Arial\"];" ]
    for node in graph.nodes:
        label = json.dumps(node, ensure_ascii=False)
        lines.append(f"  {node_ids[node]} [label={label}];")
    connector = " -> " if graph.directed else " -- "
    for edge in graph.edges:
        key = (edge.source, edge.target) if graph.directed else tuple(sorted((edge.source, edge.target)))
        on_route = key in path_edges
        color = "#d62728" if on_route else "#738091"
        width = "3" if on_route else "1"
        label = json.dumps(
            f"base {edge.base_cost:g} | load {edge.traffic_load:.2f} | final {edge.adjusted_cost:.2f}",
            ensure_ascii=False,
        )
        lines.append(
            f"  {node_ids[edge.source]}{connector}{node_ids[edge.target]} "
            f"[label={label}, color=\"{color}\", penwidth={width}];"
        )
    lines.append("}")
    return "\n".join(lines)


def _dependency_status() -> dict[str, bool]:
    modules = {
        "Streamlit dashboard": "streamlit",
        "Ultralytics YOLO": "ultralytics",
        "PyTorch backend": "torch",
        "OpenCV": "cv2",
        "Pillow": "PIL",
        "NumPy": "numpy",
        "pandas": "pandas",
        "Streaming JSON parser (ijson)": "ijson",
    }
    return {label: importlib.util.find_spec(module) is not None for label, module in modules.items()}


def _status_for_dataset(discovery: DatasetDiscovery, scan: AnnotationScan | None) -> tuple[str, str]:
    if not discovery.root_exists:
        return "Missing", "Directory is not available. Set the local path and place the dataset there."
    if discovery.image_count == 0:
        return "No images", "Directory exists, but no supported image files were discovered."
    if discovery.annotation_file_count == 0:
        return "Images ready; labels missing", "Inference is possible; ground-truth evaluation is unavailable."
    if scan is None:
        return "Images ready; labels unscanned", "Run the annotation scan to verify schemas and image matching."
    if not scan.parsed_formats:
        return "Images ready; no supported labels", "Candidate annotation files were found, but no supported schema was parsed."
    if not scan.matched_image_paths:
        return "Labels parsed; images unmatched", "Supported files parsed, but none matched discovered images."
    if scan.unsupported_files or scan.unmatched_annotation_names or scan.invalid_image_paths or scan.issues:
        return "Partially ready", "Some files/records need attention; see the scan report."
    return "Images and labels ready", "At least one supported annotation schema matched local images."


def _format_metric(value: float | None) -> str:
    return "Unavailable" if value is None else f"{value:.3f}"


st.set_page_config(page_title="Ambulance Route Optimization", page_icon="🚑", layout="wide")
st.title("Computer Vision-Based Traffic Load & Ambulance Route Optimization")
st.caption("Academic prototype · local images · illustrative graph · no live dispatch or real-world ETA")

with st.expander("Project overview and limitations", expanded=True):
    st.markdown(
        "**Pipeline:** road-scene image → pretrained YOLO vehicle detections → bounded image-based count proxy → "
        "traffic-adjusted illustrative graph → Dijkstra/A* comparison."
    )
    st.warning(
        "A count from one image is not calibrated physical traffic density, speed, queue length, or travel time. "
        "The graph and scenario inputs are illustrative; edge costs are abstract units, not minutes. "
        "This prototype does not monitor live traffic or control/dispatch ambulances."
    )

overview_tab, dataset_tab, detection_tab, routing_tab, evaluation_tab = st.tabs([
    "Overview", "Dataset status", "Detection & traffic proxy", "Road graph & routing", "Evaluation & exports"
])

# Shared local paths; these are configuration, not downloaded dataset contents.
with dataset_tab:
    st.subheader("Local dataset discovery")
    default_bdd = os.environ.get("AMBULANCE_BDD100K_DIR", str(ROOT / "data" / "bdd100k"))
    default_idd = os.environ.get("AMBULANCE_IDD_DIR", str(ROOT / "data" / "idd"))
    col_a, col_b = st.columns(2)
    with col_a:
        bdd_root_text = st.text_input("BDD100K local directory", value=default_bdd, key="bdd_root_path")
    with col_b:
        idd_root_text = st.text_input("IDD local directory", value=default_idd, key="idd_root_path")
    dataset_roots = {"BDD100K": bdd_root_text, "IDD": idd_root_text}
    if st.button("Refresh local file discovery", help="Clear cached scans after extracting/copying local dataset files."):
        cached_discovery.clear()
        cached_annotation_scan.clear()
        st.rerun()

    discoveries: dict[str, DatasetDiscovery] = {}
    scans: dict[str, AnnotationScan | None] = {}
    for dataset_name, root_text in dataset_roots.items():
        discovery = cached_discovery(dataset_name, root_text)
        discoveries[dataset_name] = discovery
        scan: AnnotationScan | None = None
        if discovery.root_exists and discovery.annotation_file_count:
            try:
                with st.spinner(f"Inspecting {dataset_name} annotation schemas and image matches…"):
                    scan = cached_annotation_scan(dataset_name, root_text)
            except Exception as exc:
                st.error(f"Annotation scan failed for {dataset_name}: {type(exc).__name__}: {exc}")
        scans[dataset_name] = scan
        status, explanation = _status_for_dataset(discovery, scan)
        with st.container(border=True):
            left, right = st.columns([2, 3])
            with left:
                st.markdown(f"### {dataset_name}")
                st.write(f"**Status:** {status}")
                st.caption(explanation)
                st.code(str(discovery.root), language="text")
            with right:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Images", discovery.image_count)
                m2.metric("Candidate annotation files", discovery.annotation_file_count)
                m3.metric("Matched images", len(scan.matched_image_paths) if scan else 0)
                m4.metric("Images with malformed labels", len(scan.invalid_image_paths) if scan else 0)
                if discovery.annotation_format_counts:
                    st.write("Candidate formats by extension:", discovery.annotation_format_counts)
                if scan:
                    st.write("Recognized annotation schemas:", scan.parsed_formats or "None")
                    st.write(
                        f"Unsupported files: {len(scan.unsupported_files)} · "
                        f"Unmatched annotation identifiers: {len(scan.unmatched_annotation_names)} · "
                        f"Images with invalid annotation records: {len(scan.invalid_image_paths)} · "
                        f"Parser/matching issues: {len(scan.issues)}"
                    )
                    if scan.unmapped_categories:
                        st.write("Unmapped categories (excluded from four-class evaluation):", scan.unmapped_categories)
                    if scan.unsupported_files:
                        st.caption("Unsupported annotation-file examples: " + ", ".join(scan.unsupported_files[:5]))
                    if scan.unmatched_annotation_names:
                        st.caption("Unmatched identifier examples: " + ", ".join(scan.unmatched_annotation_names[:5]))
                    for issue in scan.issues[:8]:
                        st.warning(issue)
                    if len(scan.issues) > 8:
                        st.caption(f"Showing 8 of {len(scan.issues)} scan issues.")
                if discovery.scan_errors:
                    for issue in discovery.scan_errors:
                        st.warning(issue)
                if discovery.image_paths:
                    examples = [_relative_path(path, discovery.root) for path in discovery.image_paths[:5]]
                    st.caption("Example discovered image paths: " + ", ".join(examples))
                    validation_key = f"image_validation::{dataset_name}::{discovery.root}::{discovery.image_count}"
                    if st.button(
                        f"Validate all {discovery.image_count} discovered image(s)",
                        key=f"validate_images::{dataset_name}",
                        help="Decodes every discovered image. On a large dataset this may take time; selected images are always checked before inference.",
                    ):
                        with st.spinner(f"Validating all {dataset_name} images; large folders may take several minutes…"):
                            st.session_state[validation_key] = validate_image_paths(discovery.image_paths)
                    validation = st.session_state.get(validation_key)
                    if validation is None:
                        st.caption("Full image-integrity scan has not been run; a selected image is validated before inference.")
                    else:
                        st.write(
                            f"Image validation: {validation.readable_count}/{validation.checked_count} readable · "
                            f"{len(validation.unreadable)} unreadable"
                        )
                        if validation.unreadable:
                            for invalid_path, invalid_reason in validation.unreadable[:5]:
                                st.warning(f"Unreadable image {invalid_path}: {invalid_reason}")
                            if len(validation.unreadable) > 5:
                                st.caption(f"Showing 5 of {len(validation.unreadable)} unreadable files.")

    st.markdown("#### Runtime dependency readiness")
    dep_status = _dependency_status()
    dep_columns = st.columns(4)
    for index, (label, available) in enumerate(dep_status.items()):
        with dep_columns[index % len(dep_columns)]:
            st.write(f"{'✅' if available else '❌'} {label}")
    missing = [label for label, available in dep_status.items() if not available]
    if missing:
        st.error("Missing packages: " + ", ".join(missing) + ". Install with `./scripts/install_dependencies.sh` in the active environment.")
    else:
        st.success("Declared package modules are present. OpenCV/model imports and checkpoint availability are exercised when inference is requested.")
    st.info(
        "No Kaggle data is bundled or downloaded automatically. The scan reports local files only. "
        "See README.md for the specified sources, expected parser support, and local acquisition steps."
    )

with overview_tab:
    st.subheader("Prototype status")
    st.write(
        "The runnable baseline connects actual image input and model inference to an explicit count proxy and "
        "traffic-dependent costs on an editable demonstration graph. Annotation-based metrics are shown only for "
        "a selected image with successfully matched supported labels."
    )
    st.code(
        "Road image → YOLO detection → min(vehicle_count / saturation_count, 1) → "
        "base_cost × (1 + alpha × edge_load) → Dijkstra / A*",
        language="text",
    )
    st.markdown("**Implemented boundaries**")
    st.markdown(
        "- Pretrained inference only; no dataset-specific training is claimed.\n"
        "- The default A* heuristic is zero because the illustrative graph has no calibrated coordinates.\n"
        "- Detection P/R and supported-class count MAE require a matched annotation; single-image metrics are not mAP.\n"
        "- Dataset comparison, video tracking, live traffic, real GIS routing, and dispatch are not implemented."
    )
    st.markdown("**Available components**")
    st.write("Dataset adapters: BDD100K detection JSON, COCO JSON, Pascal VOC XML, YOLO TXT with explicit class-name mapping.")
    st.write("Vehicle mapping: model/category names are inspected at runtime and normalized to car, truck, bus, motorcycle.")
    st.write("Traffic scenarios: two synthetic examples are supplied; all scenario values remain editable and clearly labeled.")

with detection_tab:
    st.subheader("Choose an image and run pretrained vehicle detection")
    input_mode = st.radio("Image source", ("Local dataset image", "Upload image"), horizontal=True)
    current_image = None
    current_image_key: str | None = None
    current_image_name = ""
    current_dataset_name = ""
    if input_mode == "Local dataset image":
        chosen_dataset = st.selectbox("Dataset", DATASETS)
        root_text = dataset_roots[chosen_dataset]
        discovery = discoveries[chosen_dataset]
        current_dataset_name = chosen_dataset
        if not discovery.root_exists:
            st.warning(f"{chosen_dataset} directory is missing: {discovery.root}")
        elif not discovery.image_paths:
            st.info("No local images discovered. Use the upload option or add images from the specified dataset.")
        else:
            search = st.text_input("Filter discovered image paths", placeholder="Type part of a file or folder name")
            filtered = [
                path for path in discovery.image_paths
                if search.casefold() in _relative_path(path, discovery.root).casefold()
            ]
            if not filtered:
                st.warning("No discovered images match that filter.")
            else:
                st.caption(f"{len(filtered)} matching image(s); at most the first 500 are shown in the selector.")
                options = filtered[:500]
                selected_relative = st.selectbox("Image file", options, format_func=lambda path: _relative_path(path, discovery.root))
                if selected_relative:
                    current_image_name = Path(selected_relative).name
                    current_image_key = str(Path(selected_relative).resolve())
                    try:
                        current_image = load_rgb_image(selected_relative)
                        st.caption(f"Image validation: readable · {current_image.width} × {current_image.height} pixels")
                    except ValueError as exc:
                        st.error(f"Selected image is unreadable: {exc}")
    else:
        uploaded = st.file_uploader(
            "Upload a road-scene image (JPG, PNG, BMP, TIFF, or WebP)",
            type=("jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"),
        )
        if uploaded is not None:
            image_bytes = uploaded.getvalue()
            current_dataset_name = "Uploaded image"
            current_image_name = uploaded.name
            current_image_key = "upload:" + hashlib.sha256(image_bytes).hexdigest()
            try:
                current_image = load_rgb_image(image_bytes)
                st.caption(f"Upload validation: readable · {current_image.width} × {current_image.height} pixels")
            except ValueError as exc:
                st.error(f"Uploaded image is unreadable: {exc}")

    model_default = os.environ.get("AMBULANCE_YOLO_MODEL", DEFAULT_MODEL)
    model_source = st.text_input(
        "Ultralytics model name or local checkpoint path",
        value=model_default,
        help="Default yolo11n.pt may download official pretrained weights on first inference. A local checkpoint path is also accepted.",
    )
    confidence = st.slider("Detection confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE, 0.01)
    can_infer = current_image is not None and bool(model_source.strip())
    if st.button("Run actual YOLO inference", type="primary", disabled=not can_infer):
        try:
            with st.spinner("Loading the configured detector and running inference…"):
                detector = detector_for(model_source.strip())
                run = detector.detect(
                    image=current_image,
                    image_key=current_image_key or current_image_name,
                    image_name=current_image_name,
                    dataset_name=current_dataset_name,
                    confidence_threshold=confidence,
                )
            st.session_state["detection_run"] = run
            st.success("Inference completed using the configured Ultralytics model.")
        except (DetectionError, ValueError, RuntimeError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected inference error ({type(exc).__name__}): {exc}")

    run = st.session_state.get("detection_run")
    if (
        isinstance(run, DetectionRun)
        and current_image is not None
        and run.image_key == current_image_key
        and run.model_source == model_source.strip()
        and run.confidence_threshold == confidence
    ):
        left, right = st.columns(2)
        with left:
            st.image(current_image, caption=f"Original · {run.image_name}", use_container_width=True)
        try:
            annotated = annotate_image(current_image, run.detections)
            with right:
                st.image(annotated, caption="Actual YOLO detections · boxes show class and confidence", use_container_width=True)
        except Exception as exc:
            st.error(f"Could not draw detection overlays: {exc}")
        st.caption(
            f"Model: {run.model_source} · confidence threshold: {run.confidence_threshold:.2f} · "
            f"measured inference call: {run.inference_seconds:.4f} s"
        )
        st.write("Verified loaded model class IDs → supported vehicle class:", run.vehicle_class_ids)
        count_columns = st.columns(len(run.counts) + 1)
        count_columns[0].metric("Total supported vehicles", run.vehicle_count)
        for index, (vehicle_class, count) in enumerate(run.counts.items(), 1):
            count_columns[index].metric(vehicle_class.title(), count)
        if run.detections:
            st.dataframe(pd.DataFrame([
                {
                    "Class": item.vehicle_class,
                    "Model class": item.model_class_name,
                    "Class ID": item.class_id,
                    "Confidence": item.confidence,
                    "x1": item.box_xyxy[0], "y1": item.box_xyxy[1],
                    "x2": item.box_xyxy[2], "y2": item.box_xyxy[3],
                }
                for item in run.detections
            ]), use_container_width=True, hide_index=True)
        else:
            st.info("The actual inference returned zero detections for the four supported vehicle classes at this threshold.")
    elif isinstance(run, DetectionRun):
        st.info("A prior result belongs to a different image, model, or confidence threshold. Run inference to refresh it.")
    else:
        st.info("Select or upload an image, then run inference. No detections are fabricated when inference has not run.")

    st.markdown("#### Image-based traffic-load proxy")
    saturation_count = st.number_input(
        "Saturation count (design parameter)", min_value=1, max_value=500, value=DEFAULT_SATURATION_COUNT, step=1,
        help="A configurable count at which this simple proxy saturates at 1.0; it is not an empirically calibrated traffic threshold.",
    )
    active_run = st.session_state.get("detection_run")
    current_run_is_relevant = (
        isinstance(active_run, DetectionRun)
        and current_image is not None
        and active_run.image_key == current_image_key
        and active_run.model_source == model_source.strip()
        and active_run.confidence_threshold == confidence
    )
    active_proxy: TrafficLoadEstimate | None = None
    if current_run_is_relevant:
        active_proxy = estimate_traffic_load(active_run.vehicle_count, int(saturation_count))
        density_level = estimate_density(active_run.vehicle_count)
        proxy_columns = st.columns(2)
        proxy_columns[0].metric("Image-based load proxy (0–1)", f"{active_proxy.load_fraction:.3f}")
        proxy_columns[1].metric("Illustrative density class (display only)", density_level)
        st.write(f"Formula: `{active_proxy.formula}`")
        st.caption(
            f"Input: {active_run.vehicle_count} supported vehicle detections from {active_run.image_name}. "
            "This is a reproducible image-count proxy, not physical traffic density, speed, queue length, or ETA. "
            "The Low/Medium/High class uses fixed illustrative thresholds (≤5 / 6–15 / ≥16 vehicles) for display; "
            "routing edge costs use only the continuous proxy, not the class."
        )
        st.session_state["active_traffic_load"] = active_proxy.load_fraction
        st.session_state["active_traffic_source"] = active_run.image_name
    else:
        st.info("Run inference for the currently selected image/model to calculate a fresh proxy. The routing demo can still use explicit illustrative scenarios.")
        st.session_state.pop("active_traffic_load", None)
        st.session_state.pop("active_traffic_source", None)

with routing_tab:
    st.subheader("Editable illustrative road graph")
    st.markdown("**Adjusted edge cost = base edge cost × (1 + α × edge traffic-load fraction)**")
    st.caption("Base costs are abstract units; edge loads are scenario assumptions in [0, 1]. They are not minutes or measured delays.")
    scenario = st.selectbox("Traffic scenario", SCENARIOS, format_func=lambda value: value.name)
    st.info(scenario.description)
    alpha = st.slider("Traffic penalty strength α", 0.0, 3.0, 1.0, 0.1)
    st.caption("The two supplied scenarios are synthetic. A single image has no implied road-segment location.")

    default_frame = _scenario_frame(scenario)
    edge_editor_key = f"edge_editor::{scenario.name}"
    edited_frame = st.data_editor(
        default_frame,
        num_rows="dynamic",
        key=edge_editor_key,
        use_container_width=True,
        hide_index=True,
        column_config={
            "source": st.column_config.TextColumn("From node", required=True),
            "target": st.column_config.TextColumn("To node", required=True),
            "base_cost": st.column_config.NumberColumn("Base cost (abstract units)", min_value=0.0, step=0.5, format="%.2f", required=True),
            "traffic_load": st.column_config.NumberColumn("Traffic load (0–1)", min_value=0.0, max_value=1.0, step=0.05, format="%.2f", required=True),
        },
    )
    image_load = st.session_state.get("active_traffic_load")
    image_source = st.session_state.get("active_traffic_source")
    proxy_edge_override: tuple[str, str] | None = None
    if image_load is not None:
        apply_proxy = st.checkbox(
            "Illustratively apply the current image proxy to one selected edge",
            help="This is a manual scenario assignment, not a camera-to-road mapping.",
        )
        if apply_proxy:
            edge_options: dict[str, tuple[str, str]] = {}
            for row_index, row in edited_frame.iterrows():
                left, right = row.get("source"), row.get("target")
                if pd.isna(left) or pd.isna(right) or not str(left).strip() or not str(right).strip():
                    continue
                label = f"{str(left).strip()} ↔ {str(right).strip()} (row {row_index + 1})"
                edge_options[label] = (str(left).strip(), str(right).strip())
            if not edge_options:
                st.warning("Complete at least one edge row before applying an image-derived value.")
            else:
                edge_label = st.selectbox("Edge receiving the image-derived proxy", list(edge_options))
                proxy_edge_override = edge_options[edge_label]
                st.info(
                    f"Applied {float(image_load):.3f} from image {image_source!r} to {edge_label}. "
                    "The final edge-cost table shows this override; the assignment is illustrative only."
                )

    st.caption("Edit endpoints/base costs/loads above; add or remove rows to modify topology. Duplicate edges, self-loops, negative costs, non-finite values, and loads outside [0, 1] are rejected.")
    try:
        roads = _frame_to_edges(edited_frame)
        if proxy_edge_override is not None and image_load is not None:
            roads = apply_edge_load_override(roads, *proxy_edge_override, float(image_load))
        graph = build_weighted_graph(roads, alpha=alpha, directed=False)
        nodes = list(graph.nodes)
        if len(nodes) < 1:
            raise ValueError("The graph has no routable nodes.")
        source_default = "Ambulance Base" if "Ambulance Base" in nodes else nodes[0]
        destination_default = "Emergency Site" if "Emergency Site" in nodes else nodes[-1]
        left, right = st.columns(2)
        with left:
            source = st.selectbox("Route source", nodes, index=nodes.index(source_default))
        with right:
            destination = st.selectbox("Route destination", nodes, index=nodes.index(destination_default))
        comparison = compare_algorithms(graph, source, destination)

        adjusted_table = pd.DataFrame([
            {
                "From": edge.source,
                "To": edge.target,
                "Base cost": edge.base_cost,
                "Traffic load": edge.traffic_load,
                "Traffic penalty": edge.adjusted_cost - edge.base_cost,
                "Final edge cost": edge.adjusted_cost,
            }
            for edge in graph.edges
        ])
        st.markdown("#### Edge-cost breakdown")
        st.dataframe(adjusted_table, use_container_width=True, hide_index=True)

        st.markdown("#### Dijkstra vs A* (same graph and adjusted costs)")
        cols = st.columns(2)
        for col, label, result, seconds in (
            (cols[0], "Dijkstra", comparison.dijkstra, comparison.dijkstra_seconds),
            (cols[1], "A*", comparison.astar, comparison.astar_seconds),
        ):
            with col:
                st.markdown(f"**{label}**")
                st.write("Route:", " → ".join(result.path) if result.found else "No route (destination unreachable)")
                st.metric("Total abstract cost", f"{result.total_cost:.3f}" if result.total_cost is not None else "Unreachable")
                st.metric("Measured search time", f"{seconds:.6f} s")
                st.metric("Expanded nodes", result.expanded_nodes)
        if comparison.dijkstra.found and comparison.astar.found:
            if math.isclose(comparison.dijkstra.total_cost or 0.0, comparison.astar.total_cost or 0.0, rel_tol=1e-9, abs_tol=1e-9):
                st.success("Both algorithms return equal optimal cost on the same graph.")
            else:
                st.error("The algorithms returned different costs; inspect the graph and implementation.")
        elif comparison.dijkstra.found != comparison.astar.found:
            st.error("Algorithms disagree about reachability.")
        st.caption(
            "Expanded-node convention: count each non-stale priority-queue node settled, including a reached destination; "
            "source = destination counts as one. A* uses h(n)=0 (admissible), so it is Dijkstra-style uniform-cost search. "
            "Runtime values are measured per call and may vary."
        )
        path = comparison.dijkstra.path if comparison.dijkstra.found else ()
        st.markdown("#### Route visualization (Dijkstra route highlighted)")
        st.graphviz_chart(_graph_dot(graph, path), use_container_width=True)
        st.caption("Red edges are on the computed Dijkstra route; edge labels show base cost, load fraction, and final cost.")
    except (ValueError, TypeError, KeyError) as exc:
        st.error(f"Cannot calculate routes from the current graph: {exc}")

with evaluation_tab:
    st.subheader("Ground-truth evaluation and CSV exports")
    run = st.session_state.get("detection_run")
    if not isinstance(run, DetectionRun):
        st.info("Run YOLO inference to create actual detection rows for export. No sample metrics are pre-populated.")
    else:
        st.markdown(f"**Latest actual inference:** {run.image_name} · {run.dataset_name} · {len(run.detections)} supported detections")
        detection_csv = detections_to_csv(run)
        st.download_button(
            "Download detections CSV (one row per detected box)",
            data=detection_csv,
            file_name=f"detections_{Path(run.image_name).stem or 'image'}.csv",
            mime="text/csv",
        )
        if not run.detections:
            st.caption("The detection CSV contains its column header and zero detection rows, reflecting the actual inference.")

        scan = scans.get(run.dataset_name)
        if run.dataset_name not in DATASETS:
            st.warning("Ground-truth evaluation is unavailable for an uploaded image unless it is explicitly matched to local annotations.")
        elif scan is None:
            st.warning("No parsed local annotation scan is available; evaluation metrics cannot be calculated.")
        elif scan.is_invalid(run.image_key):
            st.warning("A matching annotation record contains malformed label(s). Evaluation is withheld for this image rather than treating invalid labels as an empty ground truth.")
        elif not scan.is_matched(run.image_key):
            st.warning("This image has no successfully matched supported annotation record. Precision/recall and count MAE are not reported.")
        else:
            iou_threshold = st.slider("IoU matching threshold", 0.05, 1.0, DEFAULT_IOU_THRESHOLD, 0.05)
            ground_truth = scan.boxes_for(run.image_key)
            evaluation = evaluate_image(
                predictions=run.detections,
                ground_truth=ground_truth,
                image_name=run.image_name,
                confidence_threshold=run.confidence_threshold,
                iou_threshold=iou_threshold,
            )
            st.markdown("**Evaluation sample size: 1 matched image**")
            metrics = st.columns(5)
            metrics[0].metric("True positives", evaluation.true_positives)
            metrics[1].metric("False positives", evaluation.false_positives)
            metrics[2].metric("False negatives", evaluation.false_negatives)
            metrics[3].metric("Precision", _format_metric(evaluation.precision))
            metrics[4].metric("Recall", _format_metric(evaluation.recall))
            metrics2 = st.columns(3)
            metrics2[0].metric("F1", _format_metric(evaluation.f1))
            metrics2[1].metric("Predicted mapped vehicles", evaluation.predicted_vehicle_count)
            metrics2[2].metric("Mapped ground-truth vehicles", evaluation.ground_truth_vehicle_count)
            st.metric("Supported-class vehicle-count absolute error", f"{evaluation.vehicle_count_mae:.3f}")
            st.write("Category mapping:", evaluation.category_mapping)
            st.write("Matching rule:", evaluation.matching_rule)
            st.info(evaluation.notes)
            st.download_button(
                "Download evaluation CSV (one row per evaluated image)",
                data=evaluation_to_csv(evaluation, run.dataset_name),
                file_name=f"evaluation_{Path(run.image_name).stem or 'image'}.csv",
                mime="text/csv",
            )
    st.markdown("#### Metrics not currently reported")
    st.write(
        "Dataset-level mAP, dataset comparison, and training/validation/test metrics are not reported by this single-image "
        "baseline. They require a compatible, separated labeled sample and a documented dataset-level evaluation run. "
        "No accuracy, mAP, MAE, latency, or route-improvement value is fabricated."
    )
