from __future__ import annotations

import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def load_summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))


def make_paragraphs(styles, summary: dict) -> list:
    if "experiments" in summary:
        experiments = summary["experiments"]
        best_exp = max(experiments, key=lambda item: item["oof_accuracy"])
        weights = ", ".join(
            f'{item["experiment_name"]}: {item["weight"]:.3f}' for item in summary["ensemble_weights"]
        )
        best_line = (
            f"Best single experiment: <b>{best_exp['experiment_name']}</b> with OOF accuracy <b>{best_exp['oof_accuracy']:.4f}</b>. "
            f"Final weighted ensemble OOF accuracy: <b>{summary['ensemble_oof_accuracy']:.4f}</b>."
        )
    else:
        components = summary["blend_components"]
        best_exp = max(components, key=lambda item: item["oof_accuracy"])
        weights = ", ".join(f"{item['name']}: {item['weight']:.3f}" for item in components)
        best_line = (
            f"Best component: <b>{best_exp['name']}</b> with OOF accuracy <b>{best_exp['oof_accuracy']:.4f}</b>. "
            f"Final blended ensemble OOF accuracy: <b>{summary['ensemble_oof_accuracy']:.4f}</b>."
        )
    return [
        Paragraph("MPA-MLF Final Project 2026: Classification of Room Occupancy", styles["Title"]),
        Spacer(1, 0.25 * cm),
        Paragraph(
            "This report describes a cross-validated image classification pipeline for inferring the number of people in a room from 60 GHz delay-Doppler snapshots.",
            styles["BodyText"],
        ),
        Spacer(1, 0.15 * cm),
        Paragraph(
            "The final solution uses transfer learning on compact pretrained convolutional networks, stratified cross-validation, light geometric augmentation, and a weighted probability ensemble.",
            styles["BodyText"],
        ),
        Spacer(1, 0.15 * cm),
        Paragraph(best_line, styles["BodyText"]),
        Spacer(1, 0.15 * cm),
        Paragraph(f"Ensemble weights: {weights}.", styles["BodyText"]),
    ]


def build_results_table(summary: dict):
    rows = [["Experiment", "Input", "OOF accuracy", "Fold accuracies"]]
    experiments = summary.get("experiments", summary.get("blend_components", []))
    for exp in experiments:
        rows.append(
            [
                exp.get("model_name", exp.get("name", "component")),
                exp.get("input_mode", "blend"),
                f'{exp["oof_accuracy"]:.4f}',
                ", ".join(f"{score:.4f}" for score in exp.get("fold_scores", [])) or "-",
            ]
        )
    table = Table(rows, colWidths=[5.0 * cm, 2.5 * cm, 3.0 * cm, 6.0 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#254c7d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b0b7c3")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#edf2f7")]),
            ]
        )
    )
    return table


def add_image_if_present(path: Path, width_cm: float):
    if not path.exists():
        return []
    return [Image(str(path), width=width_cm * cm, height=width_cm * cm * 0.75), Spacer(1, 0.2 * cm)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final project PDF report.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--public-score", type=str, default="To be filled after Kaggle submission")
    parser.add_argument("--private-score", type=str, default="To be filled after leaderboard release")
    args = parser.parse_args()

    run_dir = args.run_dir
    summary = load_summary(run_dir)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SectionTitle", parent=styles["Heading2"], spaceAfter=6, textColor=colors.HexColor("#254c7d")))

    story = []
    story.extend(make_paragraphs(styles, summary))
    story.append(Spacer(1, 0.25 * cm))

    story.append(Paragraph("1. Problem and data", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "The training set contains 9,227 RGB images of size 51x45 pixels. The test set contains 3,955 images. "
            "Labels correspond to four classes: machine only, one person, two persons, and three persons.",
            styles["BodyText"],
        )
    )
    story.extend(add_image_if_present(run_dir / "class_distribution.png", 12.0))

    story.append(Paragraph("2. Method", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "The final training pipeline uses stratified K-fold cross-validation, transfer learning from ImageNet-pretrained timm backbones, "
            "resize to a higher resolution, normalization with dataset statistics, light affine augmentation, Gaussian blur, random erasing, "
            "AdamW optimization, cosine learning-rate scheduling, and early stopping on validation accuracy.",
            styles["BodyText"],
        )
    )
    story.append(
        Paragraph(
            "Two complementary input variants were trained: the original RGB encoding and a grayscale image repeated over three channels. "
            "Their predicted probabilities were combined with weights proportional to out-of-fold accuracy.",
            styles["BodyText"],
        )
    )

    story.append(Paragraph("3. Results", styles["SectionTitle"]))
    story.append(build_results_table(summary))
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        Paragraph(
            f"Kaggle public leaderboard score: <b>{args.public_score}</b>. Kaggle private leaderboard score: <b>{args.private_score}</b>.",
            styles["BodyText"],
        )
    )
    story.extend(add_image_if_present(run_dir / "ensemble_confusion_matrix.png", 12.0))

    story.append(Paragraph("4. Discussion and next steps", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "The strongest gains came from transfer learning and probability ensembling. Further improvements can still come from additional backbones, "
            "test-time augmentation, or pseudo-labeling once the public leaderboard behavior is known. The final code intentionally keeps the dataset outside the project folder so the repository remains lightweight and compliant with the assignment constraints.",
            styles["BodyText"],
        )
    )

    doc = SimpleDocTemplate(
        str(args.output_pdf),
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
    )
    doc.build(story)
    print(args.output_pdf.resolve())


if __name__ == "__main__":
    main()
