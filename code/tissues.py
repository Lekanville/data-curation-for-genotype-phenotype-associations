import argparse
import pandas as pd
from pathlib import Path


CORE_TISSUES = [
    {"Tissue_ID": "Adipose_Subcutaneous", "System": "Adipose", "Is_Metabolic": 1, "Is_Excretory": 0},
    {"Tissue_ID": "Artery_Aorta", "System": "Cardiovascular", "Is_Metabolic": 0, "Is_Excretory": 0},
    {"Tissue_ID": "Brain_Hypothalamus", "System": "Nervous", "Is_Metabolic": 0, "Is_Excretory": 0},
    {"Tissue_ID": "Heart_Left_Ventricle", "System": "Cardiovascular", "Is_Metabolic": 0, "Is_Excretory": 0},
    {"Tissue_ID": "Kidney_Cortex", "System": "Excretory", "Is_Metabolic": 0, "Is_Excretory": 1},
    {"Tissue_ID": "Liver", "System": "Digestive", "Is_Metabolic": 1, "Is_Excretory": 0},
    {"Tissue_ID": "Muscle_Skeletal", "System": "Musculoskeletal", "Is_Metabolic": 1, "Is_Excretory": 0},
    {"Tissue_ID": "Pancreas", "System": "Endocrine", "Is_Metabolic": 1, "Is_Excretory": 0},
    {"Tissue_ID": "Small_Intestine_Terminal_Ileum", "System": "Digestive", "Is_Metabolic": 1, "Is_Excretory": 0},
    {"Tissue_ID": "Whole_Blood", "System": "Immune", "Is_Metabolic": 0, "Is_Excretory": 0},
]


def build_tissue_nodes() -> pd.DataFrame:
    df_tissues_raw = pd.DataFrame(CORE_TISSUES)
    df_tissue_nodes = pd.get_dummies(df_tissues_raw, columns=["System"], prefix="Sys")
    df_tissue_nodes.set_index("Tissue_ID", inplace=True)
    return df_tissue_nodes


def save_tissue_nodes(df_tissue_nodes: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tissue_nodes.csv"
    df_tissue_nodes.to_csv(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tissue node features from the notebook workflow")
    parser.add_argument("--output-dir", default="output/tissues", help="Directory for the tissue node CSV")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    df_tissue_nodes = build_tissue_nodes()
    output_path = save_tissue_nodes(df_tissue_nodes, output_dir)

    print(f"Wrote tissue nodes to {output_path}")


if __name__ == "__main__":
    main()
