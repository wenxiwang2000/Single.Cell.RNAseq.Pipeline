
"""
sc_workflow_table_annotation_mtproxy.py
======================================

Scanpy workflow with:
1. QC filtering
2. Proxy mitochondrial stress scoring using curated Tribolium TC IDs
3. UMAP + Leiden clustering
4. Table-based annotation using a curated marker table (Tissue | Marker gene | TC ID)
5. Optional subclustering using either a cluster number or a cell-type name

Main run example
----------------
python sc_workflow_table_annotation_mtproxy.py \
    --mtx_dir /path/to/filtered_feature_bc_matrix \
    --out_dir /path/to/output \
    --marker_table /path/to/MT_scRNA_final_marker_table.xlsx \
    --mapping_tsv /path/to/trib_dmel_gene_mapping_with_TC.tsv

Subclustering example
---------------------
python sc_workflow_table_annotation_mtproxy.py \
    --out_dir /path/to/output \
    --cluster 1

python sc_workflow_table_annotation_mtproxy.py \
    --out_dir /path/to/output \
    --cluster "hindgut cells"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import scanpy as sc
except ImportError:  # pragma: no cover
    sc = None


# ========= EDIT PATHS =========
MTX_DIR = r"C:\Users\dmp753\Desktop\single cell sequence data\ScSeq filtered counts (1)\ScSeq filtered counts\filtered_feature_bc_matrix_larva"
OUT_DIR = r"C:\Users\dmp753\Desktop\single cell sequence data\ScSeq filtered counts (1)\ScSeq filtered counts\output"
MARKER_TABLE = r"C:\Users\dmp753\Downloads\scRNA_marker_table.xlsx"
MAPPING_TSV = r"C:\Users\dmp753\Downloads\trib_dmel_gene_mapping_with_TC (1).tsv"
# ==============================


# -----------------------------------------------------------------------------
# Proxy mitochondrial stress gene set
# -----------------------------------------------------------------------------
STRICT_MT_TC = {
    "TC009512",  # COX4
    "TC001862",  # COX5A
    "TC009255",  # COX5A
    "TC009596",  # COX5B
    "TC015888",  # COX7A
    "TC000453",  # COX7B
    "TC005368",  # COX7B
    "TC011750",  # COX7B
    "TC030047",  # COX7C
    "TC030048",  # COX8
    "TC003306",  # Cox11
    "TC006526",  # Cox17
    "TC010331",  # Cyt-c1
    "TC009556",  # ATPsynB
    "TC011455",  # ATPsynC
    "TC015454",  # ATPsynC
    "TC003886",  # ATPsynD
    "TC004872",  # ATPsynE
    "TC009033",  # ATPsynE
    "TC005628",  # ATPsynF
    "TC008513",  # ATPsynG
    "TC000462",  # ATPsynO
    "TC015322",  # ATPsynbeta
    "TC013931",  # ATPsyndelta
    "TC009010",  # ATPsyngamma
    "TC008728",  # blw
}


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEFAULT_MIN_GENES = 400
DEFAULT_MIN_COUNTS = 800
DEFAULT_MAX_GENES = 3000
DEFAULT_MIN_CELLS_PER_GENE = 3
DEFAULT_MAX_PCT_MT_PROXY = 15.0

N_TOP_HVGS = 2000
N_PCS = 30
N_NEIGHBORS = 15
LEIDEN_RES = 0.66
SUBCLUSTER_RES = 0.80

DEFAULT_MARKER_ROWS: List[Dict[str, str]] = [
    {"Tissue": "Visceral muscle", "Marker gene": "bHLH54F", "TC ID": "TC002349"},
    {"Tissue": "Visceral muscle", "Marker gene": "FasIII", "TC ID": "TC014942"},
    {"Tissue": "Visceral muscle", "Marker gene": "bagpipe", "TC ID": "TC012743"},
    {"Tissue": "Visceral muscle", "Marker gene": "mef2", "TC ID": "TC010850"},
    {"Tissue": "Perinephric membrane", "Marker gene": "Grip", "TC ID": "TC034397"},
    {"Tissue": "Perinephric membrane", "Marker gene": "org-1", "TC ID": "TC015327"},
    {"Tissue": "Perinephric membrane", "Marker gene": "FoxL1", "TC ID": "TC031071"},
    {"Tissue": "Perinephric membrane", "Marker gene": "twi", "TC ID": "TC014598"},
    {"Tissue": "Perinephric membrane", "Marker gene": "SoxD", "TC ID": "TC007419"},
    {"Tissue": "Perinephric membrane", "Marker gene": "FGFR", "TC ID": "TC004713"},
    {"Tissue": "Hindgut epithelium", "Marker gene": "byn", "TC ID": "TC014076"},
    {"Tissue": "Hindgut epithelium", "Marker gene": "engrailed", "TC ID": "TC009896"},
    {"Tissue": "Hindgut epithelium", "Marker gene": "otp", "TC ID": "TC033748"},
    {"Tissue": "Hindgut epithelium", "Marker gene": "sine oculis", "TC ID": "TC030468"},
    {"Tissue": "Hindgut epithelium", "Marker gene": "inv", "TC ID": "TC001164"},
    {"Tissue": "Principal cell", "Marker gene": "cut", "TC ID": "TC015699"},
    {"Tissue": "Principal cell", "Marker gene": "Tret1-2", "TC ID": "TC009681"},
    {"Tissue": "Proximal principal cell", "Marker gene": "SLC17", "TC ID": "TC014388"},
    {"Tissue": "Proximal principal cell", "Marker gene": "SLC22", "TC ID": "TC000302"},
    {"Tissue": "Proximal principal cell", "Marker gene": "Elk", "TC ID": "TC015502"},
    {"Tissue": "Proximal principal cell", "Marker gene": "sei", "TC ID": "TC015503"},
    {"Tissue": "Distal principal cell", "Marker gene": "Dac", "TC ID": "TC032755"},
    {"Tissue": "Distal principal cell", "Marker gene": "SLC44A1", "TC ID": "TC032513"},
    {"Tissue": "Distal principal cell", "Marker gene": "", "TC ID": "TC011259"},
    {"Tissue": "Secondary cell", "Marker gene": "capaR", "TC ID": "TC007170"},
    {"Tissue": "Secondary cell", "Marker gene": "pHCl-2", "TC ID": "TC006392"},
    {"Tissue": "Secondary cell", "Marker gene": "castor", "TC ID": "TC033414"},
    {"Tissue": "Secondary cell", "Marker gene": "Bx", "TC ID": "TC007525"},
    {"Tissue": "Inverse secondary cell", "Marker gene": "DH31R", "TC ID": "TC032502"},
    {"Tissue": "Inverse secondary cell", "Marker gene": "DH31R", "TC ID": "TC013321"},
    {"Tissue": "Proximal secondary cell", "Marker gene": "Urn8R", "TC ID": "TC034462"},
    {"Tissue": "Leptophragmata", "Marker gene": "NHA1", "TC ID": "TC013096"},
    {"Tissue": "Leptophragmata", "Marker gene": "Dac", "TC ID": "TC032755"},
    {"Tissue": "Leptophragmata", "Marker gene": "SLC19A1", "TC ID": "TC033143"},
    {"Tissue": "Leptophragmata", "Marker gene": "pHCl-2", "TC ID": "TC006393"},
    {"Tissue": "Leptophragmata", "Marker gene": "lim1", "TC ID": "TC014939"},
    {"Tissue": "Stem / progenitor cell", "Marker gene": "esg", "TC ID": "TC014474"},
    {"Tissue": "Midgut epithelium", "Marker gene": "GATAe", "TC ID": "TC010406"},
    {"Tissue": "Midgut EEC", "Marker gene": "Allatostatin C", "TC ID": "TC005428"},
    {"Tissue": "Midgut EEC", "Marker gene": "Tachykinin", "TC ID": "TC005685"},
    {"Tissue": "Midgut EEC", "Marker gene": "DH37/47", "TC ID": "TC030022"},
    {"Tissue": "Neuron", "Marker gene": "Sytbeta", "TC ID": "TC007583"},
    {"Tissue": "Neuron", "Marker gene": "nAChRalpha6", "TC ID": "TC013633"},
    {"Tissue": "Neuron", "Marker gene": "DH37/47", "TC ID": "TC030022"},
    {"Tissue": "Trachea", "Marker gene": "Gasp", "TC ID": "TC032197"},
]

SELECTION_GROUPS: Dict[str, List[str]] = {
    "hindgut cells": ["Hindgut epithelium"],
    "hindgut": ["Hindgut epithelium"],
    "principal cells": ["Principal cell", "Proximal principal cell", "Distal principal cell"],
    "principal cell": ["Principal cell", "Proximal principal cell", "Distal principal cell"],
    "secondary cells": ["Secondary cell", "Inverse secondary cell", "Proximal secondary cell"],
    "secondary cell": ["Secondary cell", "Inverse secondary cell", "Proximal secondary cell"],
    "enteroendocrine cells": ["Midgut EEC"],
    "enteroendocrine cell": ["Midgut EEC"],
    "eec": ["Midgut EEC"],
    "midgut eec": ["Midgut EEC"],
    "muscle cells": ["Visceral muscle"],
    "muscle cell": ["Visceral muscle"],
    "visceral muscle": ["Visceral muscle"],
    "perinephric membrane": ["Perinephric membrane"],
    "leptophragmata": ["Leptophragmata"],
    "trachea": ["Trachea"],
    "midgut epithelium": ["Midgut epithelium"],
    "enterocytes": ["Midgut epithelium"],
    "neuron": ["Neuron"],
    "neurons": ["Neuron"],
    "stem cells": ["Stem / progenitor cell"],
    "stem cell": ["Stem / progenitor cell"],
    "progenitor": ["Stem / progenitor cell"],
}


def require_scanpy() -> None:
    if sc is None:
        raise ImportError(
            "scanpy is not installed. Install it first, then rerun. Example: pip install scanpy"
        )


def norm_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def slugify(value: object) -> str:
    text = norm_text(value)
    return re.sub(r"\s+", "_", text) or "subset"


def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


# -----------------------------------------------------------------------------
# Marker table loading and matching
# -----------------------------------------------------------------------------
def default_marker_detail() -> pd.DataFrame:
    return pd.DataFrame(DEFAULT_MARKER_ROWS)


def load_marker_detail(marker_table_path: Optional[Path]) -> pd.DataFrame:
    if marker_table_path is None:
        return default_marker_detail()

    marker_table_path = Path(marker_table_path)
    if not marker_table_path.exists():
        print(f"[WARN] Marker table not found: {marker_table_path}. Using embedded defaults.")
        return default_marker_detail()

    try:
        suffix = marker_table_path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            xls = pd.ExcelFile(marker_table_path)
            if "Marker detail" in xls.sheet_names:
                df = pd.read_excel(marker_table_path, sheet_name="Marker detail")
                needed = {"Tissue", "Marker gene", "TC ID"}
                if needed.issubset(df.columns):
                    return df[["Tissue", "Marker gene", "TC ID"]].copy()
            if "Final marker table" in xls.sheet_names:
                final_df = pd.read_excel(marker_table_path, sheet_name="Final marker table")
                rows: List[Dict[str, str]] = []
                for _, row in final_df.iterrows():
                    tissue = row.get("Tissue", "")
                    genes = [g.strip() for g in str(row.get("Marker genes", "")).split(",") if g and str(g).strip()]
                    tc_ids = [t.strip() for t in str(row.get("TC IDs", "")).split(",") if t and str(t).strip()]
                    max_len = max(len(genes), len(tc_ids), 1)
                    for i in range(max_len):
                        rows.append(
                            {
                                "Tissue": tissue,
                                "Marker gene": genes[i] if i < len(genes) else "",
                                "TC ID": tc_ids[i] if i < len(tc_ids) else "",
                            }
                        )
                if rows:
                    return pd.DataFrame(rows)
        elif suffix == ".csv":
            df = pd.read_csv(marker_table_path)
            needed = {"Tissue", "Marker gene", "TC ID"}
            if needed.issubset(df.columns):
                return df[["Tissue", "Marker gene", "TC ID"]].copy()
    except Exception as exc:
        print(f"[WARN] Could not read marker table ({exc}). Using embedded defaults.")

    return default_marker_detail()


def load_tc_mapping(mapping_tsv: Optional[Path]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    if mapping_tsv is None:
        return {}, {}

    mapping_tsv = Path(mapping_tsv)
    if not mapping_tsv.exists():
        print(f"[WARN] Mapping TSV not found: {mapping_tsv}. Continuing without TC↔symbol expansion.")
        return {}, {}

    try:
        df = pd.read_csv(mapping_tsv, sep="\t")
    except Exception as exc:
        print(f"[WARN] Could not read mapping TSV ({exc}). Continuing without TC↔symbol expansion.")
        return {}, {}

    tc_to_genes: Dict[str, List[str]] = {}
    gene_to_tcs: Dict[str, List[str]] = {}

    if {"TC_ID", "gene_symbol"}.issubset(df.columns):
        clean = df[["TC_ID", "gene_symbol"]].copy()
        clean = clean.dropna(subset=["TC_ID", "gene_symbol"])
        clean["TC_ID"] = clean["TC_ID"].astype(str).str.strip()
        clean["gene_symbol"] = clean["gene_symbol"].astype(str).str.strip()

        for tc_id, sub in clean.groupby("TC_ID"):
            tc_to_genes[tc_id] = unique_keep_order(sub["gene_symbol"].tolist())
        for gene_symbol, sub in clean.groupby("gene_symbol"):
            gene_to_tcs[gene_symbol] = unique_keep_order(sub["TC_ID"].tolist())

    return tc_to_genes, gene_to_tcs


def build_feature_lookup(adata: "sc.AnnData", use_raw: bool = True) -> Dict[str, str]:
    source = adata.raw if (use_raw and adata.raw is not None) else adata
    lookup: Dict[str, str] = {}

    def add_alias(alias: object, actual: str) -> None:
        key = norm_text(alias)
        if key and key not in lookup:
            lookup[key] = actual

    for actual in source.var_names.astype(str):
        add_alias(actual, actual)

    var_df = source.var.copy()
    for col in ["gene_ids", "gene_id", "feature_id", "gene_symbols", "gene_symbol", "symbol", "feature_name"]:
        if col in var_df.columns:
            for actual, alias in zip(source.var_names.astype(str), var_df[col].astype(str)):
                add_alias(alias, actual)

    return lookup


def marker_terms_for_row(
    row: pd.Series,
    tc_to_genes: Dict[str, List[str]],
    gene_to_tcs: Dict[str, List[str]],
) -> List[str]:
    terms: List[str] = []
    marker_gene = str(row.get("Marker gene", "") or "").strip()
    tc_id = str(row.get("TC ID", "") or "").strip()

    if marker_gene:
        terms.append(marker_gene)
        terms.extend(gene_to_tcs.get(marker_gene, []))
    if tc_id:
        terms.append(tc_id)
        terms.extend(tc_to_genes.get(tc_id, []))

    return unique_keep_order([term for term in terms if term])


def build_tissue_marker_definitions(
    adata: "sc.AnnData",
    marker_df: pd.DataFrame,
    tc_to_genes: Dict[str, List[str]],
    gene_to_tcs: Dict[str, List[str]],
    use_raw: bool = True,
) -> Dict[str, Dict[str, object]]:
    lookup = build_feature_lookup(adata, use_raw=use_raw)
    definitions: Dict[str, Dict[str, object]] = {}

    for tissue, sub in marker_df.groupby("Tissue", sort=False):
        features: List[str] = []
        marker_genes_present: List[str] = []
        tc_ids_present: List[str] = []
        total_marker_rows = int(len(sub))
        matched_rows = 0

        for _, row in sub.iterrows():
            terms = marker_terms_for_row(row, tc_to_genes=tc_to_genes, gene_to_tcs=gene_to_tcs)
            matched_feature_for_row: Optional[str] = None
            for term in terms:
                actual = lookup.get(norm_text(term))
                if actual is not None:
                    matched_feature_for_row = actual
                    break

            if matched_feature_for_row is not None:
                matched_rows += 1
                features.append(matched_feature_for_row)
                marker_gene = str(row.get("Marker gene", "") or "").strip()
                tc_id = str(row.get("TC ID", "") or "").strip()
                if marker_gene:
                    marker_genes_present.append(marker_gene)
                if tc_id:
                    tc_ids_present.append(tc_id)

        definitions[tissue] = {
            "features": unique_keep_order(features),
            "marker_genes_present": unique_keep_order(marker_genes_present),
            "tc_ids_present": unique_keep_order(tc_ids_present),
            "total_marker_rows": total_marker_rows,
            "matched_rows": matched_rows,
        }

    return definitions


# -----------------------------------------------------------------------------
# QC / plot helpers
# -----------------------------------------------------------------------------
def save_current_figure(outbase: Path, dpi: int = 300) -> None:
    outbase.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outbase.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.savefig(outbase.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close()


def add_qc_metrics(adata: "sc.AnnData") -> Tuple[int, int]:
    vn = adata.var_names.to_series().astype(str)

    ribo_mask = vn.str.startswith(("RPL", "RPS", "RpL", "RpS"))
    adata.var["ribo"] = ribo_mask.to_numpy(dtype=bool)

    mt_mask = vn.isin(STRICT_MT_TC)
    adata.var["mt"] = mt_mask.to_numpy(dtype=bool)

    qc_vars = []
    if ribo_mask.any():
        qc_vars.append("ribo")
    if mt_mask.any():
        qc_vars.append("mt")
    sc.pp.calculate_qc_metrics(adata, qc_vars=qc_vars, inplace=True)
    return int(ribo_mask.sum()), int(mt_mask.sum())


def qc_plots(adata: "sc.AnnData", outdir: Path, prefix: str, title: str) -> None:
    sc.set_figure_params(dpi=130, fontsize=10, frameon=False)
    keys = [
        "total_counts",
        "n_genes_by_counts",
        "pct_counts_mt",
        "pct_counts_ribo",
        "pct_counts_in_top_20_genes",
    ]
    keys = [k for k in keys if k in adata.obs.columns]
    sc.pl.violin(adata, keys, jitter=0.35, multi_panel=True, show=False)
    plt.suptitle(title, y=1.02)
    save_current_figure(outdir / f"{prefix}_QC_violin")

    sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts", show=False)
    plt.title(f"{title} — total_counts vs n_genes")
    save_current_figure(outdir / f"{prefix}_QC_counts_vs_genes")

    if "pct_counts_ribo" in adata.obs.columns:
        sc.pl.scatter(adata, x="total_counts", y="pct_counts_ribo", show=False)
        plt.title(f"{title} — total_counts vs pct_counts_ribo")
        save_current_figure(outdir / f"{prefix}_QC_counts_vs_ribo")

    if "pct_counts_mt" in adata.obs.columns:
        sc.pl.scatter(adata, x="total_counts", y="pct_counts_mt", show=False)
        plt.title(f"{title} — total_counts vs pct_counts_mt_proxy")
        save_current_figure(outdir / f"{prefix}_QC_counts_vs_mt_proxy")


# -----------------------------------------------------------------------------
# Core scanpy pipeline
# -----------------------------------------------------------------------------
def run_umap_clustering(
    adata: "sc.AnnData",
    cluster_key: str = "leiden",
    n_top_hvgs: int = N_TOP_HVGS,
    n_pcs: int = N_PCS,
    n_neighbors: int = N_NEIGHBORS,
    leiden_res: float = LEIDEN_RES,
) -> Tuple["sc.AnnData", str]:
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_hvgs, flavor="seurat")
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10, zero_center=False)
    sc.tl.pca(adata, n_comps=n_pcs)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(adata)
    try:
        sc.tl.leiden(
            adata,
            resolution=leiden_res,
            flavor="igraph",
            n_iterations=2,
            directed=False,
            key_added=cluster_key,
        )
        backend = "igraph"
    except Exception:
        sc.tl.leiden(adata, resolution=leiden_res, key_added=cluster_key)
        backend = "default"
    return adata, backend


def save_umap_overview(
    adata: "sc.AnnData",
    outdir: Path,
    prefix: str,
    leiden_backend: str,
    cluster_key: str = "leiden",
) -> None:
    sc.set_figure_params(dpi=140, fontsize=10, frameon=False)
    colour_list = [
        c for c in [cluster_key, "total_counts", "n_genes_by_counts", "pct_counts_in_top_20_genes", "pct_counts_mt", "pct_counts_ribo"]
        if c in adata.obs.columns
    ]
    sc.pl.umap(
        adata,
        color=colour_list,
        ncols=2,
        wspace=0.35,
        size=10,
        legend_loc="on data",
        show=False,
    )
    plt.suptitle(f"UMAP ({prefix}) — resolution={LEIDEN_RES} ({leiden_backend})", y=1.02)
    save_current_figure(outdir / f"{prefix}_UMAP_overview")


def save_annotation_umaps(adata: "sc.AnnData", outdir: Path, prefix: str, cluster_key: str = "leiden") -> None:
    if "cell_type" not in adata.obs.columns:
        return

    sc.set_figure_params(dpi=140, fontsize=10, frameon=False)

    sc.pl.umap(adata, color=cluster_key, legend_loc="on data", size=10, show=False)
    plt.title(f"{prefix} — {cluster_key}")
    save_current_figure(outdir / f"{prefix}_UMAP_{cluster_key}")

    sc.pl.umap(adata, color="cell_type", legend_loc="right margin", size=10, show=False)
    plt.title(f"{prefix} — cell type")
    save_current_figure(outdir / f"{prefix}_UMAP_cell_type")

    if "cluster_cell_type" in adata.obs.columns:
        sc.pl.umap(adata, color="cluster_cell_type", legend_loc="right margin", size=10, show=False)
        plt.title(f"{prefix} — cluster + cell type")
        save_current_figure(outdir / f"{prefix}_UMAP_cluster_cell_type")


def export_markers(
    adata: "sc.AnnData",
    outdir: Path,
    cluster_key: str = "leiden",
    top_n: int = 20,
    prefix: str = "markers",
) -> pd.DataFrame:
    sc.tl.rank_genes_groups(adata, groupby=cluster_key, method="wilcoxon")
    dfs = []
    for cl in adata.obs[cluster_key].cat.categories:
        df = sc.get.rank_genes_groups_df(adata, group=cl)
        df.insert(0, cluster_key, cl)
        dfs.append(df)
    all_markers = pd.concat(dfs, ignore_index=True)
    all_markers.to_csv(outdir / f"{prefix}_all_clusters.csv", index=False)

    top_list = []
    for cl in adata.obs[cluster_key].cat.categories:
        df = all_markers[all_markers[cluster_key] == cl].copy()
        if "pvals_adj" in df.columns:
            df = df[df["pvals_adj"] < 0.05]
        if "logfoldchanges" in df.columns:
            df = df[df["logfoldchanges"] > 0]
        top_list.append(df.head(top_n))
    top_markers = pd.concat(top_list, ignore_index=True)
    top_markers.to_csv(outdir / f"{prefix}_top{top_n}_per_cluster.csv", index=False)

    try:
        all_markers.to_excel(outdir / f"{prefix}_all_clusters.xlsx", index=False)
        top_markers.to_excel(outdir / f"{prefix}_top{top_n}_per_cluster.xlsx", index=False)
    except Exception:
        pass

    return top_markers


# -----------------------------------------------------------------------------
# Table-based annotation
# -----------------------------------------------------------------------------
def score_tissues_from_marker_table(
    adata: "sc.AnnData",
    marker_df: pd.DataFrame,
    tc_to_genes: Dict[str, List[str]],
    gene_to_tcs: Dict[str, List[str]],
    cluster_key: str = "leiden",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    if cluster_key not in adata.obs.columns:
        raise ValueError(f"Cluster column '{cluster_key}' not found in adata.obs")

    definitions = build_tissue_marker_definitions(
        adata,
        marker_df=marker_df,
        tc_to_genes=tc_to_genes,
        gene_to_tcs=gene_to_tcs,
        use_raw=True,
    )

    source = adata.raw if adata.raw is not None else adata
    cluster_series = adata.obs[cluster_key].astype(str)
    clusters = pd.Categorical(cluster_series).categories.tolist()
    rows: List[Dict[str, object]] = []

    for cluster in clusters:
        cell_mask = (cluster_series == str(cluster)).to_numpy()
        n_cells = int(cell_mask.sum())

        for tissue, meta in definitions.items():
            features = list(meta["features"])
            total_marker_rows = int(meta["total_marker_rows"])
            matched_rows = int(meta["matched_rows"])

            if not features:
                mean_expr = np.nan
                score = np.nan
            else:
                expr = source[cell_mask, features].X
                if not isinstance(expr, np.ndarray):
                    expr = expr.toarray()
                mean_expr = float(expr.mean())
                coverage = matched_rows / max(total_marker_rows, 1)
                score = mean_expr * coverage

            rows.append(
                {
                    "cluster": str(cluster),
                    "n_cells": n_cells,
                    "tissue": tissue,
                    "score": score,
                    "mean_expression": mean_expr,
                    "matched_rows": matched_rows,
                    "total_marker_rows": total_marker_rows,
                    "matched_marker_genes": ", ".join(meta["marker_genes_present"]),
                    "matched_tc_ids": ", ".join(meta["tc_ids_present"]),
                    "features_used": ", ".join(features),
                }
            )

    return pd.DataFrame(rows), definitions


def choose_best_annotations(scores_df: pd.DataFrame) -> pd.DataFrame:
    if scores_df.empty:
        return pd.DataFrame(columns=["cluster", "n_cells", "cell_type"])

    annotations: List[Dict[str, object]] = []
    for cluster, sub in scores_df.groupby("cluster", sort=False):
        sub = sub.copy()
        sub = sub.sort_values(
            ["score", "matched_rows", "mean_expression", "tissue"],
            ascending=[False, False, False, True],
        )
        valid = sub[sub["score"].notna()].copy()

        if valid.empty or float(valid.iloc[0]["score"]) <= 0:
            cell_type = "Unknown"
            best_row = None
        else:
            best_row = valid.iloc[0]
            cell_type = str(best_row["tissue"])

        top3 = (
            " | ".join(f"{row['tissue']} ({row['score']:.3f})" for _, row in valid.head(3).iterrows())
            if not valid.empty
            else ""
        )

        annotations.append(
            {
                "cluster": str(cluster),
                "n_cells": int(sub.iloc[0]["n_cells"]),
                "cell_type": cell_type,
                "best_score": (float(best_row["score"]) if best_row is not None else np.nan),
                "best_mean_expression": (float(best_row["mean_expression"]) if best_row is not None else np.nan),
                "matched_rows": (int(best_row["matched_rows"]) if best_row is not None else 0),
                "total_marker_rows": (int(best_row["total_marker_rows"]) if best_row is not None else 0),
                "matched_marker_genes": (str(best_row["matched_marker_genes"]) if best_row is not None else ""),
                "matched_tc_ids": (str(best_row["matched_tc_ids"]) if best_row is not None else ""),
                "top3_predictions": top3,
            }
        )

    annot_df = pd.DataFrame(annotations)
    annot_df["cluster_cell_type"] = annot_df["cluster"].astype(str) + " | " + annot_df["cell_type"].astype(str)
    return annot_df


def apply_annotations_to_adata(adata: "sc.AnnData", annot_df: pd.DataFrame, cluster_key: str = "leiden") -> "sc.AnnData":
    mapping = dict(zip(annot_df["cluster"].astype(str), annot_df["cell_type"].astype(str)))
    label_mapping = dict(zip(annot_df["cluster"].astype(str), annot_df["cluster_cell_type"].astype(str)))
    adata.obs[cluster_key] = adata.obs[cluster_key].astype(str).astype("category")
    adata.obs["cell_type"] = adata.obs[cluster_key].astype(str).map(mapping).fillna("Unknown").astype("category")
    adata.obs["cluster_cell_type"] = adata.obs[cluster_key].astype(str).map(label_mapping).fillna("Unknown").astype("category")
    return adata


def export_table_based_annotation_outputs(
    adata: "sc.AnnData",
    outdir: Path,
    marker_df: pd.DataFrame,
    tc_to_genes: Dict[str, List[str]],
    gene_to_tcs: Dict[str, List[str]],
    cluster_key: str = "leiden",
    prefix: str = "cluster",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scores_df, _definitions = score_tissues_from_marker_table(
        adata,
        marker_df=marker_df,
        tc_to_genes=tc_to_genes,
        gene_to_tcs=gene_to_tcs,
        cluster_key=cluster_key,
    )
    annot_df = choose_best_annotations(scores_df)
    apply_annotations_to_adata(adata, annot_df, cluster_key=cluster_key)

    scores_df.to_csv(outdir / f"{prefix}_tissue_scores_table_based.csv", index=False)
    annot_df.to_csv(outdir / f"{prefix}_annotations_table_based.csv", index=False)

    try:
        with pd.ExcelWriter(outdir / f"{prefix}_annotations_table_based.xlsx") as writer:
            annot_df.to_excel(writer, sheet_name="Cluster annotation", index=False)
            scores_df.to_excel(writer, sheet_name="All tissue scores", index=False)
            marker_df.to_excel(writer, sheet_name="Marker detail used", index=False)
    except Exception:
        pass

    return scores_df, annot_df


# -----------------------------------------------------------------------------
# Gene panel
# -----------------------------------------------------------------------------
def parse_gene_list(genes_arg: str) -> List[str]:
    return [g.strip() for g in str(genes_arg).split(",") if g and str(g).strip()]


def plot_gene_panel(
    adata: "sc.AnnData",
    genes: List[str],
    outdir: Path,
    prefix: str,
    groupby: str = "leiden",
) -> None:
    if adata.raw is not None:
        valid_genes = [g for g in genes if g in adata.raw.var_names]
    else:
        valid_genes = [g for g in genes if g in adata.var_names]
    if not valid_genes:
        print("[INFO] No valid genes provided for plotting. Skipping gene panel.")
        return

    n_genes = len(valid_genes)
    ncols = 4
    nrows = int(np.ceil(n_genes / ncols))

    sc.set_figure_params(dpi=140, fontsize=9, frameon=False)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for i, gene in enumerate(valid_genes):
        ax = axes[i]
        try:
            sc.pl.umap(
                adata,
                color=gene,
                ax=ax,
                show=False,
                frameon=False,
                title=gene,
                use_raw=True,
                colorbar_loc=None,
            )
        except TypeError:
            sc.pl.umap(
                adata,
                color=gene,
                ax=ax,
                show=False,
                frameon=False,
                title=gene,
                use_raw=True,
            )

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Gene expression UMAP: {', '.join(valid_genes)}", y=1.02)
    save_current_figure(outdir / f"{prefix}_UMAP_panel")

    sc.set_figure_params(dpi=130, fontsize=9, frameon=False)
    sc.pl.violin(
        adata,
        keys=valid_genes,
        groupby=groupby,
        jitter=0.25,
        multi_panel=True,
        show=False,
        use_raw=True,
    )
    plt.suptitle(f"Gene expression violin: {', '.join(valid_genes)}", y=1.02)
    save_current_figure(outdir / f"{prefix}_Violin_panel")


# -----------------------------------------------------------------------------
# Subclustering helpers
# -----------------------------------------------------------------------------
def resolve_selection_to_mask(adata: "sc.AnnData", selector: str, cluster_key: str = "leiden") -> Tuple[np.ndarray, str]:
    raw_selector = str(selector).strip()
    selector_norm = norm_text(raw_selector)

    if cluster_key not in adata.obs.columns:
        raise ValueError(f"'{cluster_key}' not found in input h5ad. Run the main workflow first.")

    cluster_values = adata.obs[cluster_key].astype(str)
    if raw_selector in set(cluster_values.tolist()):
        return (cluster_values == raw_selector).to_numpy(), f"cluster_{raw_selector}"

    if selector_norm in set(norm_text(x) for x in cluster_values.tolist()):
        mask = cluster_values.map(norm_text) == selector_norm
        return mask.to_numpy(), f"cluster_{raw_selector}"

    if "cell_type" in adata.obs.columns:
        cell_types = adata.obs["cell_type"].astype(str)
        cell_type_norm = cell_types.map(norm_text)

        if selector_norm in set(cell_type_norm.tolist()):
            matched_label = cell_types[cell_type_norm == selector_norm].iloc[0]
            return (cell_type_norm == selector_norm).to_numpy(), matched_label

        if selector_norm in SELECTION_GROUPS:
            targets = [norm_text(x) for x in SELECTION_GROUPS[selector_norm]]
            mask = cell_type_norm.isin(targets)
            if bool(mask.any()):
                label = ", ".join(SELECTION_GROUPS[selector_norm])
                return mask.to_numpy(), label

        contains_mask = cell_type_norm.map(lambda x: selector_norm in x if selector_norm else False)
        if bool(contains_mask.any()):
            label = cell_types[contains_mask].iloc[0]
            return contains_mask.to_numpy(), label

    raise ValueError(
        f"Could not match --cluster '{selector}'. Use a cluster number (e.g. 1) or a cell type name such as 'hindgut cells'."
    )


# -----------------------------------------------------------------------------
# Main workflow and subclustering workflow
# -----------------------------------------------------------------------------
def save_loom_safe(adata: "sc.AnnData", path: Path) -> None:
    try:
        try:
            adata.X = adata.X.astype(np.float32)
        except Exception:
            pass
        adata.write_loom(str(path), write_obsm_varm=True)
        print(f"[INFO] Loom written to {path}")
    except Exception as exc:
        print(f"[WARN] Could not write loom file {path}: {exc}")


def run_main_workflow(args: argparse.Namespace) -> None:
    require_scanpy()
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    marker_df = load_marker_detail(Path(args.marker_table) if args.marker_table else None)
    tc_to_genes, gene_to_tcs = load_tc_mapping(Path(args.mapping_tsv) if args.mapping_tsv else None)

    mtx_path = Path(args.mtx_dir)
    print(f"[INFO] Loading data from {mtx_path} with gene_ids as var_names ...")
    adata = sc.read_10x_mtx(mtx_path, var_names="gene_ids", cache=True)
    adata.var_names_make_unique()
    print(f"[INFO] Loaded {adata.n_obs} cells × {adata.n_vars} genes")

    rb_n, mt_n = add_qc_metrics(adata)
    print(f"[INFO] Detected {rb_n} ribosomal genes")
    print(f"[INFO] Detected {mt_n} proxy mitochondrial genes from curated TC list")
    qc_plots(adata, outdir, "BEFORE", "QC BEFORE filtering")

    print("[INFO] Filtering cells and genes ...")
    cell_mask = (
        (adata.obs["n_genes_by_counts"] >= args.min_genes)
        & (adata.obs["total_counts"] >= args.min_counts)
        & (adata.obs["n_genes_by_counts"] <= args.max_genes)
    )
    if "pct_counts_mt" in adata.obs.columns and args.max_pct_mt_proxy >= 0:
        cell_mask = cell_mask & (adata.obs["pct_counts_mt"] <= args.max_pct_mt_proxy)

    adata = adata[cell_mask].copy()
    sc.pp.filter_genes(adata, min_cells=args.min_cells_per_gene)
    print(f"[INFO] After filtering: {adata.n_obs} cells × {adata.n_vars} genes")

    rb_n_after, mt_n_after = add_qc_metrics(adata)
    print(f"[INFO] After filtering: {rb_n_after} ribosomal genes, {mt_n_after} proxy mitochondrial genes")
    qc_plots(adata, outdir, "AFTER", "QC AFTER filtering")

    filtered_counts_path = outdir / "filtered_counts_for_subcluster.h5ad"
    adata.write(filtered_counts_path)

    print("[INFO] Running dimensionality reduction, UMAP and clustering ...")
    adata_proc, leiden_backend = run_umap_clustering(
        adata,
        cluster_key="leiden",
        n_top_hvgs=args.n_top_hvgs,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        leiden_res=args.leiden_res,
    )
    save_umap_overview(adata_proc, outdir, "AFTER", leiden_backend, cluster_key="leiden")

    print("[INFO] Exporting ranked markers ...")
    export_markers(adata_proc, outdir, cluster_key="leiden", top_n=args.top_markers, prefix="markers")

    print("[INFO] Running table-based annotation ...")
    _scores_df, annot_df = export_table_based_annotation_outputs(
        adata_proc,
        outdir,
        marker_df=marker_df,
        tc_to_genes=tc_to_genes,
        gene_to_tcs=gene_to_tcs,
        cluster_key="leiden",
        prefix="cluster",
    )
    save_annotation_umaps(adata_proc, outdir, prefix="AFTER", cluster_key="leiden")

    if args.genes:
        groupby = "cluster_cell_type" if "cluster_cell_type" in adata_proc.obs.columns else "leiden"
        gene_list = parse_gene_list(args.genes)
        print(f"[INFO] Plotting gene panel for: {', '.join(gene_list)}")
        plot_gene_panel(adata_proc, gene_list, outdir, "SelectedGenes", groupby=groupby)

    main_h5ad = outdir / "final_filtered_umap_leiden_annotated.h5ad"
    adata_proc.write(main_h5ad)
    print(f"[INFO] Main annotated object saved to {main_h5ad}")

    main_loom = outdir / "final_filtered_umap_leiden_annotated_TC.loom"
    save_loom_safe(adata_proc, main_loom)

    adata_subset = sc.read_h5ad(filtered_counts_path)
    annot_map = dict(zip(annot_df["cluster"].astype(str), annot_df["cell_type"].astype(str)))
    label_map = dict(zip(annot_df["cluster"].astype(str), annot_df["cluster_cell_type"].astype(str)))
    adata_subset.obs["leiden"] = adata_proc.obs["leiden"].astype(str).values
    adata_subset.obs["cell_type"] = adata_subset.obs["leiden"].astype(str).map(annot_map).fillna("Unknown")
    adata_subset.obs["cluster_cell_type"] = adata_subset.obs["leiden"].astype(str).map(label_map).fillna("Unknown")
    subset_h5ad = outdir / "filtered_counts_with_labels.h5ad"
    adata_subset.write(subset_h5ad)

    print("[INFO] Analysis complete.")
    print(f"[INFO] Reusable raw-count object for subclustering: {subset_h5ad}")


def run_subclustering_workflow(args: argparse.Namespace) -> None:
    require_scanpy()
    outdir = Path(args.out_dir)
    marker_df = load_marker_detail(Path(args.marker_table) if args.marker_table else None)
    tc_to_genes, gene_to_tcs = load_tc_mapping(Path(args.mapping_tsv) if args.mapping_tsv else None)

    input_h5ad = Path(args.input_h5ad) if args.input_h5ad else outdir / "filtered_counts_with_labels.h5ad"
    if not input_h5ad.exists():
        raise FileNotFoundError(
            f"Could not find {input_h5ad}. Run the main workflow first, or pass --input_h5ad explicitly."
        )

    print(f"[INFO] Loading QC-filtered object for subclustering: {input_h5ad}")
    adata = sc.read_h5ad(input_h5ad)
    mask, selection_label = resolve_selection_to_mask(adata, args.cluster, cluster_key="leiden")
    subset = adata[mask].copy()
    print(f"[INFO] Subsetting '{selection_label}' -> {subset.n_obs} cells")

    if subset.n_obs < 10:
        raise ValueError(f"Subset '{selection_label}' contains only {subset.n_obs} cells. Too few for reclustering.")

    for slot in ["X_pca", "X_umap"]:
        if slot in subset.obsm:
            del subset.obsm[slot]
    for slot in ["neighbors"]:
        if slot in subset.uns:
            del subset.uns[slot]
    for slot in ["distances", "connectivities"]:
        if slot in subset.obsp:
            del subset.obsp[slot]

    subset_dir = outdir / f"subcluster_{slugify(selection_label)}"
    subset_dir.mkdir(parents=True, exist_ok=True)
    subset.write(subset_dir / "subset_counts_before_recluster.h5ad")

    rb_n, mt_n = add_qc_metrics(subset)
    print(f"[INFO] Subset QC markers detected: ribo={rb_n}, mt_proxy={mt_n}")
    qc_plots(subset, subset_dir, "SUBSET", f"QC subset: {selection_label}")

    subset_proc, leiden_backend = run_umap_clustering(
        subset,
        cluster_key="subcluster",
        n_top_hvgs=args.n_top_hvgs,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        leiden_res=args.subcluster_res,
    )
    save_umap_overview(subset_proc, subset_dir, "SUBSET", leiden_backend, cluster_key="subcluster")

    export_markers(
        subset_proc,
        subset_dir,
        cluster_key="subcluster",
        top_n=args.top_markers,
        prefix="subcluster_markers",
    )

    _scores_df, annot_df = export_table_based_annotation_outputs(
        subset_proc,
        subset_dir,
        marker_df=marker_df,
        tc_to_genes=tc_to_genes,
        gene_to_tcs=gene_to_tcs,
        cluster_key="subcluster",
        prefix="subcluster",
    )
    save_annotation_umaps(subset_proc, subset_dir, prefix="SUBSET", cluster_key="subcluster")

    if args.genes:
        groupby = "cluster_cell_type" if "cluster_cell_type" in subset_proc.obs.columns else "subcluster"
        gene_list = parse_gene_list(args.genes)
        plot_gene_panel(subset_proc, gene_list, subset_dir, "SubsetGenes", groupby=groupby)

    sub_h5ad = subset_dir / "subcluster_annotated.h5ad"
    subset_proc.write(sub_h5ad)
    save_loom_safe(subset_proc, subset_dir / "subcluster_annotated_TC.loom")

    annot_map = dict(zip(annot_df["cluster"].astype(str), annot_df["cell_type"].astype(str)))
    label_map = dict(zip(annot_df["cluster"].astype(str), annot_df["cluster_cell_type"].astype(str)))
    subset.obs["subcluster"] = subset_proc.obs["subcluster"].astype(str).values
    subset.obs["cell_type"] = subset.obs["subcluster"].astype(str).map(annot_map).fillna("Unknown")
    subset.obs["cluster_cell_type"] = subset.obs["subcluster"].astype(str).map(label_map).fillna("Unknown")
    subset.write(subset_dir / "subset_counts_with_subcluster_labels.h5ad")

    with open(subset_dir / "run_info.txt", "w", encoding="utf-8") as fh:
        fh.write(f"Parent selection: {selection_label}\n")
        fh.write(f"Cells in subset: {subset.n_obs}\n")
        fh.write(f"Subcluster resolution: {args.subcluster_res}\n")
        fh.write(f"Proxy mitochondrial filter in main workflow: <= {args.max_pct_mt_proxy}\n")

    print("[INFO] Subclustering complete.")
    print(f"[INFO] Results saved to: {subset_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run scRNA-seq workflow with table-based annotation, QC, proxy mitochondrial scoring, and optional subclustering."
    )
    parser.add_argument("--mtx_dir", type=str, default=MTX_DIR, help="Path to the 10x filtered_feature_bc_matrix directory")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR, help="Directory where outputs will be written")
    parser.add_argument("--marker_table", type=str, default=MARKER_TABLE, help="Path to the curated marker table (.xlsx or .csv)")
    parser.add_argument("--mapping_tsv", type=str, default=MAPPING_TSV, help="Path to TC↔gene mapping TSV")
    parser.add_argument("--input_h5ad", type=str, default="", help="Optional h5ad input for subclustering mode")
    parser.add_argument("--cluster", type=str, default="", help='If set, run subclustering on this cluster number or cell type label, e.g. --cluster 1 or --cluster "hindgut cells"')
    parser.add_argument("--min_genes", type=int, default=DEFAULT_MIN_GENES, help="Minimum genes per cell")
    parser.add_argument("--min_counts", type=int, default=DEFAULT_MIN_COUNTS, help="Minimum counts per cell")
    parser.add_argument("--max_genes", type=int, default=DEFAULT_MAX_GENES, help="Maximum genes per cell")
    parser.add_argument("--min_cells_per_gene", type=int, default=DEFAULT_MIN_CELLS_PER_GENE, help="Minimum cells per gene")
    parser.add_argument("--max_pct_mt_proxy", type=float, default=DEFAULT_MAX_PCT_MT_PROXY, help="Maximum allowed pct_counts_mt from the curated proxy mitochondrial TC list; set -1 to disable this filter")
    parser.add_argument("--top_markers", type=int, default=20, help="Top marker genes to export per cluster")
    parser.add_argument("--n_top_hvgs", type=int, default=N_TOP_HVGS, help="Number of HVGs")
    parser.add_argument("--n_pcs", type=int, default=N_PCS, help="Number of PCs")
    parser.add_argument("--n_neighbors", type=int, default=N_NEIGHBORS, help="Number of neighbours")
    parser.add_argument("--leiden_res", type=float, default=LEIDEN_RES, help="Leiden resolution for main clustering")
    parser.add_argument("--subcluster_res", type=float, default=SUBCLUSTER_RES, help="Leiden resolution for subclustering")
    parser.add_argument("--genes", type=str, default="", help="Optional comma-separated genes to plot on UMAP/violin")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if str(args.cluster).strip():
        run_subclustering_workflow(args)
    else:
        run_main_workflow(args)


if __name__ == "__main__":
    main()
