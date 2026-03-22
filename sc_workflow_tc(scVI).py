"""
Refined version of ``sc_workflow.py`` that retains all of the original
functionality (QC, dimensionality reduction, clustering, marker analysis
and gene panel plots) but additionally ensures that Tribolium gene
identifiers (e.g. ``TC016185``) are used as the primary variable names
and writes a corresponding loom file with those identifiers.

Key differences to the original script:

* When reading the 10x matrix, ``var_names="gene_ids"`` is used rather
  than ``gene_symbols``.  This uses the first column of
  ``features.tsv.gz`` (containing TCxxx IDs) as ``adata.var_names``
  instead of the gene symbols.  The gene symbols remain available in
  ``adata.var['gene_symbols']``.  See the Scanpy documentation for
  details on the ``var_names`` option【103275241006487†L315-L318】.
* After performing the analysis and writing the filtered h5ad file, the
  script writes a loom file named ``final_filtered_umap_leiden_TC.loom``
  in the output directory.  This loom file retains UMAP embeddings and
  other var/obs matrices via ``write_obsm_varm=True``, and because
  ``adata.var_names`` are TC IDs, downstream viewers will display
  TCxxx rather than numeric indices.

Usage is identical to the original script:

    python sc_workflow_tc.py --mtx_dir /path/to/filtered_feature_bc_matrix \
                             --out_dir /path/to/output_directory

All other options and interactive prompts remain unchanged.  The loom
file will be saved alongside the other outputs.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import scvi
from scvi.model import SCVI

# -----------------------------------------------------------------------------
# Proxy mitochondrial stress gene set
# -----------------------------------------------------------------------------
#
# The data matrix for this project contains only Tribolium nuclear genes with
# TC identifiers (e.g. TC016185) and does not include explicit mitochondrial
# genome transcripts such as mt:CoI or mt:ND2.  Consequently, the standard
# scRNA‑seq quality control metric `pct_counts_mt` cannot be computed from
# mitochondrial‑encoded transcripts directly.  To approximate mitochondrial
# stress, we use a curated list of TC genes whose Drosophila orthologs encode
# core mitochondrial oxidative phosphorylation and ATP synthase components.
# These genes represent proteins in the electron transport chain (e.g.
# cytochrome c oxidase subunits COX4/5/7/8), respiratory chain assembly factors
# (Cox11, Cox17), cytochrome bc1 complex (Cyt‑c1), and ATP synthase subunits
# (ATPsynB/C/D/E/F/G/O/beta/delta/gamma, blw).  Since these genes are
# nuclear‑encoded, the resulting `pct_counts_mt` should be interpreted as a
# proxy stress indicator rather than a true mitochondrial transcript
# percentage.

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
    "TC010331",  # Cyt‑c1
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

# ========= EDIT PATHS =========
MTX_DIR = (r"C:\Users\dmp753\Desktop\ScSeq filtered counts\filtered_feature_bc_matrix_larva"
)
OUT_DIR = (r"C:\Users\dmp753\Desktop\ScSeq filtered counts\output")
# ==============================


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Default quality control thresholds.  These can be overridden on the
# command line via --min_genes, --min_counts and --max_genes.  The
# numbers below were chosen after inspecting QC plots: they remove
# low‑quality droplets and extreme outliers while retaining most
# genuine cells.
DEFAULT_MIN_GENES = 800
DEFAULT_MIN_COUNTS = 1500
DEFAULT_MAX_GENES = 4000
DEFAULT_MIN_CELLS_PER_GENE = 5

N_TOP_HVGS = 2500
N_PCS = 30
N_NEIGHBORS = 12
LEIDEN_RES = 0.40

# add this if your code does not already expose it
UMAP_MIN_DIST = 0.25

# Predefined marker gene sets.  Feel free to edit this dictionary to
# reflect your domain knowledge or extend it with additional cell
# types.  Keys are human‑readable cell type names and values are
# lists of gene names (case insensitive) that define the cell type.
MARKER_SETS: Dict[str, List[str]] = {
    "Secondary cells": ["tiptop", "castor"],
    "Leptophragmata": ["tiptop", "dachshund"],
    "Principal cells": ["cut"],
    "Enteroendocrine cells": ["Asti", "DH37", "NPF"],
    "Muscle cells": ["Mef2"],
    "Trachea": ["breathless"],
    # For perinephric membrane, enter the appropriate marker genes here.
    "Perinephric membrane": [],
    # Enter your marker genes for these categories if known.
    "Enterocytes": [],
    "Hindgut cells": [],
    "Ureter": ["hector"],
}


# -----------------------------------------------------------------------------
# Helper functions (copied largely verbatim from the original script)
# -----------------------------------------------------------------------------

def save_current_figure(outbase: Path, dpi: int = 300) -> None:
    """Save the currently active matplotlib figure as PNG and PDF.

    Parameters
    ----------
    outbase : Path
        The base filename (without extension) for the output files.  The
        function will append `.png` and `.pdf`.
    dpi : int, optional
        Dots per inch for the saved images.  A high dpi ensures
        publication‑quality figures.
    """
    outbase.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outbase.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.savefig(outbase.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close()


def add_qc_metrics(adata: sc.AnnData) -> Tuple[int, int]:
    """Compute QC metrics and annotate ribosomal and proxy mitochondrial genes.

    This function adds boolean columns "ribo" and "mt" to ``adata.var`` to
    mark ribosomal genes (based on common prefixes RPL/RPS/RpL/RpS) and
    proxy mitochondrial genes (membership in ``STRICT_MT_TC``).  It then
    calculates QC metrics (total counts, number of genes, pct_counts_ribo,
    pct_counts_mt, pct_counts_in_top_20_genes, etc.) and stores them in
    ``adata.obs``.  The number of ribosomal and mitochondrial proxy genes
    detected are returned.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object to annotate.

    Returns
    -------
    Tuple[int, int]
        A tuple ``(n_ribo_genes, n_mt_proxy_genes)`` giving the number of
        genes flagged as ribosomal and as mitochondrial proxy genes.
    """
    # Mark ribosomal genes based on prefixes (vertebrate and insect conventions)
    vn = adata.var_names.to_series().astype(str)
    ribo_mask = vn.str.startswith(("RPL", "RPS", "RpL", "RpS"))
    adata.var["ribo"] = ribo_mask.to_numpy(dtype=bool)

    # Mark proxy mitochondrial genes using curated TC list
    mt_mask = vn.isin(STRICT_MT_TC)
    adata.var["mt"] = mt_mask.to_numpy(dtype=bool)

    # Compute QC metrics using ribo and mt as qc_vars (present if any true)
    qc_vars = []
    if ribo_mask.any():
        qc_vars.append("ribo")
    if mt_mask.any():
        qc_vars.append("mt")
    # Always include qc_vars list even if empty; scanpy will compute default metrics
    sc.pp.calculate_qc_metrics(adata, qc_vars=qc_vars, inplace=True)
    return int(ribo_mask.sum()), int(mt_mask.sum())


def qc_plots(adata: sc.AnnData, outdir: Path, prefix: str, title: str) -> None:
    """Generate QC violin plots and scatter plots.

    Produces three plots: a violin plot of total_counts,
    n_genes_by_counts, pct_counts_in_top_20_genes and pct_counts_ribo;
    a scatter of total_counts vs n_genes_by_counts; and a scatter of
    total_counts vs pct_counts_ribo (if ribo metrics are present).

    Parameters
    ----------
    adata : sc.AnnData
        The data to plot.
    outdir : Path
        Directory where plots will be saved.
    prefix : str
        Prefix for output filenames.
    title : str
        Title to place above the violin plot.
    """
    sc.set_figure_params(dpi=130, fontsize=10, frameon=False)
    keys = [
        "total_counts",
        "n_genes_by_counts",
        # include mitochondrial proxy percentage if present
        "pct_counts_mt",
        "pct_counts_ribo",
        "pct_counts_in_top_20_genes",
    ]
    keys = [k for k in keys if k in adata.obs.columns]
    # Violin panel
    sc.pl.violin(adata, keys, jitter=0.35, multi_panel=True, show=False)
    plt.suptitle(title, y=1.02)
    save_current_figure(outdir / f"{prefix}_QC_violin")
    # Scatter: total counts vs genes
    sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts", show=False)
    plt.title(f"{title} — total_counts vs n_genes")
    save_current_figure(outdir / f"{prefix}_QC_counts_vs_genes")
    # Scatter: total counts vs ribo and mt percentage (if available)
    if "pct_counts_ribo" in adata.obs.columns:
        sc.pl.scatter(adata, x="total_counts", y="pct_counts_ribo", show=False)
        plt.title(f"{title} — total_counts vs pct_counts_ribo")
        save_current_figure(outdir / f"{prefix}_QC_counts_vs_ribo")
    if "pct_counts_mt" in adata.obs.columns:
        sc.pl.scatter(adata, x="total_counts", y="pct_counts_mt", show=False)
        plt.title(f"{title} — total_counts vs pct_counts_mt (proxy)")
        save_current_figure(outdir / f"{prefix}_QC_counts_vs_mt")


def run_umap_clustering(
    adata: sc.AnnData,
    n_top_hvgs: int = N_TOP_HVGS,
    n_pcs: int = N_PCS,
    n_neighbors: int = N_NEIGHBORS,
    leiden_res: float = LEIDEN_RES,
) -> Tuple[sc.AnnData, str]:
    """Perform normalisation, HVG selection, PCA, neighbours, UMAP and Leiden clustering.

    The input AnnData is expected to contain unnormalised counts.  A
    copy of the data is made internally when slicing highly variable
    genes.

    Returns
    -------
    adata : sc.AnnData
        The processed AnnData, with dimensionality reduction and
        clustering results stored in .obsm and .obs.
    str
        A string indicating which Leiden implementation was used
        ("igraph" or "default").
    """
    # Save raw counts for scVI
    adata.layers["counts"] = adata.X.copy()

    # Normalise/log only for plotting and marker display
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata

    # Highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_hvgs, flavor="seurat")
    adata = adata[:, adata.var["highly_variable"]].copy()

    # scVI embedding
    scvi.model.SCVI.setup_anndata(adata, layer="counts")
    model = scvi.model.SCVI(adata, n_latent=n_pcs)
    model.train()
    adata.obsm["X_scVI"] = model.get_latent_representation()

    # Neighbours and UMAP using scVI latent space
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_scVI")
    sc.tl.umap(adata)
    
    # Leiden clustering (prefer igraph)
    try:
        sc.tl.leiden(
            adata,
            resolution=leiden_res,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
        backend = "igraph"
    except Exception:
        sc.tl.leiden(adata, resolution=leiden_res)
        backend = "default"
    return adata, backend


def save_umap_overview(
    adata: sc.AnnData, outdir: Path, prefix: str, leiden_backend: str
) -> None:
    """Save a multi‑panel UMAP overview showing clusters and QC metrics.

    The function plots UMAP coordinates coloured by cluster and by
    selected QC metrics (total_counts, n_genes_by_counts,
    pct_counts_in_top_20_genes, pct_counts_ribo if available).  All
    panels are arranged in a single figure with a shared legend.
    """
    sc.set_figure_params(dpi=140, fontsize=10, frameon=False)
    colour_list = [
        c
        for c in [
            "leiden",
            "total_counts",
            "n_genes_by_counts",
            "pct_counts_in_top_20_genes",
            # include mitochondrial proxy percentage if available
            "pct_counts_mt",
            "pct_counts_ribo",
        ]
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
    plt.suptitle(
        f"UMAP ({prefix}) — leiden={LEIDEN_RES} ({leiden_backend})", y=1.02
    )
    save_current_figure(outdir / f"{prefix}_UMAP_overview")


def export_markers(
    adata: sc.AnnData, outdir: Path, top_n: int = 20
) -> pd.DataFrame:
    """Compute marker genes for each cluster and export to CSV/Excel.

    Uses the Wilcoxon rank‑sum test to compare each cluster against
    all others.  Saves two tables:

    1. A full table of all ranked genes per cluster (`markers_all_clusters.csv`).
    2. A table with the top N up‑regulated genes per cluster
       (`markers_topN_per_cluster.csv`).  Adjusted p‑values and
       log‑fold changes are used to filter for positive markers.

    Returns
    -------
    pd.DataFrame
        The DataFrame containing the top N markers per cluster.
    """
    sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")
    # Build a concatenated DataFrame of all markers
    dfs = []
    for cl in adata.obs["leiden"].cat.categories:
        df = sc.get.rank_genes_groups_df(adata, group=cl)
        df.insert(0, "cluster", cl)
        dfs.append(df)
    all_markers = pd.concat(dfs, ignore_index=True)
    all_markers_path = outdir / "markers_all_clusters.csv"
    all_markers.to_csv(all_markers_path, index=False)
    # Top N positive markers per cluster
    top_list = []
    for cl in adata.obs["leiden"].cat.categories:
        df = all_markers[all_markers["cluster"] == cl].copy()
        # Keep only positive markers if adjusted p and logFC exist
        if "pvals_adj" in df.columns:
            df = df[df["pvals_adj"] < 0.05]
        if "logfoldchanges" in df.columns:
            df = df[df["logfoldchanges"] > 0]
        top_list.append(df.head(top_n))
    top_markers = pd.concat(top_list, ignore_index=True)
    top_markers_path = outdir / f"markers_top{top_n}_per_cluster.csv"
    top_markers.to_csv(top_markers_path, index=False)
    # Also write Excel files if possible
    try:
        all_markers.to_excel(outdir / "markers_all_clusters.xlsx", index=False)
        top_markers.to_excel(
            outdir / f"markers_top{top_n}_per_cluster.xlsx", index=False
        )
    except Exception:
        pass  # Excel writing is optional
    return top_markers


def compute_marker_set_scores(
    adata: sc.AnnData, marker_sets: Dict[str, List[str]]
) -> pd.DataFrame:
    """Compute average expression per marker set and per cluster.

    For each cluster and each entry in the marker_sets dictionary,
    calculate the average expression (mean over cells and over
    marker genes) using the raw log‑normalised data.  Returns a
    tidy DataFrame with columns ['cluster', 'cell_type', 'n_genes',
    'mean_expression'].

    Parameters
    ----------
    adata : sc.AnnData
        Processed AnnData with clusters and raw log‑normalised data in
        adata.raw.
    marker_sets : dict
        Mapping from human‑readable cell type names to lists of gene
        identifiers.  Gene names are matched case insensitively.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per (cluster, cell_type) pair
        describing the mean expression of the marker genes.
    """
    if adata.raw is None:
        raise ValueError(
            "adata.raw must contain log‑normalised data to compute marker set scores"
        )
    # Lowercase version of gene names for case‑insensitive matching
    gene_index = {g.lower(): idx for idx, g in enumerate(adata.raw.var_names)}
    data = []
    for cl in adata.obs["leiden"].cat.categories:
        idx_cells = adata.obs["leiden"] == cl
        sub = adata.raw[idx_cells, :]
        for cell_type, genes in marker_sets.items():
            # Filter marker list to genes present in the dataset
            present_genes = [g for g in genes if g.lower() in gene_index]
            if not present_genes:
                mean_expr = np.nan
                n_genes = 0
            else:
                gene_indices = [gene_index[g.lower()] for g in present_genes]
                # Compute mean expression across cells and genes
                expr_matrix = sub.X[:, gene_indices]
                # expr_matrix may be sparse; convert to dense for mean
                if not isinstance(expr_matrix, np.ndarray):
                    expr_matrix = expr_matrix.toarray()
                mean_expr = float(expr_matrix.mean())
                n_genes = len(present_genes)
            data.append(
                {
                    "cluster": cl,
                    "cell_type": cell_type,
                    "n_genes": n_genes,
                    "mean_expression": mean_expr,
                }
            )
    return pd.DataFrame(data)


def plot_gene_panel(
    adata: sc.AnnData, genes: List[str], outdir: Path, prefix: str
) -> None:
    """Plot a combined UMAP and violin panel for a list of genes.

    Creates a grid of UMAP plots (one per gene) and a grid of violin
    plots (one per gene grouped by clusters).  If a gene is not found
    in the dataset, it is skipped silently.  Plots are saved to
    PNG and PDF.
    """
    # Clean the list to include only genes present in adata.raw
    if adata.raw is not None:
        valid_genes = [g for g in genes if g in adata.raw.var_names]
    else:
        valid_genes = [g for g in genes if g in adata.var_names]
    if not valid_genes:
        print(
            "No valid genes provided for plotting.  Skipping gene panel."
        )
        return
    # Determine grid dimensions
    n_genes = len(valid_genes)
    ncols = 4
    nrows = int(np.ceil(n_genes / ncols))
    # UMAP gene expression panel
    sc.set_figure_params(dpi=140, fontsize=9, frameon=False)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 3.0 * nrows),
        constrained_layout=True,
    )
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
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(
        f"Gene expression UMAP: {', '.join(valid_genes)}", y=1.02
    )
    save_current_figure(outdir / f"{prefix}_UMAP_panel")
    # Violin plots per gene
    sc.set_figure_params(dpi=130, fontsize=9, frameon=False)
    sc.pl.violin(
        adata,
        keys=valid_genes,
        groupby="leiden",
        jitter=0.25,
        multi_panel=True,
        show=False,
        use_raw=True,
    )
    plt.suptitle(
        f"Gene expression violin: {', '.join(valid_genes)}", y=1.02
    )
    save_current_figure(outdir / f"{prefix}_Violin_panel")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a comprehensive scRNA‑seq pipeline with TC identifiers as var_names, "
            "including QC, UMAP, clustering, marker analysis and gene panel plots. "
            "Writes both h5ad and loom outputs."
        )
    )
    parser.add_argument(
        "--mtx_dir",
        type=str,
        default=MTX_DIR,
        help="Path to the 10x filtered_feature_bc_matrix directory",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=OUT_DIR,
        help="Directory where outputs will be written",
    )
    parser.add_argument(
        "--min_genes",
        type=int,
        default=DEFAULT_MIN_GENES,
        help="Minimum genes per cell",
    )
    parser.add_argument(
        "--min_counts",
        type=int,
        default=DEFAULT_MIN_COUNTS,
        help="Minimum counts per cell",
    )
    parser.add_argument(
        "--max_genes",
        type=int,
        default=DEFAULT_MAX_GENES,
        help="Maximum genes per cell",
    )
    parser.add_argument(
        "--min_cells_per_gene",
        type=int,
        default=DEFAULT_MIN_CELLS_PER_GENE,
        help="Minimum number of cells expressing a gene to retain that gene",
    )
    parser.add_argument(
        "--top_markers",
        type=int,
        default=20,
        help="Number of top marker genes to export per cluster",
    )
    parser.add_argument(
        "--no_marker_analysis",
        action="store_true",
        help="Skip marker gene analysis and gene panel plotting",
    )
    args = parser.parse_args()

    mtx_path = Path(args.mtx_dir)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load dataset with gene_ids as var_names (TCxxx IDs)
    print(f"[INFO] Loading data from {mtx_path} with gene_ids as var_names…")
    adata = sc.read_10x_mtx(mtx_path, var_names="gene_ids", cache=True)
    adata.var_names_make_unique()
    print(f"[INFO] Loaded {adata.n_obs} cells × {adata.n_vars} genes")

    # Compute QC metrics before filtering
    rb_n, mt_n = add_qc_metrics(adata)
    print(f"[INFO] Detected {rb_n} ribosomal genes")
    if mt_n > 0:
        print(f"[INFO] Detected {mt_n} proxy mitochondrial genes (strict TC list)")
    else:
        print("[INFO] No proxy mitochondrial genes detected in this dataset")
    qc_plots(adata, outdir, "BEFORE", "QC BEFORE filtering")

    # Filter cells and genes
    print("[INFO] Filtering cells and genes…")
    # Cell filters
    cell_mask = (
        (adata.obs["n_genes_by_counts"] >= args.min_genes)
        & (adata.obs["total_counts"] >= args.min_counts)
        & (adata.obs["n_genes_by_counts"] <= args.max_genes)
    )
    adata = adata[cell_mask].copy()
    # Gene filter
    sc.pp.filter_genes(adata, min_cells=args.min_cells_per_gene)
    print(f"[INFO] After filtering: {adata.n_obs} cells × {adata.n_vars} genes")
    # Recompute QC metrics after filtering for updated plots
    rb_n_after, mt_n_after = add_qc_metrics(adata)
    qc_plots(adata, outdir, "AFTER", "QC AFTER filtering")

    # Dimensionality reduction and clustering
    print("[INFO] Running dimensionality reduction, UMAP and clustering…")
    adata_proc, leiden_backend = run_umap_clustering(adata)
    # Save UMAP overview
    save_umap_overview(adata_proc, outdir, "AFTER", leiden_backend)

    # Marker gene analysis
    if not args.no_marker_analysis:
        print("[INFO] Performing marker gene analysis…")
        top_markers = export_markers(adata_proc, outdir, top_n=args.top_markers)
        print(
            f"[INFO] Exported top {args.top_markers} markers per cluster"
        )
        # Compute marker set scores
        scores_df = compute_marker_set_scores(adata_proc, MARKER_SETS)
        scores_df.to_csv(outdir / "marker_set_scores_by_cluster.csv", index=False)
        # Determine cluster annotations by selecting the marker set with the highest mean expression per cluster
        annotations = []
        for cl in scores_df["cluster"].unique():
            sub = scores_df[scores_df["cluster"] == cl]
            # If all NaN mean expressions, assign unknown
            if sub["mean_expression"].notna().any():
                best = sub.iloc[sub["mean_expression"].idxmax()]
                cell_type = best["cell_type"]
            else:
                cell_type = "Unknown"
            annotations.append({"cluster": cl, "cell_type": cell_type})
        annot_df = pd.DataFrame(annotations)
        annot_df.to_csv(outdir / "cluster_annotations.csv", index=False)
        # Prompt user for gene panel
        try:
            genes_input = input(
                "Enter gene names to plot (comma separated), or press Enter to skip: "
            ).strip()
        except EOFError:
            genes_input = ""
        if genes_input:
            gene_list = [g.strip() for g in genes_input.split(",") if g.strip()]
            print(
                f"[INFO] Plotting gene panel for: {', '.join(gene_list)}"
            )
            plot_gene_panel(adata_proc, gene_list, outdir, "SelectedGenes")
        else:
            print("[INFO] No genes provided.  Skipping gene panel plots.")
    else:
        print(
            "[INFO] Skipping marker analysis and gene panel as requested."
        )

    # Save processed AnnData object
    adata_proc.write(outdir / "final_filtered_umap_leiden.h5ad")
    print(
        f"[INFO] Analysis complete.  AnnData saved to {outdir / 'final_filtered_umap_leiden.h5ad'}"
    )

    # --- Write loom file with TC IDs ---
    # write_obsm_varm=True keeps UMAP coordinates and varm matrices in the loom
    loom_path = outdir / "final_filtered_umap_leiden_TC.loom"
    print(f"[INFO] Writing loom file to {loom_path} …")
    # Ensure X is float32 to reduce loom size
    try:
        adata_proc.X = adata_proc.X.astype(np.float32)
    except Exception:
        pass  # If conversion fails (dense matrix), ignore
    adata_proc.write_loom(str(loom_path), write_obsm_varm=True)
    print(
        f"[INFO] Loom with TC identifiers written to {loom_path}" 
    )


if __name__ == "__main__":
    main()