# 🪲 Beetle scRNA-seq Tissue Annotation Pipeline

A reproducible **single-cell RNA-seq analysis workflow** for **beetle tissue annotation**, built around a biologically informed **Drosophila mitochondrial proxy filtering strategy**.

This pipeline processes Cell Ranger `filtered_feature_bc_matrix` outputs using **Scanpy**, with an optional **Streamlit interface** used only for parameter input and visualization. All computational analysis runs in the backend Python pipeline to ensure **reproducibility**, **stability**, and **independence from UI state**. ⚙️📊

---

# ✨ Key idea

Many beetle genomes lack reliable **canonical mitochondrial gene annotations**, which makes traditional mitochondrial QC filtering difficult.

To overcome this limitation, this workflow uses a **mitochondrial proxy strategy**:

1. Use **Drosophila mitochondrial gene sets** as reference  
2. Identify beetle homologs using **BLAST mapping**  
3. Map homologs to **TCxxxx genes in the beetle genome**  
4. Use these genes to estimate **mitochondrial proxy expression**

This enables biologically informed **low-quality cell detection and filtering** even when mitochondrial annotations are incomplete.

🧬 **Cross-species mapping:**  
`Drosophila mt genes → BLAST → Beetle homologs → QC filtering`

---

# 🔬 Pipeline overview

The pipeline follows a **standard and transparent single-cell workflow**:

1. Load **10x Genomics matrices**
2. Compute **quality control metrics**
3. Filter cells and genes
4. Normalize and log-transform data
5. Select **highly variable genes (HVGs)**
6. Perform **PCA dimensionality reduction**
7. Construct **neighbor graph**
8. Perform **Leiden clustering**
9. Generate **UMAP visualization**
10. Identify **cluster marker genes**

Outputs are exported as:

- `.h5ad` analysis objects  
- QC plots  
- UMAP embeddings  
- Marker gene tables (Excel)

📊 This ensures **transparent and reproducible downstream analysis**.

---

# ⚙️ Typical workflow

1. Input **Cell Ranger filtered_feature_bc_matrix**
2. Run **QC filtering**
3. Apply **mitochondrial proxy filtering**
4. Compute **HVG selection**
5. Perform **PCA + neighbor graph**
6. Run **Leiden clustering**
7. Visualize clusters using **UMAP**
8. Export **cluster markers and analysis outputs**

---

## 🧪 Core parameters and Output

<table>
<tr>
<td width="50%">

### 🧪 Core parameters

| Parameter | Default |
|-----------|--------|
| MIN_GENES | 400 |
| MIN_COUNTS | 800 |
| MAX_GENES | 3000 |
| MIN_CELLS_PER_GENE | 3 |
| N_TOP_HVGS | 2000 |
| N_PCS | 30 |
| N_NEIGHBORS | 15 |
| LEIDEN_RES | 0.66 |

These parameters allow users to refine clustering resolution and biological signal detection.

</td>

<td width="50%">

### 📊 Example Output

<img src="https://github.com/user-attachments/assets/8b19a508-7d34-4ef0-9c67-39731a49257b" width="100%">

</td>
</tr>
</table>
---

# 🧩 Supported workflows

The pipeline supports both:

• **Single sample datasets**  
• **Multi-sample integration**

Optional batch correction can be performed using:

- **Harmony integration**

This enables analysis of **developmental stages**, **experimental conditions**, or **multi-dataset comparisons**.

---

# 📦 Output files

After analysis, the pipeline produces:

<img width="461" height="286" alt="image" src="https://github.com/user-attachments/assets/8b19a508-7d34-4ef0-9c67-39731a49257b" />


<img width="1074" height="714" alt="image" src="https://github.com/user-attachments/assets/60459f09-e10e-4b2c-9715-565731339394" />


These outputs provide **complete reproducibility and downstream compatibility**.

---

# 🎯 Goal

Turn complex beetle scRNA-seq datasets into a **clean, reproducible cellular atlas** using:

• Cross-species mitochondrial proxy QC  
• Standardized Scanpy workflows  
• Transparent parameter control  
• Fully reproducible outputs

This framework provides a **robust and flexible platform for beetle single-cell tissue annotation and cross-species mitochondrial proxy filtering**. 🧬✨
