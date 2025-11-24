import scanpy as sc
import anndata as ad
import mygene
from Bio import Entrez
import time, json, os
from tqdm import tqdm

Entrez.email = "xijie_guo@brown.edu"
Entrez.api_key = "917fbf331a8c67b5a1011d9ff221489f7108"
BATCH_SIZE = 100

data = sc.read_h5ad("data/DLPFC_151507.h5ad")
print(data.X.shape)
# print(data.var_names)
# print(data.obsm["spatial"])

gene_symbols = data.var_names.tolist()

mg = mygene.MyGeneInfo()

results = mg.querymany(gene_symbols, scopes="symbol", species="human", fields="entrezgene")

symbol_to_ncbi = {}
for res in results:
    symbol = res.get("query")
    gene_id = res.get("entrezgene")
    symbol_to_ncbi[symbol] = gene_id

# print(symbol_to_ncbi)

def fetch_ncbi_summary_in_batch(ncbi_ids):
    ids_str = ",".join(str(i) for i in ncbi_ids if i is not None)

    if ids_str == "":
        return []

    for attempt in range(4):
        try:
            handle = Entrez.efetch(
                db="gene",
                id=ids_str,
                rettype="xml",
                retmode="xml"
            )
            return Entrez.read(handle)
        except Exception as e:
            print(f"[Retry {attempt+1}/4] Batch fetch failed: {e}")
            time.sleep(1 + attempt)  # exponential backoff

    print("[FAILED BATCH] Could not fetch:", ids_str)
    return []

gene_summaries = {}

print("Fetching NCBI summaries")

for i in tqdm(range(0, len(gene_symbols), BATCH_SIZE)):
    batch_symbols = gene_symbols[i : i + BATCH_SIZE]
    batch_ids     = [symbol_to_ncbi[s] for s in batch_symbols]

    records = fetch_ncbi_summary_in_batch(batch_ids)

    for symbol, ncbi_id in zip(batch_symbols, batch_ids):

        if ncbi_id is None:
            gene_summaries[symbol] = ""
            continue

        rec = next(
            (r for r in records
             if str(r["Entrezgene_track-info"]["Gene-track"]["Gene-track_geneid"]) == str(ncbi_id)),
            None
        )

        if rec is None:
            gene_summaries[symbol] = ""
        else:
            summary = rec.get("Entrezgene_summary", "")
            gene_summaries[symbol] = summary


with open("ncbi_gene_summaries.json", "w") as f:
    json.dump(gene_summaries, f, indent=2)

print("DONE. Saved NCBI summaries to gene_summaries.json")