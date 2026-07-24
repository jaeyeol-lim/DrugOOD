import argparse
import json
import random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_path", type=str, required=True, help="Path to your json file")
    ap.add_argument("--split", type=str, default="train", choices=["train", "ood_val", "ood_test"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="samples_10.png")
    args = ap.parse_args()

    # RDKit import
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Scaffolds import MurckoScaffold

    with open(args.json_path, "r") as f:
        data = json.load(f)

    rows = data["split"][args.split]
    random.seed(args.seed)
    random.shuffle(rows)

    mols = []
    legends = []
    kept = 0
    for r in rows:
        smi = r.get("smiles", None)
        if not smi:
            continue

        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue

        scaf = MurckoScaffold.GetScaffoldForMol(m)

        cls_label = r.get("cls_label", "NA")
        reg_label = r.get("reg_label", "NA")
        assay_id = r.get("assay_id", "NA")
        domain_id = r.get("domain_id", "NA")

        mols.extend([m, scaf])
        legends.extend([
            f"{kept}: y={cls_label}, reg={reg_label:.3f} | assay={assay_id} dom={domain_id}",
            f"{kept}: scaffold"
        ])

        kept += 1
        if kept >= args.n:
            break

    if kept == 0:
        raise RuntimeError("No valid molecules parsed. Check your json_path and smiles validity.")

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=2,          # (molecule | scaffold)
        subImgSize=(340, 340),
        legends=legends,
        useSVG=False
    )
    img.save(args.out)
    print(f"[OK] saved to {args.out} (kept={kept})")

if __name__ == "__main__":
    main()