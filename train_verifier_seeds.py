"""Multi-seed training of the crop-verifier variants, selecting on validation only.

Three roles, no exceptions:
  train  the balanced pool the model fits
  val    the only data used to choose the checkpoint (lowest validation loss) and the
         operating threshold (highest validation F1)
  test   reported once, never used to choose anything

VeryLowLight rows are assigned to a role from the new block split in vll_split.py, not
from the held_out column the datasets were built with. The held_out column records the
old two-way split, which was used for selection in earlier rounds, so it cannot be used
for the clean protocol. HaGRID rows use the dataset's own tr/valid/test division, with
hagrid_valid as validation and hagrid_test as test.

    .venv-blazepalm\\Scripts\\python.exe train_verifier_seeds.py --seeds 0,1,2,3,4,5,6,7,8,9

Outputs, all under results/:
  verifier_seeds/<variant>_seed<seed>.pt        selected checkpoint
  verifier_seeds/<variant>_seed<seed>.json      per-run record
  verifier_seeds_curves_<variant>.csv           per-epoch train/val loss
  verifier_seeds_summary.csv / .txt             mean and SD across seeds
"""
import argparse
import csv
import json
import os
import re
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from crop_verifier import SIZE, SmallCNN          # noqa: E402
from vll_split import SPLIT_SEED, role_of_frame   # noqa: E402

OUTDIR = os.path.join(HERE, "results", "verifier_seeds")

VARIANTS = [
    ("verifier_cnn.pt", "verifier_dataset"),
    ("verifier_cnn_big.pt", "big_dataset"),
    ("verifier_cnn_vll.pt", "vll_dataset"),
    ("verifier_dark.pt", "dark_dataset"),
]

VLL_SOURCES = ("vll_real_mp", "vll_real_det", "vll_real_clutter")


def strip_variant_tokens(source):
    """'hagrid512_aug_dark' -> 'hagrid512';  'vll_real_mp_aug_dark' -> 'vll_real_mp';
    'hagrid_train_vll_aug' -> 'hagrid_train'.

    The last one matters: the target-condition dataset carries 19,956 rows of
    HaGRID training images put through the VeryLowLight augmentation recipe, and
    without the token they map to no role and are silently dropped, which would
    retrain that variant on a third of its data.
    """
    s = source
    changed = True
    while changed:
        changed = False
        for tok in ("_vll_aug", "_aug_dark", "_aug", "_dark"):
            if s.endswith(tok):
                s = s[: -len(tok)]
                changed = True
    return s


def vll_frame_of(crop, source):
    """Frame index of a VeryLowLight-derived crop, read from its filename.

    build_vll_dataset.py names crops 'vll_mp_0023.png' for the MediaPipe-anchored
    positives and 'vll_<source>_0005_0374.png' for the detector-recipe crops, and the
    augmented and dark copies embed the original name, so the frame can be recovered.
    """
    base = os.path.basename(crop)
    m = re.search(r"vll_mp_(\d{4})", base)
    if m:
        return int(m.group(1))
    m = re.search(r"real_(?:mp|det|clutter)_(\d{4})_", base)
    if m:
        return int(m.group(1))
    return None


def role_of_row(row):
    src = strip_variant_tokens(row["source"])
    if src in VLL_SOURCES:
        n = vll_frame_of(row["crop"], row["source"])
        if n is None:
            return None
        return role_of_frame(n)
    if src == "hagrid_train":
        return "train"
    if src == "own_clip":
        return "train"
    if src == "hagrid512":
        return "train"
    if src == "hagrid_valid":
        return "val"
    if src == "own_nohand":
        return "val"
    if src == "hagrid_test":
        return "test"
    return None


def load_threshold_set(dataset="dark_dataset"):
    """As-recorded VeryLowLight validation crops, one shared set for every variant.

    The 11k dataset predates VeryLowLight and contains none of its rows, so picking each
    model's operating threshold on its own validation pool would compare a HaGRID-tuned
    threshold against a VeryLowLight-tuned one. Every model is therefore thresholded on
    the same crops: the un-augmented validation-block crops of the target clip.

    Un-augmented only, so the set is real frames rather than photometric copies of a few
    of them.
    """
    path = os.path.join(HERE, dataset, "manifest.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    X, y, n_used = [], [], 0
    for r in rows:
        if r["source"] not in VLL_SOURCES:
            continue
        if role_of_row(r) != "val":
            continue
        img = cv2.imread(os.path.join(HERE, dataset, r["crop"]))
        if img is None:
            continue
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        X.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        y.append(1 if r["label"] == "hand" else 0)
        n_used += 1
    if not X:
        return None, None
    return np.stack(X).astype(np.uint8), np.array(y, dtype=np.float32)


def load_pool(dataset):
    """Return (X uint8 NCHW-ready, y, roles) for a dataset."""
    path = os.path.join(HERE, dataset, "manifest.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    X, y, roles, meta = [], [], [], []
    skipped = {}
    for r in rows:
        role = role_of_row(r)
        if role is None:
            skipped[r["source"]] = skipped.get(r["source"], 0) + 1
            continue
        img = cv2.imread(os.path.join(HERE, dataset, r["crop"]))
        if img is None:
            skipped["missing_file"] = skipped.get("missing_file", 0) + 1
            continue
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        X.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        y.append(1 if r["label"] == "hand" else 0)
        roles.append(role)
        meta.append(r)
    X = np.stack(X).astype(np.uint8)
    return X, np.array(y, dtype=np.float32), np.array(roles), meta, skipped


def auc(scores, labels):
    """Rank form of ROC-AUC, nan when one class is absent."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def predict(model, X, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).permute(0, 3, 1, 2).float() / 255.0
            out.append(torch.sigmoid(model(xb)).numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def loss_on(model, X, y, lossf, bs=256):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).permute(0, 3, 1, 2).float() / 255.0
            yb = torch.from_numpy(y[i:i + bs])
            tot += float(lossf(model(xb), yb)) * len(xb)
            n += len(xb)
    return tot / max(n, 1)


def metrics_at(scores, labels, thr):
    pred = scores >= thr
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if (prec == prec and rec == rec and prec + rec) else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1)


def pick_threshold(scores, labels, grid=np.arange(0.05, 0.96, 0.01)):
    best, best_f1 = 0.5, -1.0
    for t in grid:
        m = metrics_at(scores, labels, t)
        if m["f1"] == m["f1"] and m["f1"] > best_f1:
            best, best_f1 = float(t), m["f1"]
    return best, best_f1


def train_one(Xtr, ytr, Xva, yva, epochs, seed, lr, bs, log):
    torch.manual_seed(seed)
    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    n_params = sum(p.numel() for p in model.parameters())
    perm_rng = np.random.default_rng(seed)
    curve = []
    best_loss, best_state, best_epoch = float("inf"), None, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = perm_rng.permutation(len(Xtr))
        tot, seen = 0.0, 0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb = torch.from_numpy(Xtr[idx]).permute(0, 3, 1, 2).float() / 255.0
            yb = torch.from_numpy(ytr[idx])
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
            seen += len(idx)
        tr_loss = tot / max(seen, 1)
        va_loss = loss_on(model, Xva, yva, lossf)
        curve.append(dict(epoch=ep, train_loss=round(tr_loss, 6),
                          val_loss=round(va_loss, 6)))
        if va_loss < best_loss:
            best_loss, best_epoch = va_loss, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    train_seconds = time.time() - t0
    model.load_state_dict(best_state)
    log(f"      best epoch {best_epoch}/{epochs}  val loss {best_loss:.4f}  "
        f"({train_seconds:.0f}s, {n_params} params)")
    return model, curve, best_epoch, best_loss, train_seconds, n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(str(s) for s in range(10)))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--variants", default="all")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    os.makedirs(OUTDIR, exist_ok=True)

    variants = VARIANTS
    if args.variants != "all":
        want = set(args.variants.split(","))
        variants = [v for v in VARIANTS if v[0] in want]

    THRX, THRY = load_threshold_set("dark_dataset")
    if THRX is None:
        sys.exit("could not build the VeryLowLight validation threshold set")
    print(f"threshold selection set: {len(THRY)} as-recorded VLL validation crops "
          f"({int((THRY == 1).sum())} hand / {int((THRY == 0).sum())} clutter)")

    all_rows = []
    for out_name, dataset in variants:
        print(f"\n=== {out_name}  ({dataset}) ===", flush=True)
        X, y, roles, meta, skipped = load_pool(dataset)
        print(f"  rows kept: {len(y)}   skipped: {skipped}")
        counts = {r: int((roles == r).sum()) for r in ("train", "val", "test")}
        print(f"  pool sizes: {counts}")
        if counts["train"] == 0 or counts["val"] == 0:
            print("  no training or validation rows, skipping")
            continue

        itr = np.where(roles == "train")[0]
        iva = np.where(roles == "val")[0]
        ite = np.where(roles == "test")[0]
        src_norm = np.array([strip_variant_tokens(m["source"]) for m in meta])
        is_hag_test = src_norm == "hagrid_test"
        is_hag_val = src_norm == "hagrid_valid"
        is_vll = np.isin(src_norm, VLL_SOURCES)
        vll_val = iva[is_vll[iva]]
        vll_test = ite[is_vll[ite]]
        print(f"  VLL rows: {int(is_vll.sum())}  (val {len(vll_val)}, test {len(vll_test)})")

        Xva, yva = X[iva], y[iva]
        # balanced training pool: subsample the majority class only
        pos, neg = itr[y[itr] == 1], itr[y[itr] == 0]
        n_bal = min(len(pos), len(neg))
        print(f"  train: {len(pos)} hand / {len(neg)} clutter -> {2*n_bal} balanced")

        curves_accum = {}
        for seed in seeds:
            rng = np.random.default_rng(seed)          # same seed list for every variant
            keep = np.concatenate([rng.choice(pos, n_bal, replace=False),
                                   rng.choice(neg, n_bal, replace=False)])
            rng.shuffle(keep)
            print(f"    seed {seed}", flush=True)
            model, curve, best_ep, best_loss, secs, n_params = train_one(
                X[keep], y[keep], Xva, yva, args.epochs, seed, args.lr, args.batch,
                print)
            for c in curve:
                curves_accum.setdefault(c["epoch"], []).append(
                    (c["train_loss"], c["val_loss"]))

            save_path = os.path.join(OUTDIR, f"{out_name}_seed{seed}.pt")
            torch.save(model.state_dict(), save_path)

            p_va = predict(model, Xva)
            p_te = predict(model, X[ite]) if len(ite) else np.zeros(0, np.float32)
            p_thr = predict(model, THRX) if THRX is not None else p_va
            y_thr = THRY if THRX is not None else yva
            thr, thr_f1 = pick_threshold(p_thr, y_thr)
            m_va = metrics_at(p_va, yva, thr)
            m_thr = metrics_at(p_thr, y_thr, thr)
            m_te = metrics_at(p_te, y[ite], thr) if len(ite) else {}
            p_hagte = predict(model, X[is_hag_test]) if is_hag_test.any() else np.zeros(0, np.float32)
            p_hagva = predict(model, X[is_hag_val]) if is_hag_val.any() else np.zeros(0, np.float32)
            p_vllva = predict(model, X[vll_val]) if len(vll_val) else np.zeros(0, np.float32)
            p_vllte = predict(model, X[vll_test]) if len(vll_test) else np.zeros(0, np.float32)
            m_hagte = metrics_at(p_hagte, y[is_hag_test], thr) if is_hag_test.any() else {}
            row = dict(
                variant=out_name, dataset=dataset, seed=seed, params=n_params,
                train_seconds=round(secs, 2), best_epoch=best_ep,
                best_val_loss=round(best_loss, 6), threshold=round(thr, 4),
                val_auc=round(auc(p_va, yva), 6),
                hag_valid_auc=round(auc(p_hagva, y[is_hag_val]), 6) if is_hag_val.any() else float("nan"),
                hag_test_auc=round(auc(p_hagte, y[is_hag_test]), 6) if is_hag_test.any() else float("nan"),
                vll_val_auc=round(auc(p_vllva, y[vll_val]), 6) if len(vll_val) else float("nan"),
                vll_test_auc=round(auc(p_vllte, y[vll_test]), 6) if len(vll_test) else float("nan"),
                thr_set_f1=round(m_thr["f1"], 6),
                thr_set_precision=round(m_thr["precision"], 6),
                thr_set_recall=round(m_thr["recall"], 6),
                val_precision=round(m_va["precision"], 6),
                val_recall=round(m_va["recall"], 6), val_f1=round(m_va["f1"], 6),
                test_precision=round(m_te.get("precision", float("nan")), 6),
                test_recall=round(m_te.get("recall", float("nan")), 6),
                test_f1=round(m_te.get("f1", float("nan")), 6),
                test_tp=m_te.get("tp", ""), test_fp=m_te.get("fp", ""),
                test_fn=m_te.get("fn", ""), test_tn=m_te.get("tn", ""),
                hag_test_precision=round(m_hagte.get("precision", float("nan")), 6),
                hag_test_recall=round(m_hagte.get("recall", float("nan")), 6),
            )
            all_rows.append(row)
            with open(os.path.join(OUTDIR, f"{out_name}_seed{seed}.json"), "w",
                      encoding="utf-8") as f:
                json.dump(row, f, indent=2)
            print(f"      thr {thr:.2f}  (VLL val F1 {m_thr['f1']:.3f})  "
                  f"VLL test F1 {m_te.get('f1', float('nan')):.3f}  "
                  f"hagrid-test AUC {row['hag_test_auc']}", flush=True)

        with open(os.path.join(HERE, "results",
                               f"verifier_seeds_curves_{out_name}.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss_mean", "train_loss_sd",
                        "val_loss_mean", "val_loss_sd", "n_seeds"])
            for ep in sorted(curves_accum):
                a = np.array(curves_accum[ep], dtype=float)
                w.writerow([ep, round(a[:, 0].mean(), 6), round(a[:, 0].std(ddof=1), 6),
                            round(a[:, 1].mean(), 6), round(a[:, 1].std(ddof=1), 6),
                            len(a)])

    if not all_rows:
        sys.exit("nothing trained")
    fields = list(all_rows[0].keys())
    with open(os.path.join(HERE, "results", "verifier_seeds_summary.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    # ---- mean and SD across seeds, per variant ----
    lines = [f"VERIFIER MULTI-SEED RESULTS  (seeds {seeds}, {len(seeds)} runs per variant)",
             f"VeryLowLight split seed {SPLIT_SEED}. Checkpoint and threshold chosen on "
             f"validation only.", ""]
    num = ["hag_test_auc", "val_auc", "test_f1", "test_precision", "test_recall",
           "thr_set_f1", "threshold", "best_epoch", "train_seconds"]
    for out_name, _d in variants:
        rows = [r for r in all_rows if r["variant"] == out_name]
        if not rows:
            continue
        lines.append(f"{out_name}  (n={len(rows)} seeds)")
        for k in num:
            v = np.array([r[k] for r in rows if r[k] == r[k]], dtype=float)
            if len(v):
                lines.append(f"   {k:<18} mean {v.mean():>10.4f}   SD {v.std(ddof=1):>8.4f}"
                             f"   min {v.min():>8.4f}  max {v.max():>8.4f}")
        lines.append("")
    txt = "\n".join(lines)
    with open(os.path.join(HERE, "results", "verifier_seeds_summary.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt)


if __name__ == "__main__":
    main()
