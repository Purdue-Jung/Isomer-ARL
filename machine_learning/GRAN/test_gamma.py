#!/usr/bin/env python3
"""
Minimal evaluation for GRAN‑γ checkpoints.

Per sample i (0000 … n-1) it writes only:
   i_graph.png   – GT & predicted edges overlaid
   i_edges.txt   – ASCII dump of GT and predicted edge‑lists
"""

import os, argparse, numpy as np, torch, matplotlib.pyplot as plt
import networkx as nx
from termcolor import cprint
from tqdm import tqdm

from utils.arg_helper            import get_config
from model.gran_mixture_bernoulli import GRANMixtureBernoulli
from dataset.gamma_dataset        import GammaSchemeDataset

# ────────────────────────────── helpers ──────────────────────────────────
def graph_from_adj(adj):
    r, c = np.where(adj > 0.5)
    G = nx.Graph()
    G.add_edges_from(zip(r.tolist(), c.tolist()))
    return G

def edge_set(G):
    return {tuple(sorted(e)) for e in G.edges()}

def save_graph_png(idx, gt_G, pred_G, out_dir, layout="spring"):
    # choose which graph defines the layout box
    base_G = gt_G if len(gt_G) else pred_G
    if   layout == "spring":
        pos = nx.spring_layout(base_G, seed=0, k=0.3)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(base_G)
    elif layout == "spectral":
        pos = nx.spectral_layout(base_G)
    else:
        pos = nx.circular_layout(base_G)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)

    if len(gt_G):
        nx.draw_networkx_edges(
            gt_G, pos, ax=ax,
            edge_color="#00d7ff", width=2.5,
            connectionstyle="arc3,rad=0.0")

    if len(pred_G):
        nx.draw_networkx_edges(
            pred_G, pos, ax=ax,
            edge_color="crimson", style="dashed", width=1.6,
            connectionstyle="arc3,rad=0.25")

    nx.draw_networkx_nodes(base_G, pos, ax=ax,
                           node_size=24, node_color="black", linewidths=0)

    ax.set_axis_off()
    fig.tight_layout(pad=0.02)
    fig.savefig(os.path.join(out_dir, f"{idx:04d}_graph.png"))
    plt.close(fig)

def save_edges_txt(idx, gt_G, pred_G, out_dir):
    txt_path = os.path.join(out_dir, f"{idx:04d}_edges.txt")
    with open(txt_path, "w") as f:
        f.write("GT:\n")
        f.write(" ".join(map(str, sorted(edge_set(gt_G)))) + "\n\n")
        f.write("PRED:\n")
        f.write(" ".join(map(str, sorted(edge_set(pred_G)))) + "\n")

# adjacency‑overlay helper
def save_adj_overlay(idx, A_gt, A_pred, out_dir):
    """
    Build a 3‑channel (H×W×3) image:
        green = TP, red = FP, blue = FN
    """
    TP = (A_gt & A_pred)
    FP = (~A_gt & A_pred)
    FN = (A_gt & ~A_pred)

    overlay = np.zeros((*A_gt.shape, 3), dtype=float)
    overlay[..., 1] = TP                   # G channel
    overlay[..., 0] = FP                   # R channel
    overlay[..., 2] = FN                   # B channel

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    ax.imshow(overlay, interpolation="nearest")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(os.path.join(out_dir, f"{idx:04d}_overlay.png"),
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
       
 
# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg",  required=True, help="training YAML")
    ap.add_argument("--ckpt", required=True, help="model_snapshot_XXXX.pth")
    ap.add_argument("--out",  default="eval_gamma_edges", help="output dir")
    ap.add_argument("-n",     type=int, default=10, help="#samples to test")
    ap.add_argument("--cpu",  action="store_true", help="force CPU")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # config & model 
    cfg         = get_config(args.cfg, is_test=True)
    cfg.device  = "cpu" if args.cpu else cfg.device
    cfg.use_gpu = False if args.cpu else cfg.use_gpu
    dev         = torch.device(cfg.device)

    cprint(f"Loading checkpoint  {args.ckpt}", "cyan")
    net   = GRANMixtureBernoulli(cfg).to(dev)
    state = torch.load(args.ckpt, map_location=dev)
    net.load_state_dict(state["model"])
    net.eval()

    # dataset 
    ds = GammaSchemeDataset(cfg.dataset.data_path,
                            max_nodes=cfg.model.max_num_nodes)
    loader = torch.utils.data.DataLoader(ds, batch_size=1,
                                         shuffle=True, num_workers=0)

    # loop 
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(loader, total=args.n)):
            if idx >= args.n: break

            yy   = sample["yy_matrix"].to(dev)          # 1×1×512×512
            ctx  = net.yy_encoder(yy)
            A_pred = net._sampling(B=1, context_vec=ctx)[0]   # tensor

            n = int(sample["n_nodes"])
            A_pred = A_pred[:n, :n].cpu().numpy()
            A_pred = ((A_pred + A_pred.T) > 0.5).astype(np.uint8)

            gt_adj = sample["adj"][0] if sample["adj"].ndim == 3 else sample["adj"]
            A_gt   = gt_adj[:n, :n].numpy()
            A_gt   = ((A_gt + A_gt.T) > 0.5).astype(np.uint8)

            gt_G   = graph_from_adj(A_gt)
            pred_G = graph_from_adj(A_pred)

            save_graph_png(idx, gt_G, pred_G, args.out)
            save_edges_txt(idx, gt_G, pred_G, args.out)
            save_adj_overlay(idx, A_gt, A_pred, args.out)

    cprint(f"\n✓ wrote {idx+1} PNG + TXT pairs to “{args.out}”", "green")

if __name__ == "__main__":
    main()