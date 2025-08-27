import pickle, argparse, matplotlib.pyplot as plt, os, re, glob

ap = argparse.ArgumentParser()
ap.add_argument("exp_dir", help="e.g. exp/GRAN/GRANMixtureBernoulli_gamma_*")
ap.add_argument("--png", default=None, help="optional output file")
args = ap.parse_args()

# ── find the newest run if a glob is given
runs = sorted(glob.glob(args.exp_dir))
if not runs:
    ap.error(f"No match for {args.exp_dir!r}")
exp = runs[-1]
stats_p = os.path.join(exp, "train_stats.p")

results = pickle.load(open(stats_p, "rb"))
steps  = results["train_step"]
losses = results["train_loss"]

plt.figure(figsize=(6,3))
plt.plot(steps, losses, lw=1.4)
plt.xlabel("iteration")
plt.ylabel("NLL loss")
plt.title(os.path.basename(exp))
plt.grid(True, alpha=.3)

if args.png:
    plt.tight_layout()
    plt.savefig(args.png, dpi=150)
    print("✓ saved", args.png)
else:
    plt.show()
