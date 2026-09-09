"""Command line entry point: ``lcsa <command>``.

The pipeline splits at one seam.  ``build`` needs a GPU and produces the nested
cache; everything after it is CPU-only and reads that cache.  Each command
writes its artifacts to an output directory and prints a short summary, so a
Kaggle session that dies halfway loses at most the leg it was running.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("lcsa")

__all__ = ["main", "build_parser"]


def _load_config(path):
    if not path:
        return {}
    import yaml

    p = Path(path)
    if not p.exists():
        raise SystemExit(f"config file not found: {p}")
    return yaml.safe_load(p.read_text()) or {}


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _targets_csv(path: Path, keys) -> None:
    import pandas as pd

    pd.DataFrame(list(keys), columns=["text_id", "word_number"]).to_csv(path, index=False)


def _read_keys(path: Path):
    import pandas as pd

    df = pd.read_csv(path)
    return list(zip(df["text_id"].astype(int), df["word_number"].astype(int)))


# -- commands -----------------------------------------------------------------


def cmd_build(args) -> int:
    from lcsa.build import BuildConfig, build_corpus, gate_g0
    from lcsa.cache import ReferenceScorer
    from lcsa.data.provo import load_provo, read_provo_csv
    from lcsa.data.subtlex import load_subtlex
    from lcsa.gates import g0_data_integrity, g1_throughput
    from lcsa.store import save_corpus

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    provo = load_provo(args.provo_dir, require_eye=not args.no_eye)
    log.info("provo: %s", provo.summary())
    raw, enc = read_provo_csv(Path(args.provo_dir) / args.norms_name)
    g0 = g0_data_integrity(raw, provo)
    (out / "g0.json").write_text(json.dumps(g0.as_row(), indent=2, default=str))
    log.info("G0 %s: %s", "passed" if g0.passed else "FAILED", g0.measured)
    if not g0.passed and not args.force:
        raise SystemExit(
            "G0 failed and --force was not given; the fallback is in the gate record. "
            "Fix the decode or the join before spending GPU time on a wrong cache."
        )

    unigrams = load_subtlex(args.subtlex)
    if unigrams.source == "uniform":
        log.warning("SUBTLEX not supplied: the unigram channel is a uniform stand-in "
                    "and every result carries that label")
    scorer = ReferenceScorer(args.model, dtype=args.dtype)
    log.info("reference %s on %s, forward path %s", args.model, scorer.device,
             scorer.resolve_path())

    t0 = time.time()
    cfg = BuildConfig(max_candidates=args.max_candidates, max_depth=args.max_depth,
                      top_k_expansions=args.top_k)
    keys, words = [], []
    n_seen = [0]

    def progress(n):
        n_seen[0] = n
        if n % 50 == 0:
            log.info("built %d targets (%.1f s)", n, time.time() - t0)

    frozen = None
    if args.candidates or args.targets:
        if not (args.candidates and args.targets):
            raise SystemExit("--candidates and --targets go together; both come from the "
                             "primary build directory")
        frozen_keys = _read_keys(Path(args.targets))
        frozen_words = json.loads(Path(args.candidates).read_text())
        if len(frozen_keys) != len(frozen_words):
            raise SystemExit(f"{args.targets} has {len(frozen_keys)} rows but "
                             f"{args.candidates} has {len(frozen_words)} lists")
        frozen = dict(zip(frozen_keys, frozen_words))
        log.info("candidate sets frozen to the %d targets of %s", len(frozen), args.targets)
    corpus = build_corpus(provo, scorer, unigrams, cfg, limit=args.limit,
                          progress=progress, keep_words=words, keep_keys=keys,
                          candidates=frozen)
    dt = time.time() - t0
    log.info("built %d targets in %.1f s", len(corpus), dt)
    if frozen is not None and args.limit is None and keys != frozen_keys:
        raise SystemExit(
            f"the frozen build produced {len(keys)} targets in a different order from "
            f"the {len(frozen_keys)} of {args.targets}; the Provo files differ from the "
            "primary build's and this cache cannot serve as its reference")

    save_corpus(out / "cache.npz", corpus)
    _targets_csv(out / "targets.csv", keys)
    (out / "candidates.json").write_text(json.dumps(words))

    # G1 is a throughput measurement, so it is reported from the build itself
    # rather than from a separate benchmark that could differ in shape.
    ctx = sum(t.K + 1 for t in corpus)
    (out / "g1.json").write_text(json.dumps({
        "seconds": dt, "targets": len(corpus), "contexts": ctx,
        "contexts_per_second": ctx / dt if dt > 0 else None,
        "note": "TFLOP/s requires the checkpoint's FLOP count; "
                "contexts per second is the portable form",
    }, indent=2))
    g2 = gate_g0(corpus, provo)
    (out / "g0_cache.json").write_text(json.dumps(g2, indent=2, default=str))
    print(json.dumps({"targets": len(corpus), "clusters": corpus.n_clusters,
                      "mean_K": corpus.mean_K, "seconds": dt,
                      "cache": str(out / "cache.npz")}, indent=2))
    return 0


def _load(args):
    from lcsa.store import load_corpus

    p = Path(args.cache)
    if not p.exists():
        raise SystemExit(f"cache not found: {p}. Run `lcsa build` first.")
    return load_corpus(p)


def _models(names):
    from lcsa.likelihood import get_model

    return [get_model(n) for n in names]


def cmd_e1(args) -> int:
    from lcsa.experiments.e1_exactness import run

    corpus = _load(args)
    res = run(corpus, _models(args.estimators), args.out, seed=args.seed)
    print(json.dumps({"exactness_passed": res["exactness"]["passed"],
                      "gradients_passed": all(g["passed"] for g in res["gradients"]),
                      "g2_passed": res["g2"].passed}, indent=2))
    return 0


def _rep_range(args, n_default, start_attr="rep_start", stop_attr="rep_stop") -> range:
    """Replicates for this process: the whole run unless a shard range was given."""
    from lcsa.experiments.shards import rep_indices

    start, stop = getattr(args, start_attr), getattr(args, stop_attr)
    if start is None and stop is None:
        return rep_indices(n_default)
    return rep_indices(n_default, (start or 0, stop if stop is not None else n_default))


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_e2(args) -> int:
    from lcsa.experiments import e2_ladder as e2
    from lcsa.fitting import fit_constrained
    from lcsa.store import load_corpus

    corpus = _load(args)
    models = _models(args.estimators)
    gen = corpus
    if args.gen_cache:
        gp = Path(args.gen_cache)
        if not gp.exists():
            raise SystemExit(f"generating cache not found: {gp}")
        gen = load_corpus(gp)
        if len(gen) != len(corpus):
            raise SystemExit(f"{gp} has {len(gen)} targets but {args.cache} has "
                             f"{len(corpus)}; the ladder needs the same targets.csv")
    theta_path = Path(args.out) / "e2_theta0.json"
    if args.stage in ("all", "ladder"):
        # The nuisance vector comes from the human counts under the primary
        # estimator and is written once, so every coverage shard reads the same
        # numbers instead of refitting them on a node with a different BLAS.
        theta0 = fit_constrained(corpus, models[0], n_starts=3, seed=args.seed).theta
        theta_path.parent.mkdir(parents=True, exist_ok=True)
        theta_path.write_text(json.dumps({"theta0": [float(x) for x in theta0],
                                          "estimator": models[0].name,
                                          "gen_cache": args.gen_cache}))
    else:
        if not theta_path.exists():
            raise SystemExit(f"{theta_path} is missing; run `lcsa e2 --stage ladder` first")
        theta0 = np.asarray(json.loads(theta_path.read_text())["theta0"], dtype=np.float64)
    if args.stage == "all":
        res = e2.run(gen, corpus, theta0, models, args.out, n_rep=args.n_rep, seed=args.seed)
    elif args.stage == "ladder":
        e2.run_ladder(gen, corpus, theta0, models, args.out, seed=args.seed)
        _print({"stage": "ladder", "out": args.out})
        return 0
    else:
        reps = _rep_range(args, args.n_rep)
        rows = e2.run_coverage_shard(gen, corpus, theta0, models, args.out, reps,
                                     seed=args.seed)
        _print({"stage": "coverage", "replicates": [reps.start, reps.stop],
                "rows": len(rows), "failed": sum(1 for r in rows if r.get("failed"))})
        return 0
    _print({"g6_passed": res["g6"].passed, "coverage": res["g6"].measured,
            "ceiling": res["ceiling"]})
    return 0


def _e3_h_specs(args, corpus) -> dict:
    from lcsa.experiments.e3_nulls import build_h_order, build_h_topic, h_lexical

    wanted = [x.strip().lower() for x in args.nulls.split(",") if x.strip()]
    wanted = [w for w in wanted if w != "none"]
    h_specs = {}
    if "lex" in wanted:
        from lcsa.likelihood import REPAIRED

        h_specs["N-LEX"] = h_lexical(corpus, REPAIRED, seed=args.seed)
    if {"topic", "order"} & set(wanted):
        if not (args.provo_dir and args.targets and args.candidates):
            raise SystemExit(
                "the topic and order nulls need --provo-dir, --targets and "
                "--candidates from the build step, plus the reference checkpoint"
            )
        from lcsa.cache import ReferenceScorer, context_string
        from lcsa.data.provo import load_provo

        provo = load_provo(args.provo_dir, require_eye=False)
        keys = _read_keys(Path(args.targets))
        words = json.loads(Path(args.candidates).read_text())
        contexts = [context_string(provo.passages[int(t)], int(w) - 1, None)
                    for t, w in keys]
        scorer = ReferenceScorer(args.model, dtype=args.dtype)
        if "topic" in wanted:
            h_specs["N-TOPIC"] = build_h_topic(corpus, words, contexts, scorer)
        if "order" in wanted:
            h_specs["N-ORDER"] = build_h_order(corpus, words, contexts, scorer,
                                               seed=args.seed)
    return h_specs


def cmd_e3(args) -> int:
    from lcsa.experiments import e3_nulls as e3

    corpus = _load(args)
    models = _models(args.estimators)
    readers = ([x.strip() for x in args.readers.split(",") if x.strip()]
               if args.readers else None)
    if args.stage == "all":
        res = e3.run(corpus, models, args.out, h_specs=_e3_h_specs(args, corpus) or None,
                     n_rep=args.n_rep, n_boot=args.n_boot, seed=args.seed,
                     fit_human=not args.no_human, readers=readers)
        _print({"g5_passed": res["g5"].passed,
                "rates": [{k: r[k] for k in ("reader", "estimator", "reject_cluster_robust",
                                             "reject_naive_LR")}
                          for r in res["rejection_rates"]]})
        return 0
    if args.stage == "prepare":
        prep = e3.prepare(corpus, models, args.out, _e3_h_specs(args, corpus) or None,
                          seed=args.seed, readers=readers)
        _print({"stage": "prepare", "readers": list(prep["readers"]),
                "calibration": prep["calibration"]})
        return 0
    prep = e3.load_prepared(args.out, corpus)
    if args.stage == "replicates":
        reps = _rep_range(args, args.n_rep)
        out = {}
        for nm in readers or list(prep["readers"]):
            rows = e3.run_replicate_shard(corpus, prep, models, args.out, nm, reps,
                                          seed=args.seed)
            out[nm] = {"rows": len(rows), "failed": sum(1 for r in rows if r["failed"])}
        _print({"stage": "replicates", "replicates": [reps.start, reps.stop], "readers": out})
    elif args.stage == "human":
        human = e3.run_human(corpus, prep, models, args.out, seed=args.seed)
        _print({"stage": "human", "fits": [{k: f[k] for k in ("estimator", "delta_hat",
                                                              "p_headline")}
                                           for f in human["fits"]]})
    else:
        reps = _rep_range(args, args.n_boot, "boot_start", "boot_stop")
        recs = e3.run_contrast_shard(corpus, prep, models, args.out, reps, seed=args.seed)
        _print({"stage": "contrast", "replicates": [reps.start, reps.stop], "rows": len(recs)})
    return 0


def _e4_references(args, corpus, keys) -> dict | None:
    """The zero-decay references for the sweep, keyed by name, primary first."""
    from lcsa.store import load_corpus

    refs = {"primary": None}
    for spec in args.reference or []:
        if "=" not in spec:
            raise SystemExit(f"--reference takes NAME=DIR, got {spec!r}")
        name, d = spec.split("=", 1)
        d = Path(d)
        if not (d / "cache.npz").exists() or not (d / "targets.csv").exists():
            raise SystemExit(f"reference {name}: {d} needs cache.npz and targets.csv")
        if _read_keys(d / "targets.csv") != keys:
            raise SystemExit(f"reference {name}: {d / 'targets.csv'} lists different "
                             f"targets from {args.targets}; rebuild it with --candidates "
                             "and --targets from the primary build")
        refs[name] = load_corpus(d / "cache.npz")
    if args.tilted_from:
        from lcsa.experiments.e3_nulls import FLOORS, load_prepared
        from lcsa.readers import tilted_cache

        prep = load_prepared(args.tilted_from, corpus)
        for nm, rec in prep["readers"].items():
            if nm in FLOORS or "directions" not in rec:
                continue
            refs[nm] = tilted_cache(corpus, rec["directions"], rec["alpha"])
    return refs if len(refs) > 1 else None


def cmd_e4(args) -> int:
    from lcsa.data.provo import load_provo
    from lcsa.data.subtlex import load_subtlex
    from lcsa.experiments import e4_reading as e4
    from lcsa.fitting import fit

    corpus = _load(args)
    models = _models(args.estimators)
    provo = load_provo(args.provo_dir, require_eye=True)
    keys = _read_keys(Path(args.targets))
    if len(keys) != len(corpus):
        raise SystemExit(
            f"{args.targets} has {len(keys)} rows but the cache has {len(corpus)} "
            "targets; they must come from the same build"
        )
    unigrams = load_subtlex(args.subtlex)
    y, ctrl, passage = e4.gaze_table(provo, keys, unigrams)
    refs = _e4_references(args, corpus, keys)
    if args.stage in ("all", "sweep"):
        f = fit(corpus, models[0], n_starts=3, seed=args.seed)
        fitted = {"human": (f.theta, models[0])}
    if args.stage == "all":
        res = e4.run(corpus, y, ctrl, passage, models, args.out, references=refs,
                     fitted=fitted, n_boot=args.n_boot, n_folds=args.n_folds,
                     seed=args.seed)
        _print({"selected_k": res["selected"], "prediction_8": res["prediction_8"]})
    elif args.stage == "sweep":
        stage = e4.run_sweep(corpus, y, ctrl, passage, models, args.out, references=refs,
                             fitted=fitted, n_folds=args.n_folds, seed=args.seed)
        _print({"stage": "sweep", "selected_k": {n: s["argmax_k"]
                                                 for n, s in stage["sweeps"].items()}})
    else:
        reps = _rep_range(args, args.n_boot, "boot_start", "boot_stop")
        rows = e4.run_argmax_shard(corpus, y, ctrl, passage, args.out, reps,
                                   references=refs, seed=args.seed)
        _print({"stage": "boot", "replicates": [reps.start, reps.stop], "rows": len(rows)})
    return 0


def cmd_merge(args) -> int:
    """Combine the stage outputs and shards of each leg into the registered tables."""
    legs = [x.strip().lower() for x in args.legs.split(",") if x.strip()]
    models = _models(args.estimators)
    out = {}
    for leg in legs:
        if leg == "e2":
            from lcsa.experiments.e2_ladder import merge

            res = merge(args.out, models)
            out["e2"] = {"g6_passed": res["g6"].passed, "coverage": res["g6"].measured}
        elif leg == "e3":
            from lcsa.experiments.e3_nulls import merge

            res = merge(args.out, margin=args.margin)
            out["e3"] = {"g5_passed": res["g5"].passed,
                         "human": None if res["human"] is None else res["human"]["g4"]}
        elif leg == "e4":
            from lcsa.experiments.e4_reading import merge

            res = merge(args.out)
            out["e4"] = {"selected_k": res["selected"], "prediction_8": res["prediction_8"]}
        else:
            raise SystemExit(f"unknown leg {leg!r}; choose from e2, e3, e4")
    _print(out)
    return 0


def cmd_selftest(args) -> int:
    """End-to-end run on a synthetic corpus, with no data and no GPU.

    This is the command to run first on a new machine: it exercises every leg the
    real pipeline uses, at a size that finishes in about a minute, and fails
    loudly if any of them is broken.
    """
    from lcsa.corpusdata import Corpus
    from lcsa.experiments.e1_exactness import run as run_e1
    from lcsa.experiments.e2_ladder import run as run_e2
    from lcsa.experiments.e3_nulls import h_lexical, run as run_e3
    from lcsa.experiments.e4_reading import run as run_e4, window_surprisal
    from lcsa.fitting import fit_constrained
    from lcsa.likelihood import NAIVE, REPAIRED, evaluate_target

    rng = np.random.default_rng(args.seed)
    T, C = 120, 12
    P_list, u, f, g, cl = [], [], [], [], []
    for t in range(T):
        K = int(rng.integers(4, 20))
        V = int(rng.integers(8, 16))
        rows = [rng.dirichlet(np.ones(V) * 0.5)]
        for _ in range(K):
            rows.append(np.abs(rows[-1] + rng.normal(scale=0.03, size=V)))
        P = np.asarray(rows)
        P /= P.sum(axis=1, keepdims=True)
        P_list.append(P)
        u.append(rng.dirichlet(np.ones(V)))
        f.append(rng.normal(size=(V, 4)))
        g.append((rng.random(V) < 0.2).astype(float))
        cl.append(t % C)
    corpus = Corpus(P_list, [np.zeros(P.shape[1]) for P in P_list], u, f, g, cl,
                    target_slots=[0] * T)
    theta = np.array([0.316, 0.10, 0.9])
    counts = [rng.multinomial(40, evaluate_target(t, theta, NAIVE, corpus.M).q).astype(float)
              for t in corpus]
    corpus = corpus.with_counts(counts)

    out = Path(args.out)
    r1 = run_e1(corpus, [NAIVE, REPAIRED], out / "e1", seed=args.seed)
    assert r1["exactness"]["passed"], "exactness identities failed"
    assert all(x["passed"] for x in r1["gradients"]), "gradient check failed"

    th0 = fit_constrained(corpus, NAIVE, n_starts=2, seed=args.seed).theta
    r2 = run_e2(corpus, corpus, th0, [NAIVE], out / "e2", n_rep=args.n_rep,
                coverage_rungs=(8.0,), seed=args.seed)
    r3 = run_e3(corpus, [NAIVE, REPAIRED], out / "e3",
                h_specs={"N-LEX": h_lexical(corpus, REPAIRED, seed=args.seed)},
                n_rep=args.n_rep, n_boot=args.n_boot, seed=args.seed)
    passage = np.array([t.cluster for t in corpus])
    s = window_surprisal(corpus, 4)
    y = 250 + 12 * np.nan_to_num(s) + rng.normal(0, 25, len(corpus))
    ctrl = np.column_stack([rng.normal(size=len(corpus)), rng.normal(size=len(corpus))])
    r4 = run_e4(corpus, y, ctrl, passage, [NAIVE], out / "e4", k_grid=(0, 2, 4, 8),
                n_boot=5, n_folds=4, use_mixed=False, seed=args.seed)

    summary = {
        "e1_exactness": r1["exactness"]["passed"],
        "e1_residual_fraction_naive": r1["residual_fraction"][0]["residual_fraction"],
        "e2_coverage": r2["g6"].measured["coverage"],
        "e3_floor_rates": r3["g5"].measured["rates"],
        "e3_human_delta": [x["delta_hat"] for x in r3["human"]["fits"]],
        "e4_selected_k": r4["selected"],
    }
    print(json.dumps(summary, indent=2, default=str))
    ok = (r1["exactness"]["passed"]
          and np.isfinite(summary["e1_residual_fraction_naive"])
          and r4["selected"]["primary"] is not None)
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lcsa", description=__doc__)
    p.add_argument("--config", help="YAML file whose keys become defaults")
    p.add_argument("--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="build the nested cache (needs a GPU)")
    b.add_argument("--provo-dir", required=True)
    b.add_argument("--norms-name", default="Provo_Corpus-Predictability_Norms.csv")
    b.add_argument("--subtlex", default=None)
    b.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    b.add_argument("--dtype", default="float16")
    b.add_argument("--out", default="artifacts/build")
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--max-depth", type=int, default=32)
    b.add_argument("--max-candidates", type=int, default=120)
    b.add_argument("--top-k", type=int, default=50)
    b.add_argument("--no-eye", action="store_true", help="skip the eye-tracking arm")
    b.add_argument("--force", action="store_true", help="build even if G0 fails")
    b.add_argument("--candidates", default=None,
                   help="candidates.json of the primary build: freeze its candidate sets")
    b.add_argument("--targets", default=None,
                   help="targets.csv of the primary build, paired with --candidates")
    b.set_defaults(func=cmd_build)

    def common(sp):
        sp.add_argument("--cache", default="artifacts/build/cache.npz")
        sp.add_argument("--out", default="artifacts")
        sp.add_argument("--estimators", nargs="+", default=["naive", "repaired"])
        sp.add_argument("--seed", type=int, default=0)

    e1 = sub.add_parser("e1", help="exactness, sensitivity, residual fractions")
    common(e1)
    e1.set_defaults(func=cmd_e1)

    def shard(sp, prefix, what):
        sp.add_argument(f"--{prefix}-start", type=int, default=None,
                        help=f"first {what} of this shard (inclusive)")
        sp.add_argument(f"--{prefix}-stop", type=int, default=None,
                        help=f"one past the last {what} of this shard")

    e2 = sub.add_parser("e2", help="recovery ladder, coverage, identification ceiling")
    common(e2)
    e2.add_argument("--n-rep", type=int, default=200)
    e2.add_argument("--stage", choices=["all", "ladder", "coverage"], default="all")
    e2.add_argument("--gen-cache", default=None,
                    help="cache.npz of another checkpoint to generate the ladder under")
    shard(e2, "rep", "coverage replicate")
    e2.set_defaults(func=cmd_e2)

    e3 = sub.add_parser("e3", help="zero-decay nulls, then the human fit")
    common(e3)
    e3.add_argument("--n-rep", type=int, default=200)
    e3.add_argument("--n-boot", type=int, default=200)
    e3.add_argument("--nulls", default="lex",
                    help="comma list of lex,topic,order; none for the floors alone")
    e3.add_argument("--provo-dir", default=None)
    e3.add_argument("--targets", default="artifacts/build/targets.csv")
    e3.add_argument("--candidates", default="artifacts/build/candidates.json")
    e3.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    e3.add_argument("--dtype", default="float16")
    e3.add_argument("--no-human", action="store_true")
    e3.add_argument("--stage", choices=["all", "prepare", "replicates", "human", "contrast"],
                    default="all")
    e3.add_argument("--readers", default=None,
                    help="comma list of readers to prepare or replicate, default all")
    shard(e3, "rep", "null replicate")
    shard(e3, "boot", "contrast bootstrap replicate")
    e3.set_defaults(func=cmd_e3)

    e4 = sub.add_parser("e4", help="context-limitation sweep and reading times")
    common(e4)
    e4.add_argument("--provo-dir", required=True)
    e4.add_argument("--targets", default="artifacts/build/targets.csv")
    e4.add_argument("--subtlex", default=None)
    e4.add_argument("--n-boot", type=int, default=200)
    e4.add_argument("--n-folds", type=int, default=None)
    e4.add_argument("--stage", choices=["all", "sweep", "boot"], default="all")
    e4.add_argument("--reference", action="append", default=None, metavar="NAME=DIR",
                    help="a build directory of another checkpoint on the frozen "
                         "candidate sets; repeatable")
    e4.add_argument("--tilted-from", default=None, metavar="DIR",
                    help="an E3 output directory whose prepared tilts become references")
    shard(e4, "boot", "argmax bootstrap replicate")
    e4.set_defaults(func=cmd_e4)

    mg = sub.add_parser("merge", help="combine shards into the registered tables")
    mg.add_argument("--out", default="artifacts")
    mg.add_argument("--legs", default="e2,e3,e4")
    mg.add_argument("--estimators", nargs="+", default=["naive", "repaired"])
    mg.add_argument("--margin", type=float, default=0.25, help="TOST margin for E3")
    mg.set_defaults(func=cmd_merge)

    st = sub.add_parser("selftest", help="synthetic end-to-end run, no data, no GPU")
    st.add_argument("--out", default="artifacts/selftest")
    st.add_argument("--n-rep", type=int, default=8)
    st.add_argument("--n-boot", type=int, default=20)
    st.add_argument("--seed", type=int, default=0)
    st.set_defaults(func=cmd_selftest)
    return p


def main(argv=None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    cfg = _load_config(getattr(args, "config", None))
    for k, v in cfg.items():
        key = k.replace("-", "_")
        if hasattr(args, key) and getattr(args, key) in (None, False):
            setattr(args, key, v)
    _setup_logging(args.verbose)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
