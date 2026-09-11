"""What the Slurm scripts do that a local run does not.

The cluster points legs at directories nothing creates, requeues single array
tasks, and cuts a participant list whose length is not known until the file is
read.  Each of these produced a failure that only appeared on the cluster, so
each has a test here.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from conftest import draw_true_delta, make_corpus

from lcsa.cli import main
from lcsa.store import save_corpus

SLURM = Path(__file__).resolve().parents[1] / "slurm"


def _cache(tmp_path, counts=True):
    corpus = make_corpus(n_targets=60, n_clusters=10, seed=17)
    if counts:
        corpus = draw_true_delta(corpus, 0.316, seed=18, n_per_target=60)
    p = tmp_path / "build" / "cache.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    save_corpus(p, corpus)
    return corpus, p


def test_a_leg_creates_the_output_directory_the_sbatch_scripts_assume(tmp_path):
    """``e3_self.sbatch`` writes into ``$LCSA_ART/self_<slug>`` and
    ``refsweep.sbatch`` into ``$LCSA_ART/refsweep/<slug>``; nothing creates
    either, so the leg used to load its cache, compute, then die on the first
    write."""
    _, cache = _cache(tmp_path)
    out = tmp_path / "artifacts" / "self_gpt2"
    assert not out.exists()
    assert main(["e1", "--cache", str(cache), "--out", str(out)]) == 0
    assert out.is_dir() and any(out.iterdir())


def _participant_file(corpus, path, n_people, seed=5):
    keys = [(1, i + 1) for i in range(len(corpus))]
    words = [[f"w{j}" for j in range(t.V)] for t in corpus]
    rows = [{"participant": f"p{p}", "text_id": tid, "word_number": wn, "response": w[0]}
            for p in range(n_people) for (tid, wn), w in zip(keys, words)]
    pd.DataFrame(rows).to_csv(path, index=False)
    return keys, words


def test_e5_refuses_a_participant_file_the_shards_do_not_tile(tmp_path, monkeypatch):
    """The array cuts the list by position before the file has been read, so a
    file with more participants than N_PART would drop the tail of the list
    from the last shard without a word."""
    from lcsa.experiments import e5_participants as e5

    monkeypatch.setattr(e5, "MIN_TARGETS", 1)
    corpus, cache = _cache(tmp_path, counts=False)
    part = tmp_path / "cloze.csv"
    keys, words = _participant_file(corpus, part, n_people=6)
    (tmp_path / "targets.csv").write_text(
        "text_id,word_number\n" + "".join(f"{a},{b}\n" for a, b in keys))
    (tmp_path / "candidates.json").write_text(json.dumps(words))
    argv = ["e5", "--cache", str(cache), "--out", str(tmp_path / "art"),
            "--participants", str(part), "--targets", str(tmp_path / "targets.csv"),
            "--candidates", str(tmp_path / "candidates.json"), "--stage", "pooled",
            "--estimators", "naive"]
    with pytest.raises(SystemExit, match="yields 6 participants.*shards tile 470"):
        main(argv + ["--n-participants", "470"])
    assert main(argv + ["--n-participants", "6"]) == 0


def _shard_range(i, n, total):
    script = f'. "{SLURM}/env.sh" >/dev/null 2>&1; shard_range {i} {n} {total}'
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "LCSA_VARS_ONLY": "1",
                            "USER": "t", "HOME": str(Path.home()),
                            "HF_HOME": "/tmp/lcsa-shard-range-test/hf",
                            "LCSA_ROOT": "/tmp/lcsa-shard-range-test"})
    assert r.returncode == 0, r.stderr
    return tuple(int(x) for x in r.stdout.split())


def test_shard_range_tiles_every_replicate_exactly_once():
    for n in (1, 2, 3, 7, 10, 40):
        for total in (1, 20, 200, 201):
            spans = [_shard_range(i, n, total) for i in range(n)]
            assert spans[0][0] == 0 and spans[-1][1] == total
            assert all(b[0] == a[1] for a, b in zip(spans, spans[1:]))


def test_the_sbatch_scripts_take_the_shard_count_from_the_registered_variable():
    """Requeueing one failed array task leaves SLURM_ARRAY_TASK_MAX at that
    task's own index, so a count read from it recomputed the wrong range."""
    for name in ("e2_cov", "e4_boot", "e6_crossed", "e3_human", "e5_participants"):
        lines = [ln for ln in (SLURM / f"{name}.sbatch").read_text().splitlines()
                 if not ln.lstrip().startswith("#")]
        assert not any("SLURM_ARRAY_TASK_MAX" in ln for ln in lines), name
        assert any("shard_range" in ln for ln in lines), name


def test_every_array_script_guards_its_index_against_the_list_it_indexes():
    for name in ("build_refs", "refsweep", "e3_reps"):
        text = (SLURM / f"{name}.sbatch").read_text()
        assert "SLURM_ARRAY_TASK_ID\" -lt" in text or 'SLURM_ARRAY_TASK_ID" -lt' in text, name


def test_the_appendix_sweep_cannot_cancel_the_merge():
    """``refsweep.sbatch`` feeds one appendix table and runs the two largest
    checkpoints, one of them gated.  Under ``afterok`` an out-of-memory task or
    a checkpoint the prefetch job skipped would leave the merge cancelled with
    DependencyNeverSatisfied, losing every registered leg with it."""
    pipeline = (SLURM / "pipeline.sh").read_text()
    merge = next(ln for ln in pipeline.splitlines() if ln.startswith("MERGE="))
    assert "afterany:$SWEEP" in merge, merge
    assert ":$SWEEP" not in merge.split("afterok:")[1].split(",")[0], merge
    sweep = (SLURM / "refsweep.sbatch").read_text()
    assert "require_prefetched" not in [ln.strip().split()[0] for ln in sweep.splitlines()
                                        if ln.strip() and not ln.lstrip().startswith("#")]
    assert "exit 0" in sweep


def test_the_submit_script_checks_every_shard_count_against_what_it_tiles():
    """A count above the grid hands the last tasks an empty range, which each of
    them discovers separately after Slurm has queued them all."""
    pipeline = (SLURM / "pipeline.sh").read_text()
    block = pipeline.split("done <<EOF\n")[1].split("EOF\n")[0]
    declared = dict(ln.split() for ln in block.strip().splitlines())
    assert declared == {"E2_SHARDS": "$N_REP", "E3_SHARDS": "$N_REP", "E6_SHARDS": "$N_REP",
                        "E3_BOOT_SHARDS": "$N_BOOT", "E4_SHARDS": "$N_BOOT",
                        "E5_SHARDS": "$N_PART"}, declared


def test_the_first_job_logs_somewhere_that_already_exists():
    """Slurm rejects a job whose ``--output`` directory is absent, and on a fresh
    account no directory under the lab scratch exists yet.  ``setup.sbatch`` is
    the job that creates them, so its own log has to be relative to the
    directory it was submitted from."""
    out = [ln for ln in (SLURM / "setup.sbatch").read_text().splitlines()
           if ln.startswith("#SBATCH --output=")]
    assert len(out) == 1, out
    assert "/" not in out[0].split("=", 1)[1], out[0]
    for name in ("e1", "build", "merge", "e2_cov"):
        other = [ln for ln in (SLURM / f"{name}.sbatch").read_text().splitlines()
                 if ln.startswith("#SBATCH --output=")]
        assert other and "/logs/" in other[0], name
