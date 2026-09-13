"""The resource lines every sbatch script has to satisfy on this cluster.

Slurm accepts a job whose limits are wrong for the partition it names and then
fails it hours later, or silently, so the facts the SCU scheduler imposes are
asserted here against the files rather than discovered from a cancelled run.
Every number below comes from the cluster's own published configuration:
`scu-cpu` allows seven days, `scu-gpu` two, a job with no `--mem` gets 8000M,
a job with no `--time` gets the partition maximum because DefaultTime is NONE,
and QOS `low` is rejected on both production partitions.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SLURM = Path(__file__).resolve().parents[1] / "slurm"
SBATCH = sorted(SLURM.glob("*.sbatch"))

PARTITION_MAX_SECONDS = {"scu-cpu": 7 * 86400, "scu-gpu": 2 * 86400}
# The smallest GPU node has 112 cpus and 754 GB; anything larger never starts.
NODE_MAX_CPUS = 112
NODE_MAX_MEM_MB = 754 * 1000


def directives(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"#SBATCH\s+--([a-z-]+)=(.*)", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def seconds(spec: str) -> int:
    days, _, rest = spec.rpartition("-")
    h, m, s = (int(x) for x in rest.split(":"))
    return int(days or 0) * 86400 + h * 3600 + m * 60 + s


def megabytes(spec: str) -> int:
    unit = spec[-1]
    n = int(spec[:-1] if unit.isalpha() else spec)
    return {"K": n // 1000, "M": n, "G": n * 1000, "T": n * 1000 * 1000}[unit.upper()]


def test_there_are_sbatch_scripts_to_check():
    assert len(SBATCH) >= 15, SBATCH


@pytest.mark.parametrize("path", SBATCH, ids=lambda p: p.name)
def test_every_job_sets_its_own_memory_and_time(path):
    d = directives(path)
    # Omitting either takes a default that is wrong in opposite directions: 8000M
    # is below what a build needs, and the partition maximum holds a slot for
    # days after the work is done.
    assert "mem" in d, f"{path.name} has no --mem, so it would get the 8000M default"
    assert "time" in d, f"{path.name} has no --time, so it would get the partition maximum"
    assert "partition" in d, f"{path.name} names no partition"
    assert "cpus-per-task" in d, f"{path.name} sets no --cpus-per-task"


@pytest.mark.parametrize("path", SBATCH, ids=lambda p: p.name)
def test_every_job_fits_the_partition_it_names(path):
    d = directives(path)
    part = d["partition"]
    assert part in PARTITION_MAX_SECONDS, f"{path.name} names partition {part}"
    assert seconds(d["time"]) <= PARTITION_MAX_SECONDS[part], (
        f"{path.name} asks for {d['time']} on {part}, over its ceiling")
    assert megabytes(d["mem"]) <= NODE_MAX_MEM_MB, f"{path.name} asks for {d['mem']}"
    assert int(d["cpus-per-task"]) <= NODE_MAX_CPUS, f"{path.name} asks for {d['cpus-per-task']} cpus"


@pytest.mark.parametrize("path", SBATCH, ids=lambda p: p.name)
def test_no_job_sets_a_qos_or_an_account(path):
    d = directives(path)
    # accardilab is the default account, and a single --qos cannot be right for
    # both partitions: scu-cpu takes [normal, cpu-limited] and scu-gpu takes
    # [normal, gpu-limited], so anything set globally is rejected on one of them.
    assert "qos" not in d, f"{path.name} sets --qos={d.get('qos')}"
    assert "account" not in d, f"{path.name} sets --account={d.get('account')}"
    assert "partition" in d and not d["partition"].startswith("preempt"), (
        f"{path.name} uses a preempt partition, where PreemptMode=CANCEL loses the work")


@pytest.mark.parametrize("path", SBATCH, ids=lambda p: p.name)
def test_a_gpu_job_asks_for_a_gpu_and_a_cpu_job_does_not(path):
    d = directives(path)
    if d["partition"] == "scu-gpu":
        assert "gres" in d, f"{path.name} is on scu-gpu with no --gres, so it gets no card"
        assert re.fullmatch(r"gpu(:[a-z0-9]+)?:[1-9]\d*", d["gres"]), d["gres"]
    else:
        assert "gres" not in d, f"{path.name} is on {d['partition']} and asks for {d.get('gres')}"


@pytest.mark.parametrize("path", SBATCH, ids=lambda p: p.name)
def test_an_array_job_gives_each_task_its_own_log(path):
    d = directives(path)
    if "array" not in d:
        return
    assert "%a" in d["output"] or "%j" in d["output"], (
        f"{path.name} is an array whose tasks would all append to {d['output']}")
    lo, _, hi = d["array"].partition("-")
    assert int(hi or lo) < 100000, f"{path.name} exceeds MaxArraySize"


@pytest.mark.parametrize("path", SBATCH, ids=lambda p: p.name)
def test_nothing_writes_to_the_node_local_scratch(path):
    # $TMPDIR is /scratch/$USER_$JOBID and Slurm deletes it when the job ends.
    body = path.read_text()
    assert "$TMPDIR" not in body, f"{path.name} writes to $TMPDIR, which is deleted at job end"
    assert "/scu-storage03" not in body, f"{path.name} names a login-only filesystem"


def test_the_gres_the_submit_script_passes_matches_the_files():
    # pipeline.sh overrides each file's --gres with $LCSA_GPU_GRES, so a default
    # that named a card the build cannot fit would silently replace the pinned one.
    env = (SLURM / "env.sh").read_text()
    for var in ("LCSA_GPU_GRES", "LCSA_SWEEP_GPU_GRES"):
        m = re.search(rf'{var}:-([^}}]+)}}', env)
        assert m, f"{var} has no default in env.sh"
        assert re.fullmatch(r"gpu(:[a-z0-9]+)?:[1-9]\d*", m.group(1)), m.group(1)


PIPELINE = (SLURM / "pipeline.sh").read_text()
# The installer is run by hand before anything else and is the one script the
# chain does not submit; every other file here is a job or it is dead.
NOT_SUBMITTED = {"setup.sbatch"}


@pytest.mark.parametrize("path", SBATCH, ids=lambda p: p.name)
def test_the_chain_submits_every_job_this_directory_holds(path):
    """A job nothing submits is a registered prediction nobody scores.

    Predictions 12 and 13 were unscoreable for exactly this reason: the flags
    that carry them existed on the command line and no sbatch passed them.
    """
    if path.name in NOT_SUBMITTED:
        return
    submitted = [ln for ln in PIPELINE.splitlines()
                 if "jid " in ln and path.name in ln]
    assert submitted, f"{path.name} is never passed to jid in slurm/pipeline.sh"


def test_the_two_optional_inputs_are_read_only_where_a_job_writes_them():
    """Both are joined with afterany, so both readers must tolerate absence."""
    e1 = (SLURM / "e1.sbatch").read_text()
    human = (SLURM / "e3_human.sbatch").read_text()
    assert "--prefix-probe" in e1 and "prefix_probe.json" in e1
    assert "--uncapped-cache" in human and "build_uncapped/cache.npz" in human
    # afterok on either would let one unscored prediction cancel a whole leg.
    assert "afterany:$PROBE" in PIPELINE
    assert "afterany:$UNC" in PIPELINE
