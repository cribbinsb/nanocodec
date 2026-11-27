"""The shard layout (format version 6): the columns every shard must have.

The encoder declares its columns in one place (crates/nc-cli/src/label_columns.rs) and lists
them with their dtypes and row shapes in each shard's labels.json. The loader checks the
shard's arrays against that list, and the list against REQUIRED here: the columns the
validator and the tools in this package read. A column the encoder adds later is loaded and
checked without a change here. docs/analysis/nn-dataset-schema.md says what each one means.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The shard format the encoder writes (label_dump.rs FORMAT_VERSION).
FORMAT_VERSION = 7
#: Candidate slots of a node, in the encoder's order (labels.rs CANDIDATE_SLOTS).
SLOTS = ("leaf", "split", "horizontal", "vertical", "horizontal_triple", "vertical_triple")
SLOT_LEAF, SLOT_SPLIT = 0, 1
#: A node below a planned leaf has no planned slot.
SLOT_NONE = 255
#: node.gates bits (labels.rs GATE_*): what the preset's gates allow at the node.
GATE_MARGIN_DEEPEN, GATE_MARGIN_MERGE, GATE_CONTEXT_DEEPEN, GATE_CONTEXT_MERGE = 1, 2, 4, 8
GATE_RESIDUAL_DEEPEN = 16
#: leaf.mode values (labels.rs MODE_*).
MODE_INTRA, MODE_INTER, MODE_SECONDARY, MODE_COMPOUND, MODE_STATIC_SKIP = 0, 1, 2, 3, 4
MODE_PRIMARY_PAIR = 5
MOTION_GRID = 8
#: Values per cell (nc-analysis cell_stats.rs) in sb.cell_source: mean, variance, gradient
#: energy, 8 orientation bins, noise floor (the source alone: stored once per clip in a
#: released version); in sb.cell_motion: zero SAD, vector SAD, residual energy.
CELL_SOURCE, CELL_MOTION = 12, 3
#: The sb columns that depend on the source alone (a version's patch store holds them).
SOURCE_COLUMNS = ("patch", "cell_source")
PATCH_SIDE = 32
#: The trial value network's inputs per node (crates/nc-encoder/src/trial_inputs.rs).
TRIAL_INPUTS = 11
TABLES = ("frame", "sb", "node", "leaf")


#: A row-shape entry that is the shard's tool count (labels.json "tools": the preset's
#: own slot, then each `--label-tool`).
TOOLS = -1


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str
    row_shape: tuple[int, ...] = ()

    def shape_for(self, tools: int) -> tuple[int, ...]:
        return tuple(tools if d == TOOLS else d for d in self.row_shape)


PAIR, QUAD, PATCH = (2,), (4,), (PATCH_SIDE, PATCH_SIDE)
SLOT_ROW = (len(SLOTS),)

#: The columns this package reads, per table (a shard may declare more).
COLUMNS: dict[str, tuple[Column, ...]] = {
    "frame": (
        Column("index", "<u4"), Column("key", "|u1"), Column("temporal_layer", "|u1"),
        Column("key_distance", "<u4"), Column("quantizer", "<u4"), Column("lambda", "<f4"),
        Column("noise_sigma", "<f4"), Column("rate", "<f4", QUAD),
        Column("width", "<u4"), Column("height", "<u4"), Column("superblock_log2", "|u1"),
        Column("min_block_log2", "|u1"),
    ),
    "sb": (
        Column("frame", "<u4"), Column("x", "<u4"), Column("y", "<u4"), Column("inside", "|u1"),
        Column("cost", "<f4"), Column("ns", "<u8"), Column("luma_mean", "<f4"),
        Column("luma_variance", "<f4"), Column("gradient", "<f4", PAIR),
        Column("coherence", "<f4"), Column("block_variance", "<f4", PAIR),
        Column("static_share", "<f4"), Column("still", "<f4", PAIR), Column("zero_sad", "<f4"),
        Column("best_sad", "<f4"), Column("motion", "<f4", QUAD),
        Column("search_vector", "<i2", PAIR), Column("residual_energy", "<f4"),
        Column("second_residual", "<f4", PAIR),
        Column("block_sad", "<f4", (MOTION_GRID, MOTION_GRID)),
        Column("block_mv", "<f4", (MOTION_GRID, MOTION_GRID, 2)),
        Column("cell_source", "<u2", (MOTION_GRID, MOTION_GRID, CELL_SOURCE)),
        Column("cell_motion", "<u2", (MOTION_GRID, MOTION_GRID, CELL_MOTION)),
        Column("patch", "|u1", PATCH), Column("residual", "<i2", PATCH),
        Column("tool_cost", "<f4", (TOOLS,)), Column("tool_ns", "<u8", (TOOLS,)),
        Column("tool_positions", "<u4", (TOOLS,)),
    ),
    "node": (
        Column("frame", "<u4"), Column("sb", "<u4"), Column("parent", "<i4"), Column("x", "<u4"),
        Column("y", "<u4"), Column("log2", "|u1"), Column("planned", "|u1"),
        Column("chosen", "|u1"), Column("eligible", "|u1"), Column("plan_margin", "<f4"),
        Column("plan_costs", "<f4", PAIR), Column("gates", "|u1"), Column("context", "|i1", PAIR),
        Column("final_context", "|i1", PAIR), Column("has_inputs", "|u1"),
        Column("inputs", "|u1", (TRIAL_INPUTS,)), Column("search", "<f4", QUAD),
        Column("pre", "<f4", (5,)), Column("first", "<f4", (3,)),
        Column("cost", "<f4", SLOT_ROW), Column("ns", "<u8", SLOT_ROW),
        Column("distortion", "<f4", SLOT_ROW), Column("bits", "<f4", SLOT_ROW),
        Column("cand_positions", "<u4", SLOT_ROW), Column("nonzero", "<u2", SLOT_ROW),
        Column("tx_blocks", "<u2", SLOT_ROW), Column("max_tx", "|u1", SLOT_ROW),
    ),
    "leaf": (
        Column("frame", "<u4"), Column("sb", "<u4"), Column("x", "<u4"), Column("y", "<u4"),
        Column("log2", "|u1", PAIR), Column("mode", "|u1"), Column("skip", "|u1"),
        Column("mv", "<i2", PAIR), Column("qp_offset", "<i2"), Column("positions", "<u2"),
    ),
}


#: Node trial summaries and search positions (format 6): absent from earlier formats.
V6_ADDED = {"frame": set(), "sb": {"tool_positions"},
            "node": {"cand_positions", "nonzero", "tx_blocks", "max_tx"}, "leaf": {"positions"}}
#: The second reference's residuals (format 7).
V7_ADDED = {"frame": set(), "sb": {"second_residual"}, "node": set(), "leaf": set()}
#: Per earlier format, the columns it does not have. Format 4 (v0) shards also have no
#: labels.json column list.
ABSENT = {
    6: V7_ADDED,
    5: {t: V6_ADDED[t] | V7_ADDED[t] for t in V6_ADDED},
    4: {"frame": {"key_distance", "rate"},
        "sb": {"block_sad", "block_mv", "cell_source", "cell_motion"} | V6_ADDED["sb"]
        | V7_ADDED["sb"],
        "node": {"plan_costs", "final_context", "search", "pre", "first", "distortion",
                 "bits"} | V6_ADDED["node"],
        "leaf": {c.name for c in COLUMNS["leaf"]}},
}
#: The formats this reader reads.
READABLE = (4, 5, 6, FORMAT_VERSION)


def required(version: int) -> dict[str, tuple[Column, ...]]:
    """The columns a shard of format `version` must have."""
    if version == FORMAT_VERSION:
        return COLUMNS
    absent = ABSENT[version]
    return {t: tuple(c for c in cols if c.name not in absent[t]) for t, cols in COLUMNS.items()
            if any(c.name not in absent[t] for c in cols)}
