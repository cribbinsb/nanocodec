"""nceval.content_id: the disjointness rule catches the old PGO / AOM-CTC A2 collision, and
the PGO training set of 2026-11-23 shares nothing with any dataset."""
import json
from pathlib import Path

from nceval.content_id import (collection, collection_overlaps, content_id, overlaps, shoot,
                               source_overlaps)

#: The PGO training set until 2026-11-23 (Xiph derf).
PGO = ["720p50_parkrun_ter", "old_town_cross_420_720p50", "in_to_tree_420_720p50",
       "ducks_take_off_420_720p50"]


def test_same_content_under_two_names():
    assert content_id("OldTownCross_1920x1080p50.y4m") == content_id("old_town_cross_420_720p50")
    assert content_id("720p50_parkrun_ter") == "park_run"


def test_pgo_training_overlaps_aom_ctc_a2():
    a2 = ["OldTownCross_1920x1080p50.y4m", "CrowdRun_1920x1080p50.y4m",
          "Boat_1920x1080_5994_10bit_420.y4m"]
    found = overlaps(PGO, a2)
    assert ("old_town_cross_420_720p50", "OldTownCross_1920x1080p50.y4m") in found
    # CrowdRun is another content of the same shoot as the training clips
    assert any(y.startswith("CrowdRun") for _, y in found)
    assert not any(y.startswith("Boat") for _, y in found)


def test_unrelated_names_do_not_collide():
    assert shoot("Johnny_1280x720_60") != shoot("KristenAndSara_1280x720_60")
    assert overlaps(["ho_caltech_set02_v003"], ["ho_caltech_set07_v008"]) == []


def test_parkrun_shares_a_shoot_with_ctc_stockholm():
    found = overlaps(["720p50_parkrun_ter"], ["Stockholm_640x360_5994", "Johnny_1280x720_60"])
    assert found == [("720p50_parkrun_ter", "Stockholm_640x360_5994")]


def test_same_source_file_under_two_names():
    a = [{"name": "ho_x", "source": "/mldata/video/a.mp4"}]
    b = [{"name": "pool_y", "source": "/mldata/video/a.mp4"}, {"name": "z", "source": "/b.mp4"}]
    assert source_overlaps(a, b) == [("ho_x", "pool_y")]


# --- the PGO training set (BP-3.1 task T5) against every dataset --------------------------

REPO = Path(__file__).resolve().parents[2]
#: AOM CTC class A2 (media.xiph.org/video/aomctc/test_set/a2_2k/, listed 2026-11-23).
AOM_CTC_A2 = ["Aerial3200_1920x1080_5994_10bit_420", "Boat_1920x1080_5994_10bit_420",
              "CrowdRun_1920x1080p50", "DinnerSceneCropped_1920x1080_2997fps_10bit_420",
              "FoodMarket_1920x1080_5994_10bit_420", "MeridianTalk_sdr_1920x1080p_5994_10bit",
              "Motorcycle_1920x1080_30fps_8bit", "MountainBike_1920x1080_30fps_8bit",
              "OldTownCross_1920x1080p50", "PedestrianArea_1920x1080p25",
              "RitualDance_1920x1080_5994_10bit_420", "Riverbed_1920x1080p25",
              "RushFieldCuts_1920x1080_2997", "Skater227_1920x1080_30fps",
              "ToddlerFountain_1920x1080_2997fps_10bit_420", "TreesAndGrass_1920_1080_30fps_8bit",
              "TunnelFlag_1920x1080_5994_10bit_420", "Vertical_Carnaby_1080x1920_5994",
              "Vertical_bees_1080x1920_2997", "WalkingInStreet_1920x1080_30fps",
              "WorldCup_1920x1080_30p", "WorldCup_far_1920x1080_30p"]


def _clips(relative: str) -> list[dict]:
    return json.loads((REPO / relative).read_text())["clips"]


def test_pgo_training_set_is_disjoint_from_every_dataset():
    training = _clips("eval/clipset/pgo_training_2026-11-23.json")
    # the benchmark's datasets and the tuning pool and sets
    datasets = [clip for manifest in ("heldout_2026-11-23b.manifest", "vc1k", "pool", "main",
                                      "independent", "quick", "long")
                for clip in _clips(f"eval/clipset/{manifest}.json")]
    ctc = json.loads((REPO / "docs/results/standards-ctc-2026-11-14/sources.json").read_text())
    names = [c["name"] for c in datasets] + [c["name"] for c in ctc] + AOM_CTC_A2
    assert overlaps([c["name"] for c in training], names) == []
    assert source_overlaps(training, datasets) == []
    assert collection_overlaps(training, datasets) == []
    # the old training set fails the same check (A2's OldTownCross and CrowdRun)
    assert overlaps(PGO, names)


def test_collection_of_a_source():
    assert collection("/mldata/downloaded_datasets/other/nasa_videos/KSC-x~orig.mp4") == \
        "/mldata/downloaded_datasets/other/nasa_videos"
    assert collection("/mldata/downloaded_datasets/other/vc1k/a.mp4") != \
        collection("/mldata/downloaded_datasets/other/vcd2/a.mp4")
