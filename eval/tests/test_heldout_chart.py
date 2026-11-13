"""nceval.heldout_chart: held-out BD-J joined with the main chart's x86 speed."""
from nceval.heldout_chart import heldout_points

HELDOUT = """family,label,ours,bd_j_heldout_pct,bd_j_chart_pct,delta_pts,delta_vs_median_pts,rank_heldout,rank_chart,rank_move,bd_psnr_heldout_pct,bd_vmaf_heldout_pct,clips_scored,clips,flag
nanocodec-hevc,HEVC,1,-41.73,-49.26,+7.53,-0.03,4,4,0,-43.22,-45.73,70,70,
svtav1-m13,SVT-AV1 M13,0,-18.24,-33.23,+14.99,+7.44,21,19,-2,-19.13,-17.18,70,70,FLAG
"""
X86 = """family,label,fps_x86,speed_vs_anchor
nanocodec-hevc,HEVC,126.9,1.88
"""


def test_points_take_heldout_bd_and_chart_speed(tmp_path):
    (tmp_path / "h.csv").write_text(HELDOUT)
    (tmp_path / "x.csv").write_text(X86)
    hevc, m13 = heldout_points(tmp_path / "h.csv", tmp_path / "x.csv")
    assert (hevc.ours, hevc.bd_j, hevc.fps, hevc.speed) == (True, -41.73, 126.9, 1.88)
    assert (m13.ours, m13.bd_j, m13.fps) == (False, -18.24, None)  # untimed: left off the chart
