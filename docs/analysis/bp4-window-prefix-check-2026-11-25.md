# Window prefix check: is a 150-frame window the first 150 frames of a 600-frame encode? (2026-11-25)

BP-4.0 codes each clip once, at its longest window, and reads the 150-frame window from the first
150 frames of that encode (procedure D7). This holds only if no encoder uses the declared frame
count to code its first frames. Checked here.

**Method:**
- **Input:** a tuning-pool clip (`long_static_chirla_s024c2`, not a sealed set), 600 frames
  from its source scaled to 320x240, 30 fps, and its first 150 frames as a separate file.
- **Encodes:** each encoder coded both with the record's CBR command
  (`nceval.fairness.command`, 150 kbps, a key every 150 frames), once told 150 frames and once
  told 600.
- **Builds:** the x86_64 anchors (`anchors/x86_64-opt-bp3`) and ours at 889309c99.
- **Compared:** the first 150 packets' sizes (`ffprobe`) and the SHA-256 of the first 150
  decoded frames.

| Family | First 150 packets identical | First 150 decoded frames identical |
|---|---|---|
| x264-medium | yes | yes |
| x265-medium | yes | yes |
| openh264-baseline | yes | yes |
| svtav1-m12 | yes | yes |
| libvpx-vp9-cpu9 | yes | yes |
| libaom-av1-cpu8 (aomenc) | yes | yes |
| libaom-av1-cpu8+drop (svc_encoder_rtc) | yes | yes |
| kvazaar-veryfast | yes | yes |
| nanocodec-av1@medium | yes | yes |
| nanocodec-h264@medium | yes | yes |
| nanocodec-hevc@medium | yes | yes |

**Result:** every encoder's first 150 frames are byte-identical whatever frame count it is told.
The 150-frame window read from a 600-frame encode is the 150-frame encode, so no clip is coded
twice.

**Not covered:** one clip and one rate, x86 only. The planner and report record the windows each
job serves, so a later check on the run's own streams can confirm it per encode.
