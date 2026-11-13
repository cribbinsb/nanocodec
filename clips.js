// Hosted sample clips: openly licensed, public-domain or freely redistributable excerpts shipped
// in web/clips/. Paths are relative so the demo works from any static host (e.g.
// GitHub Pages). Full attribution, segment times and changes: web/clips/ATTRIBUTION.md.
// A clip file is fetched only when that clip is selected, so the list can grow
// without slowing page open; only DEFAULT_CLIP loads at start.
const BLENDER_CREDIT = '© Blender Foundation · CC BY 3.0';
const SVT_CREDIT = '© SVT (Sveriges Television) · CC BY 4.0';
const SVT_SUITE = 'SVT Natural Complexity';
const XIPH_CREDIT = 'Public domain · Xiph.org derf collection';
const CONFERENCING_SET = 'JCT-VC / Vidyo conferencing';
// The SVT 2006 Multi Format set's terms (SVT_MultiFormat_v10.pdf, §2) require its
// copyright and use restriction to travel with every excerpt; the credit carries the
// restriction and the tooltip the full notice, verbatim.
const SVT_MULTIFORMAT_CREDIT = '© SVT · only for developing, testing and presenting technology standards';
const SVT_MULTIFORMAT_NOTICE = 'Individuals and organizations extracting sequences from this archive agree that the '
  + 'sequences and all intellectual property rights therein remain the property of Sveriges Television AB (SVT), '
  + 'Sweden. These sequences may only be used for the purpose of developing, testing and presenting technology '
  + 'standards. SVT makes no warranties with respect to the materials and expressly disclaim any warranties '
  + 'regarding their fitness for any purpose.';
const SVT_MULTIFORMAT_SUITE = 'SVT HD Multi Format (2006)';
const NETFLIX_CREDIT = 'Copyright (C) NETFLIX INC. 2015 · CC BY-NC-ND 4.0';
// MOTChallenge sequences are CC BY-NC-SA 3.0: the excerpts (and the pre-encoded
// streams made from them) are offered under the same licence, non-commercially.
const MOT_SET = 'MOTChallenge';
const motCredit = (set, authors) => `MOTChallenge ${set} (${authors}) · CC BY-NC-SA 3.0`;
const motNotice = (set, authors) => `MOTChallenge ${set} (${authors}), CC BY-NC-SA 3.0, https://motchallenge.net. `
  + 'Changes: excerpt, scaled, retimed to 30 fps where the source is slower, re-encoded. This excerpt and the streams '
  + 'made from it are offered under the same CC BY-NC-SA 3.0 licence, for non-commercial use.';

// Picker groups, in display order.
export const CLIP_GROUPS = ['People', 'Motion', 'Hard texture', 'Animation'];

// Every clip ships at these heights as `clips/<id>-<height>p.mp4`: 384x216 and
// 640x360 read the light 360p file; 854x480 and 1280x720 read the 720p one, as a
// 720p downscale to 480p is sharper than a 360p upscale and the file is small
// (1.3 MB for the default clip).
const VARIANT_HEIGHTS = [360, 720];

// Each file's own frame rate (`fps`), which maps its frames onto the page's
// 30 Hz source timeline (timeline.js). Sources shot at 24 or 25 fps are retimed to
// 30 fps (every frame once, tools/web/retime-clips.py, tools/web/make-mot-clips.py),
// so they play faster than real time: `speed` says by how much.
const XIPH_FPS = 30, NETFLIX_FPS = 60, RETIMED_FPS = 30, SVT_SPEED = 30 / 25, BLENDER_SPEED = 30 / 24;
const xiph = (id, title) => ({ id, title, group: 'People', film: CONFERENCING_SET, credit: XIPH_CREDIT, fps: XIPH_FPS });
const netflix = (id, title, group, film) => ({ id, title, group, film: `Netflix ${film}`, credit: NETFLIX_CREDIT, fps: NETFLIX_FPS });
const svtMultiFormat = (id, title, group) =>
  ({ id, title, group, film: SVT_MULTIFORMAT_SUITE, credit: SVT_MULTIFORMAT_CREDIT, notice: SVT_MULTIFORMAT_NOTICE, fps: RETIMED_FPS, speed: SVT_SPEED });
const svt = (id, title) => ({ id, title, group: 'Hard texture', film: SVT_SUITE, credit: SVT_CREDIT, fps: RETIMED_FPS, speed: SVT_SPEED });
const blender = (id, title, group, film) => ({ id, title, group, film, credit: BLENDER_CREDIT, fps: RETIMED_FPS, speed: BLENDER_SPEED });
const mot = (id, title, group, set, authors, speed) =>
  ({ id, title, group, film: `${MOT_SET} ${set}`, credit: motCredit(set, authors), notice: motNotice(set, authors), fps: RETIMED_FPS, speed });
const MOT17_AUTHORS = 'Milan et al.', MOT20_AUTHORS = 'Dendorfer et al.', MOT20_SPEED = 30 / 25;

export const CLIPS = [
  xiph('xiph-johnny', 'Talking head'),
  xiph('xiph-kristen-sara', 'Video call · two people'),
  xiph('xiph-four-people', 'Meeting · four people'),
  xiph('xiph-vidyo1', 'Meeting table · three people'),
  xiph('xiph-vidyo4', 'Webcam · talking with hands'),
  xiph('xiph-vidyo3', 'Whiteboard · presenter walking'),
  netflix('netflix-narrator', 'Narrator · walking and talking', 'People', 'El Fuente'),
  netflix('netflix-dinner-scene', 'Dinner table · low light', 'People', 'Chimera'),
  netflix('netflix-bar-scene', 'Bar · shuffleboard', 'People', 'Chimera'),
  mot('mot17-night-crowd', 'Night street · crowd, static camera', 'People', 'MOT17-04', MOT17_AUTHORS),
  blender('tos-bridge', 'Bridge talk · static', 'People', 'Tears of Steel'),
  blender('tos-canal', 'Canal walk · motion', 'Motion', 'Tears of Steel'),
  mot('mot17-mall-walk', 'Mall walk · moving camera', 'Motion', 'MOT17-11', MOT17_AUTHORS),
  netflix('netflix-crosswalk', 'Crosswalk · street crowd', 'Motion', 'El Fuente'),
  netflix('netflix-food-market', 'Food market · handheld', 'Motion', 'El Fuente'),
  netflix('netflix-tango', 'Tango · dance hall crowd', 'Motion', 'El Fuente'),
  netflix('netflix-boxing-practice', 'Boxing practice · fast moves', 'Motion', 'El Fuente'),
  netflix('netflix-dancers', 'Dancers · dark stage', 'Motion', 'Chimera'),
  netflix('netflix-toddler-fountain', 'Toddler at a fountain · water', 'Motion', 'Chimera'),
  svtMultiFormat('svt-old-town-cross', 'Old town · slow pan', 'Motion'),
  svtMultiFormat('svt-in-to-tree', 'Into the tree · zoom', 'Motion'),
  svt('svt-waterfall', 'Waterfall · fast detail'),
  svt('svt-midnight-sun', 'Midnight sun · wind and glitter'),
  svt('svt-smoke-sauna', 'Smoke sauna · smoke and fire'),
  mot('mot20-dense-crowd', 'Dense crowd · ~200 people, dark', 'Hard texture', 'MOT20-05', MOT20_AUTHORS, MOT20_SPEED),
  svtMultiFormat('svt-crowd-run', 'Crowd run · many runners', 'Hard texture'),
  svtMultiFormat('svt-park-joy', 'Park joy · pan past trees', 'Hard texture'),
  svtMultiFormat('svt-ducks-take-off', 'Ducks take off · splashing water', 'Hard texture'),
  blender('bbb-tree', 'Tree · fine texture', 'Animation', 'Big Buck Bunny'),
];

// The demo is about live camera streaming, so it opens on people in a real scene.
export const DEFAULT_CLIP = CLIPS.find(clip => clip.id === 'netflix-bar-scene');

export function clipById(id) {
  return CLIPS.find(clip => clip.id === id) ?? DEFAULT_CLIP;
}

// The smallest variant at least `height` tall (the largest when none is), so the
// canvas only ever scales the clip down.
export function clipUrl(clip, height) {
  const variant = VARIANT_HEIGHTS.find(h => h >= height) ?? VARIANT_HEIGHTS.at(-1);
  return `clips/${clip.id}-${variant}p.mp4`;
}
