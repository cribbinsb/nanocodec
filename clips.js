// Hosted sample clips: openly licensed excerpts shipped in web/clips/.
// Paths are relative so the demo works from any static host (e.g. GitHub Pages).
// Full attribution, segment times and changes: web/clips/ATTRIBUTION.md.
const BLENDER_CREDIT = '© Blender Foundation · CC BY 3.0';
const SVT_CREDIT = '© SVT (Sveriges Television) · CC BY 4.0';
const SVT_SUITE = 'SVT Natural Complexity';

// Every clip ships at these heights as `clips/<id>-<height>p.mp4`: the demo's
// smaller resolutions read the light 360p file and only 720p fetches the large one.
const VARIANT_HEIGHTS = [360, 720];

export const CLIPS = [
  { id: 'tos-canal', title: 'Canal walk · motion', film: 'Tears of Steel', credit: BLENDER_CREDIT },
  { id: 'tos-bridge', title: 'Bridge talk · static', film: 'Tears of Steel', credit: BLENDER_CREDIT },
  { id: 'bbb-tree', title: 'Tree · fine texture', film: 'Big Buck Bunny', credit: BLENDER_CREDIT },
  { id: 'svt-waterfall', title: 'Waterfall · fast detail', film: SVT_SUITE, credit: SVT_CREDIT },
  { id: 'svt-midnight-sun', title: 'Midnight sun · wind and glitter', film: SVT_SUITE, credit: SVT_CREDIT },
  { id: 'svt-smoke-sauna', title: 'Smoke sauna · smoke and fire', film: SVT_SUITE, credit: SVT_CREDIT },
];

export const DEFAULT_CLIP = CLIPS[0];

export function clipById(id) {
  return CLIPS.find(clip => clip.id === id) ?? DEFAULT_CLIP;
}

// The smallest variant at least `height` tall (the largest when none is), so the
// canvas only ever scales the clip down.
export function clipUrl(clip, height) {
  const variant = VARIANT_HEIGHTS.find(h => h >= height) ?? VARIANT_HEIGHTS.at(-1);
  return `clips/${clip.id}-${variant}p.mp4`;
}
