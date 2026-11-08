// Hosted sample clips: openly licensed film excerpts shipped in web/clips/.
// Paths are relative so the demo works from any static host (e.g. GitHub Pages).
// Full attribution, segment times and changes: web/clips/ATTRIBUTION.md.
const BLENDER_CREDIT = '© Blender Foundation · CC BY 3.0';

export const CLIPS = [
  { id: 'tos-canal', title: 'Canal walk · motion', film: 'Tears of Steel', credit: BLENDER_CREDIT, url: 'clips/tos-canal.mp4' },
  { id: 'tos-bridge', title: 'Bridge talk · static', film: 'Tears of Steel', credit: BLENDER_CREDIT, url: 'clips/tos-bridge.mp4' },
  { id: 'bbb-tree', title: 'Tree · fine texture', film: 'Big Buck Bunny', credit: BLENDER_CREDIT, url: 'clips/bbb-tree.mp4' },
];

export const DEFAULT_CLIP = CLIPS[0];

export function clipById(id) {
  return CLIPS.find(clip => clip.id === id) ?? DEFAULT_CLIP;
}
