// Deterministic motion and fine detail, useful without granting camera access.
export function drawScene(ctx, width, height, seconds) {
  ctx.fillStyle = '#102b2c'; ctx.fillRect(0, 0, width, height);
  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, '#183c3c'); gradient.addColorStop(.58, '#6a8975'); gradient.addColorStop(1, '#d5d5b3');
  ctx.fillStyle = gradient; ctx.fillRect(0, 0, width, height);
  ctx.save(); ctx.translate(width * .57, height * .5);
  ctx.rotate(seconds * .12);
  for (let i = 0; i < 42; i++) {
    ctx.strokeStyle = i % 2 ? '#e3dbb994' : '#183f3866';
    ctx.lineWidth = Math.max(.6, width / 900);
    ctx.strokeRect(-width * .27 + i * 2.5, -height * .3 + i * 1.3, width * .54 - i * 5, height * .6 - i * 2.6);
  }
  ctx.restore();
  const x = width * (.5 + .3 * Math.sin(seconds * .7));
  const y = height * (.48 + .18 * Math.cos(seconds * .9));
  ctx.fillStyle = '#f2b566'; ctx.beginPath(); ctx.arc(x, y, height * .14, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = '#faf0c8'; ctx.lineWidth = Math.max(1, width / 500);
  for (let r = 1; r < 8; r++) { ctx.beginPath(); ctx.arc(x, y, height * (.14 + r * .009), 0, Math.PI * 2); ctx.stroke(); }
  ctx.fillStyle = '#0e2a2b'; ctx.fillRect(width * .07, height * .16, width * .23, height * .67);
  for (let row = 0; row < 20; row++) for (let col = 0; col < 20; col++) {
    ctx.fillStyle = (col + row) % 2 ? '#6c998a' : '#bdd2b3';
    ctx.fillRect(width * .09 + col * width * .009, height * .24 + row * height * .019, Math.max(1, width * .004), Math.max(1, height * .009));
  }
  ctx.fillStyle = '#e7e9d4'; ctx.font = `${Math.max(10, width / 45)}px monospace`;
  ctx.fillText('NANO / MOTION STUDY', width * .07, height * .12);
  ctx.font = `${Math.max(9, width / 55)}px monospace`;
  ctx.fillText(`${seconds.toFixed(2).padStart(7, '0')} s`, width * .72, height * .9);
}
