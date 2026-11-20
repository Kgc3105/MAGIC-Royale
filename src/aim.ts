/** Aiming maths: angles, intercept for straight projectiles, ballistic pitch for the mortar. */

export const TAU = Math.PI * 2

/** wrap an angle to (-pi, pi] */
export function wrap(a: number): number {
  if (!Number.isFinite(a)) return 0
  a = a % TAU
  if (a > Math.PI) a -= TAU
  else if (a <= -Math.PI) a += TAU
  return a
}

export const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v)

/**
 * Time for a straight projectile of speed `s`, fired now from the origin, to meet a target at
 * relative position d moving at constant velocity v. Returns -1 if it never gets there.
 */
export function interceptTime(dx: number, dy: number, dz: number, vx: number, vy: number, vz: number, s: number): number {
  const a = vx * vx + vy * vy + vz * vz - s * s
  const b = 2 * (dx * vx + dy * vy + dz * vz)
  const c = dx * dx + dy * dy + dz * dz
  if (Math.abs(a) < 1e-9) {
    if (Math.abs(b) < 1e-9) return -1
    const t = -c / b
    return t > 0 ? t : -1
  }
  const disc = b * b - 4 * a * c
  if (disc < 0) return -1
  const sq = Math.sqrt(disc)
  const t1 = (-b - sq) / (2 * a)
  const t2 = (-b + sq) / (2 * a)
  const lo = Math.min(t1, t2)
  const hi = Math.max(t1, t2)
  if (lo > 1e-6) return lo
  if (hi > 1e-6) return hi
  return -1
}

/**
 * Launch pitch for a shell of speed v under gravity g to land on a point d metres away
 * horizontally and dz metres up. `high` picks the lobbed arc. Returns null if out of reach.
 */
export function ballisticPitch(d: number, dz: number, v: number, g: number, high: boolean): number | null {
  if (d < 1e-3) return null
  const v2 = v * v
  const disc = v2 * v2 - g * (g * d * d + 2 * dz * v2)
  if (disc < 0) return null
  const sq = Math.sqrt(disc)
  return Math.atan2(high ? v2 + sq : v2 - sq, g * d)
}

/** deterministic RNG so a match replays the same way for the same seed */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
