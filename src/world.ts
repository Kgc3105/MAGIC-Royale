/**
 * Static map model: terrain + building/stadium/bridge solids rasterised onto the 1 m terrain grid.
 * Gives ground height, line-of-sight / shot-path tests, and a wall-clearance field for navigation.
 */
import { readFileSync } from "node:fs"

export type Prism = { kind: string; base: number; top: number; ring: number[]; holes: number[][] }

const STEP_UP = 1.2
/** terrain steeper than this (rise per metre) is treated as impassable when planning (server limit is 50 deg) */
const MAX_PLAN_SLOPE = 0.9

export class World {
  nx: number
  ny: number
  minX: number
  minY: number
  /** playable rectangle (the map edge is a wall) */
  bMinX: number
  bMaxX: number
  bMinY: number
  bMaxY: number
  /** terrain height at each grid vertex, metres */
  ground: Float32Array
  /** top of whatever solid stands on the ground at this vertex (== ground where nothing does) */
  top: Float32Array
  /** underside / top of a solid that hangs above the ground here (bridges); Infinity / -Infinity if none */
  ohBase: Float32Array
  ohTop: Float32Array
  /** distance (m) from each vertex to the nearest thing a mech cannot walk through, for a normal-height mech */
  clearLow: Float32Array
  /** same, for a mech too tall to pass under any bridge */
  clearTall: Float32Array
  prisms: Prism[]

  constructor(dir: string, files: string[] = ["v2/world.json", "v2/terrain.i16"]) {
    const jsonFile = files.find((f) => f.endsWith("world.json")) ?? "v2/world.json"
    const terrFile = files.find((f) => f.endsWith(".i16")) ?? "v2/terrain.i16"
    const w = JSON.parse(readFileSync(`${dir}/${jsonFile}`, "utf8"))
    const T = w.terrain
    this.nx = T.nx
    this.ny = T.ny
    this.minX = T.minX
    this.minY = T.minY
    const step = T.step ?? 1
    if (step !== 1) throw new Error(`terrain step ${step} not supported`)
    const scale = T.scale ?? 0.01
    const buf = readFileSync(`${dir}/${terrFile}`)
    const n = this.nx * this.ny
    if (buf.length < n * 2) throw new Error("terrain file too short")
    this.ground = new Float32Array(n)
    for (let i = 0; i < n; i++) this.ground[i] = buf.readInt16LE(i * 2) * scale
    const b = w.bounds ?? { minX: this.minX, maxX: this.minX + this.nx - 1, minY: this.minY, maxY: this.minY + this.ny - 1 }
    this.bMinX = b.minX
    this.bMaxX = b.maxX
    this.bMinY = b.minY
    this.bMaxY = b.maxY

    this.top = new Float32Array(this.ground)
    this.ohBase = new Float32Array(n).fill(Infinity)
    this.ohTop = new Float32Array(n).fill(-Infinity)
    this.prisms = (w.prisms ?? []) as Prism[]
    for (const p of this.prisms) {
      this.fill([p.ring, ...(p.holes ?? [])], (idx) => {
        const g = this.ground[idx]
        if (p.base > g + 3) {
          // hangs above the ground: you can see, shoot and maybe walk under it
          if (p.base < this.ohBase[idx]) this.ohBase[idx] = p.base
          if (p.top > this.ohTop[idx]) this.ohTop[idx] = p.top
        } else if (p.top > this.top[idx]) this.top[idx] = p.top
      })
    }
    this.clearLow = this.clearance(9.4)
    this.clearTall = this.clearance(1e9)
  }

  /** scanline fill (even-odd, so holes are honoured) over grid vertices */
  private fill(rings: number[][], cb: (idx: number) => void) {
    const { nx, ny, minX, minY } = this
    let y0 = Infinity
    let y1 = -Infinity
    const outer = rings[0]
    for (let k = 1; k < outer.length; k += 2) {
      if (outer[k] < y0) y0 = outer[k]
      if (outer[k] > y1) y1 = outer[k]
    }
    const j0 = Math.max(0, Math.ceil(y0 - minY))
    const j1 = Math.min(ny - 1, Math.floor(y1 - minY))
    const xs: number[] = []
    for (let j = j0; j <= j1; j++) {
      const y = minY + j + 1e-4
      xs.length = 0
      for (const r of rings) {
        const m = r.length >> 1
        for (let a = 0, c = m - 1; a < m; c = a++) {
          const ya = r[2 * a + 1]
          const yc = r[2 * c + 1]
          if (ya > y !== yc > y) xs.push(r[2 * a] + ((y - ya) / (yc - ya)) * (r[2 * c] - r[2 * a]))
        }
      }
      xs.sort((p, q) => p - q)
      for (let k = 0; k + 1 < xs.length; k += 2) {
        const i0 = Math.max(0, Math.ceil(xs[k] - minX))
        const i1 = Math.min(nx - 1, Math.floor(xs[k + 1] - minX))
        for (let i = i0; i <= i1; i++) cb(j * nx + i)
      }
    }
  }

  /** chamfer distance (m) to the nearest vertex a mech of this height cannot stand on / pass */
  private clearance(mechHeight: number): Float32Array {
    const { nx, ny, minX, minY, ground, top, ohBase } = this
    const n = nx * ny
    const d = new Float32Array(n)
    const INF = 1e6
    for (let j = 0; j < ny; j++) {
      const y = minY + j
      for (let i = 0; i < nx; i++) {
        const x = minX + i
        const idx = j * nx + i
        const g = ground[idx]
        let blocked = x < this.bMinX + 0.5 || x > this.bMaxX - 0.5 || y < this.bMinY + 0.5 || y > this.bMaxY - 0.5
        if (!blocked && top[idx] - g > STEP_UP) blocked = true
        if (!blocked && ohBase[idx] - g < mechHeight + 0.5) blocked = true
        if (!blocked && i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
          const s = Math.max(Math.abs(ground[idx - 1] - g), Math.abs(ground[idx + 1] - g), Math.abs(ground[idx - nx] - g), Math.abs(ground[idx + nx] - g))
          if (s > MAX_PLAN_SLOPE) blocked = true
        }
        d[idx] = blocked ? 0 : INF
      }
    }
    const D1 = 1
    const D2 = Math.SQRT2
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const idx = j * nx + i
        let v = d[idx]
        if (v === 0) continue
        if (i > 0 && d[idx - 1] + D1 < v) v = d[idx - 1] + D1
        if (j > 0) {
          if (d[idx - nx] + D1 < v) v = d[idx - nx] + D1
          if (i > 0 && d[idx - nx - 1] + D2 < v) v = d[idx - nx - 1] + D2
          if (i < nx - 1 && d[idx - nx + 1] + D2 < v) v = d[idx - nx + 1] + D2
        }
        d[idx] = v
      }
    }
    for (let j = ny - 1; j >= 0; j--) {
      for (let i = nx - 1; i >= 0; i--) {
        const idx = j * nx + i
        let v = d[idx]
        if (v === 0) continue
        if (i < nx - 1 && d[idx + 1] + D1 < v) v = d[idx + 1] + D1
        if (j < ny - 1) {
          if (d[idx + nx] + D1 < v) v = d[idx + nx] + D1
          if (i < nx - 1 && d[idx + nx + 1] + D2 < v) v = d[idx + nx + 1] + D2
          if (i > 0 && d[idx + nx - 1] + D2 < v) v = d[idx + nx - 1] + D2
        }
        d[idx] = v
      }
    }
    return d
  }

  /** nearest-vertex index, clamped to the grid */
  index(x: number, y: number): number {
    let i = Math.round(x - this.minX)
    let j = Math.round(y - this.minY)
    if (i < 0) i = 0
    else if (i >= this.nx) i = this.nx - 1
    if (j < 0) j = 0
    else if (j >= this.ny) j = this.ny - 1
    return j * this.nx + i
  }

  /** terrain height, bilinear */
  groundAt(x: number, y: number): number {
    let fx = x - this.minX
    let fy = y - this.minY
    if (fx < 0) fx = 0
    else if (fx > this.nx - 1.001) fx = this.nx - 1.001
    if (fy < 0) fy = 0
    else if (fy > this.ny - 1.001) fy = this.ny - 1.001
    const i = Math.floor(fx)
    const j = Math.floor(fy)
    const u = fx - i
    const v = fy - j
    const k = j * this.nx + i
    const g = this.ground
    return (g[k] * (1 - u) + g[k + 1] * u) * (1 - v) + (g[k + this.nx] * (1 - u) + g[k + this.nx + 1] * u) * v
  }

  /** height of the solid surface at a point: roof if there is a building, else terrain */
  topAt(x: number, y: number): number {
    return this.top[this.index(x, y)]
  }

  /**
   * True if nothing solid lies on the straight segment (buildings, terrain, bridge decks).
   * Trees are ignored: they do not stop shots. The metre or two next to each end is skipped so an
   * endpoint standing against a wall is not counted as blocked.
   */
  losClear(x0: number, y0: number, z0: number, x1: number, y1: number, z1: number): boolean {
    const dx = x1 - x0
    const dy = y1 - y0
    const d = Math.hypot(dx, dy)
    if (d < 4) return true
    const n = Math.ceil(d)
    const sx = dx / n
    const sy = dy / n
    const sz = (z1 - z0) / n
    const { nx, ny, top, ohBase, ohTop } = this
    let x = x0 + sx * 2 - this.minX + 0.5
    let y = y0 + sy * 2 - this.minY + 0.5
    let z = z0 + sz * 2
    for (let k = 2; k <= n - 2; k++) {
      const i = x | 0
      const j = y | 0
      if (i >= 0 && i < nx && j >= 0 && j < ny) {
        const idx = j * nx + i
        if (z < top[idx]) return false
        if (z > ohBase[idx] && z < ohTop[idx]) return false
      }
      x += sx
      y += sy
      z += sz
    }
    return true
  }
}
