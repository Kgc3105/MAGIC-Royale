/** Ground navigation: A* over a 2 m grid derived from the world's wall-clearance field. */
import type { World } from "./world.ts"

const CELL = 2

export class Nav {
  w: World
  gnx: number
  gny: number
  /** active clearance field (switches when the mech grows too tall for the bridges) */
  clear: Float32Array
  /** clearance a cell needs for our mech to stand there (radius + margin) */
  need = 3.0
  /** match time, used to expire temporary blocks */
  now = 0
  lastExpanded = 0
  lastPartial = false

  private g: Float32Array
  private par: Int32Array
  private stamp: Uint32Array
  private done: Uint32Array
  private gen = 0
  private hNode: Int32Array
  private hKey: Float32Array
  private hn = 0
  /** per nav cell: match time until which it is treated as blocked (learned from getting stuck) */
  private tmp: Float32Array

  constructor(w: World) {
    this.w = w
    this.gnx = Math.floor((w.nx - 1) / CELL) + 1
    this.gny = Math.floor((w.ny - 1) / CELL) + 1
    const n = this.gnx * this.gny
    this.clear = w.clearLow
    this.g = new Float32Array(n)
    this.par = new Int32Array(n)
    this.stamp = new Uint32Array(n)
    this.done = new Uint32Array(n)
    this.tmp = new Float32Array(n).fill(-1)
    const cap = 1 << 20
    this.hNode = new Int32Array(cap)
    this.hKey = new Float32Array(cap)
  }

  cellOf(x: number, y: number): number {
    let ci = Math.round((x - this.w.minX) / CELL)
    let cj = Math.round((y - this.w.minY) / CELL)
    if (ci < 0) ci = 0
    else if (ci >= this.gnx) ci = this.gnx - 1
    if (cj < 0) cj = 0
    else if (cj >= this.gny) cj = this.gny - 1
    return cj * this.gnx + ci
  }
  cellX(c: number): number {
    return this.w.minX + (c % this.gnx) * CELL
  }
  cellY(c: number): number {
    return this.w.minY + Math.floor(c / this.gnx) * CELL
  }
  private cellClear(c: number): number {
    const ci = c % this.gnx
    const cj = (c - ci) / this.gnx
    return this.clear[cj * CELL * this.w.nx + ci * CELL]
  }
  walkable(c: number): boolean {
    return this.tmp[c] <= this.now && this.cellClear(c) >= this.need
  }
  /** wall clearance (m) at a world point */
  clearAt(x: number, y: number): number {
    return this.clear[this.w.index(x, y)]
  }
  /** can our mech stand at this world point? */
  free(x: number, y: number, slack = 0): boolean {
    return this.clear[this.w.index(x, y)] >= this.need - slack && this.tmp[this.cellOf(x, y)] <= this.now
  }

  /** straight segment walkable? sampled every metre */
  segOk(x0: number, y0: number, x1: number, y1: number, slack = 0.35): boolean {
    const dx = x1 - x0
    const dy = y1 - y0
    const d = Math.hypot(dx, dy)
    const n = Math.max(1, Math.ceil(d))
    const lim = this.need - slack
    for (let k = 1; k <= n; k++) {
      const x = x0 + (dx * k) / n
      const y = y0 + (dy * k) / n
      if (this.clear[this.w.index(x, y)] < lim) return false
      if (this.tmp[this.cellOf(x, y)] > this.now) return false
    }
    return true
  }

  /** mark a disc as blocked until `until` (used when the real world disagrees with the map model) */
  block(x: number, y: number, r: number, until: number) {
    const c0 = this.cellOf(x, y)
    const ci0 = c0 % this.gnx
    const cj0 = (c0 - ci0) / this.gnx
    const rr = Math.ceil(r / CELL)
    for (let dj = -rr; dj <= rr; dj++) {
      for (let di = -rr; di <= rr; di++) {
        const ci = ci0 + di
        const cj = cj0 + dj
        if (ci < 0 || cj < 0 || ci >= this.gnx || cj >= this.gny) continue
        if (di * di + dj * dj > rr * rr) continue
        const c = cj * this.gnx + ci
        if (this.tmp[c] < until) this.tmp[c] = until
      }
    }
  }

  /** forget every temporary block */
  clearBlocks() {
    this.tmp.fill(-1)
  }

  /** nearest walkable cell to a world point within maxRing cells, or -1 */
  nearestWalkable(x: number, y: number, maxRing: number): number {
    const c0 = this.cellOf(x, y)
    if (this.walkable(c0)) return c0
    const ci0 = c0 % this.gnx
    const cj0 = (c0 - ci0) / this.gnx
    for (let r = 1; r <= maxRing; r++) {
      let best = -1
      let bd = Infinity
      for (let dj = -r; dj <= r; dj++) {
        const cj = cj0 + dj
        if (cj < 0 || cj >= this.gny) continue
        const edge = dj === -r || dj === r
        for (let di = -r; di <= r; di += edge ? 1 : 2 * r) {
          const ci = ci0 + di
          if (ci < 0 || ci >= this.gnx) continue
          const c = cj * this.gnx + ci
          if (!this.walkable(c)) continue
          const ddx = this.w.minX + ci * CELL - x
          const ddy = this.w.minY + cj * CELL - y
          const dd = ddx * ddx + ddy * ddy
          if (dd < bd) {
            bd = dd
            best = c
          }
        }
      }
      if (best >= 0) return best
    }
    return -1
  }

  private push(node: number, key: number) {
    if (this.hn >= this.hNode.length) return
    let i = this.hn++
    const hN = this.hNode
    const hK = this.hKey
    while (i > 0) {
      const p = (i - 1) >> 1
      if (hK[p] <= key) break
      hN[i] = hN[p]
      hK[i] = hK[p]
      i = p
    }
    hN[i] = node
    hK[i] = key
  }
  private pop(): number {
    const hN = this.hNode
    const hK = this.hKey
    const topNode = hN[0]
    const n = --this.hn
    if (n > 0) {
      const node = hN[n]
      const key = hK[n]
      let i = 0
      for (;;) {
        let c = 2 * i + 1
        if (c >= n) break
        if (c + 1 < n && hK[c + 1] < hK[c]) c++
        if (hK[c] >= key) break
        hN[i] = hN[c]
        hK[i] = hK[c]
        i = c
      }
      hN[i] = node
      hK[i] = key
    }
    return topNode
  }

  /**
   * A* from (sx,sy) to (gx,gy). Returns world points [x0,y0,x1,y1,...] (empty if nothing usable).
   * If the search runs out of budget it returns the path to the explored cell nearest the goal
   * and sets lastPartial.
   */
  findPath(sx: number, sy: number, gx: number, gy: number, maxExpand = 45000): number[] {
    this.lastExpanded = 0
    this.lastPartial = false
    const s = this.nearestWalkable(sx, sy, 10)
    const e = this.nearestWalkable(gx, gy, 14)
    if (s < 0 || e < 0) return []
    const gnx = this.gnx
    const gny = this.gny
    const ei = e % gnx
    const ej = (e - ei) / gnx
    const gen = ++this.gen
    const { g, par, stamp, done } = this
    const wnx = this.w.nx
    const clear = this.clear
    const need = this.need
    const soft = need + 3
    const now = this.now
    const tmp = this.tmp
    this.hn = 0
    g[s] = 0
    par[s] = -1
    stamp[s] = gen
    this.push(s, 0)
    let best = s
    let bestH = Infinity
    let expanded = 0
    let found = false
    while (this.hn > 0) {
      const cur = this.pop()
      if (done[cur] === gen) continue
      done[cur] = gen
      if (cur === e) {
        found = true
        break
      }
      if (++expanded > maxExpand) break
      const ci = cur % gnx
      const cj = (cur - ci) / gnx
      const gc = g[cur]
      for (let dj = -1; dj <= 1; dj++) {
        const nj = cj + dj
        if (nj < 0 || nj >= gny) continue
        for (let di = -1; di <= 1; di++) {
          if (di === 0 && dj === 0) continue
          const ni = ci + di
          if (ni < 0 || ni >= gnx) continue
          const nb = nj * gnx + ni
          if (done[nb] === gen) continue
          const c = clear[nj * CELL * wnx + ni * CELL]
          if (c < need || tmp[nb] > now) continue
          if (di !== 0 && dj !== 0) {
            // no corner cutting
            const a = cj * gnx + ni
            const b = nj * gnx + ci
            if (clear[cj * CELL * wnx + ni * CELL] < need || clear[nj * CELL * wnx + ci * CELL] < need || tmp[a] > now || tmp[b] > now) continue
          }
          const len = di !== 0 && dj !== 0 ? 2.8284 : 2
          const pen = c < soft ? (soft - c) * 0.12 : 0
          const ng = gc + len * (1 + pen)
          if (stamp[nb] === gen && ng >= g[nb]) continue
          g[nb] = ng
          par[nb] = cur
          stamp[nb] = gen
          const ax = ni > ei ? ni - ei : ei - ni
          const ay = nj > ej ? nj - ej : ej - nj
          const h = 2 * (ax > ay ? ax + 0.4142 * ay : ay + 0.4142 * ax)
          if (h < bestH) {
            bestH = h
            best = nb
          }
          this.push(nb, ng + h * 1.15)
        }
      }
    }
    this.lastExpanded = expanded
    let end = e
    if (!found) {
      this.lastPartial = true
      end = best
      if (end === s) return []
    }
    const cells: number[] = []
    for (let c = end; c >= 0; c = par[c]) {
      cells.push(c)
      if (cells.length > 20000) break
    }
    const out: number[] = []
    for (let k = cells.length - 1; k >= 0; k--) out.push(this.cellX(cells[k]), this.cellY(cells[k]))
    if (found) out.push(gx, gy)
    return out
  }
}
