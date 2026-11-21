/** Diagnostic: list the walkable areas cut off from the main one, and the jet crossing planned out of each. */
import { readFileSync } from "node:fs"
import { gunzipSync } from "node:zlib"
import { Nav } from "../src/nav.ts"
import { World } from "../src/world.ts"
const KIT = decodeURIComponent(new URL("../mechfight-kit", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1"))
const hello = JSON.parse(gunzipSync(readFileSync(`${KIT}/check/stream.jsonl.gz`)).toString("utf8").split("\n")[0])
const t0 = performance.now()
const w = new World(`${KIT}/arena`, hello.map.files)
const t1 = performance.now()
const nav = new Nav(w)
console.log(`world ${(t1 - t0).toFixed(0)} ms, nav ${(performance.now() - t1).toFixed(0)} ms, main area id ${nav.mainComp}`)
const n = nav.gnx * nav.gny
const size = new Map<number, number>()
const sx = new Map<number, number>()
const sy = new Map<number, number>()
for (let c = 0; c < n; c++) {
  const id = nav.comp[c]
  if (id < 0) continue
  size.set(id, (size.get(id) ?? 0) + 1)
  sx.set(id, (sx.get(id) ?? 0) + nav.cellX(c))
  sy.set(id, (sy.get(id) ?? 0) + nav.cellY(c))
}
const ids = [...size.keys()].filter((i) => i !== nav.mainComp).sort((a, b) => size.get(b)! - size.get(a)!)
for (const id of ids.slice(0, 14)) {
  const k = size.get(id)!
  const cx = sx.get(id)! / k
  const cy = sy.get(id)! / k
  // a cell of this area near its centre
  let sc = -1
  let bd = Infinity
  for (let c = 0; c < n; c++) {
    if (nav.comp[c] !== id) continue
    const d = Math.hypot(nav.cellX(c) - cx, nav.cellY(c) - cy)
    if (d < bd) {
      bd = d
      sc = c
    }
  }
  const px = nav.cellX(sc)
  const py = nav.cellY(sc)
  const a = performance.now()
  const out = nav.hopPlan(px, py, nav.mainComp)
  const b = performance.now()
  const back = out ? nav.hopPlan(out.tx, out.ty, id) : null
  console.log(
    `area ${id}: ${k * 4} m2 at ${cx.toFixed(0)},${cy.toFixed(0)} z ${w.groundAt(px, py).toFixed(0)} | out: ${out ? `launch ${out.lx.toFixed(0)},${out.ly.toFixed(0)} land ${out.tx.toFixed(0)},${out.ty.toFixed(0)} rise ${out.rise.toFixed(1)} m energy ${out.energy.toFixed(0)}` : "NONE"} (${(b - a).toFixed(1)} ms) | in: ${back ? `rise ${back.rise.toFixed(1)} energy ${back.energy.toFixed(0)}` : "NONE"} (${(performance.now() - b).toFixed(1)} ms)`,
  )
}
