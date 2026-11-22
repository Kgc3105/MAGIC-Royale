/**
 * Automated tuning against the local simulator.
 *
 *   node tools/tune.ts                       one screening + confirmation round from the defaults
 *   node tools/tune.ts --start '{"lmin":140}' --workers 12 --screen 300 --confirm 600
 *
 * Each round: every single-parameter change is screened against the incumbent on the same seeds
 * (paired comparison), the most promising few are re-tested on fresh seeds, and those that hold
 * up are adopted and checked jointly. The simulator's hunters are our own guess at the house
 * bots, so a win here is a candidate for the real scorer, not a guarantee.
 */
import { spawn } from "node:child_process"
import { fileURLToPath } from "node:url"

const args = new Map<string, string>()
for (let i = 2; i < process.argv.length; i++) {
  const a = process.argv[i]
  if (a === "--simargs") args.set("simargs", process.argv[++i] ?? "")
  else if (a.startsWith("--")) args.set(a.slice(2), process.argv[i + 1] && !process.argv[i + 1].startsWith("--") ? process.argv[++i] : "1")
}
const SIM = fileURLToPath(new URL("./sim.ts", import.meta.url))
const WORKERS = Number(args.get("workers") ?? 12)
const N_SCREEN = Number(args.get("screen") ?? 300)
const N_CONFIRM = Number(args.get("confirm") ?? 600)
const BLOCK = 50
const SCENARIO = args.get("scenario") ?? "gauntlet"
const SEED_BASE = Number(args.get("seedbase") ?? 20000)
const TOP = Number(args.get("top") ?? 6)
/** extra simulator arguments for every run, e.g. --simargs "--house real" */
const EXTRA = (args.get("simargs") ?? "").split(" ").filter(Boolean)

type Cfg = Record<string, number>
/** alternatives to try for each parameter (the default is whatever the bot does without an override) */
const SPACE: Record<string, number[]> = {
  engR: [140, 210], engNear: [40, 90],
  evHp: [0.3, 0.5], evN: [3, 5, 6], evHp2: [0.5, 0.8], exHp0: [0.6, 0.9], exHp1: [0.7], ogHp: [0.6, 0.95],
  lmin: [100, 150], lmax: [165, 199], gmin: [30, 60], gmax: [70, 110], kw: [1.5, 3.5], kboost: [0, 40],
  flipA: [0.3, 0.7], flipB: [0.4, 1.2], jinkLos: [0.4, 1.1], jinkNo: [0, 0.5], hideW: [0.5, 1.5], losW1: [0.8, 2.0],
  dodgeW: [4, 10, 14], dodgeR: [1, 3.5], smooth: [0, 0.4], h1: [0.3, 0.6],
  tolYaw: [0.45, 0.8], lat: [0.05, 0.1], gunT: [0.6, 1.0], preCharge: [0.7, 3], runChargeHp: [0, 0.3], relIdle: [0.5, 0.9], relFight: [0.2, 0.6],
  tsD: [40, 100], tsHp: [0.7, 3], tsLos: [1.5, 4], tsStick: [0.2, 1.2], tsAtk: [0, 2],
  farmHp: [0.75], farmWait: [3], crateR: [100, 200], cubeR: [60, 130], grabR: [30, 70], grabLaser: [80, 170],
  upg: [1, 2, 3], predAim: [0], drop: [1],
}

// ------------------------------------------------------------------ running the simulator
let running = 0
const queue: (() => void)[] = []
function slot(): Promise<void> {
  if (running < WORKERS) {
    running++
    return Promise.resolve()
  }
  return new Promise((res) => queue.push(res))
}
function release() {
  const next = queue.shift()
  if (next) next()
  else running--
}

/** per-match scores for one block of seeds */
async function block(cfg: Cfg, seed: number, n: number): Promise<number[]> {
  await slot()
  try {
    return await new Promise<number[]>((resolve) => {
      const a = [SIM, "--scenario", SCENARIO, "--matches", String(n), "--seed", String(seed), "--me", JSON.stringify(cfg), ...EXTRA]
      const p = spawn(process.execPath, a, { stdio: ["ignore", "pipe", "ignore"] })
      let out = ""
      p.stdout.on("data", (d) => (out += d))
      p.on("close", () => {
        const scores: number[] = []
        for (const m of out.matchAll(/^seed\s+\d+ \| outlived\s+(\d+)/gm)) scores.push(Number(m[1]))
        resolve(scores)
      })
      p.on("error", () => resolve([]))
    })
  } finally {
    release()
  }
}
async function evaluate(cfg: Cfg, seed0: number, n: number): Promise<number[]> {
  const jobs: Promise<number[]>[] = []
  for (let k = 0; k < n; k += BLOCK) jobs.push(block(cfg, seed0 + k, Math.min(BLOCK, n - k)))
  return (await Promise.all(jobs)).flat()
}
const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / Math.max(1, v.length)
/** paired difference b - a: mean, standard error, t */
function paired(a: number[], b: number[]) {
  const n = Math.min(a.length, b.length)
  const d: number[] = []
  for (let i = 0; i < n; i++) d.push(b[i] - a[i])
  const m = mean(d)
  const sd = Math.sqrt(d.reduce((s, x) => s + (x - m) * (x - m), 0) / Math.max(1, n - 1))
  const se = sd / Math.sqrt(Math.max(1, n))
  return { diff: m, se, t: se > 0 ? m / se : 0, n }
}
const fmt = (x: number) => (x >= 0 ? "+" : "") + x.toFixed(2)

// ------------------------------------------------------------------ one round
async function main() {
  let inc: Cfg = args.has("start") ? JSON.parse(args.get("start")!) : {}
  const rounds = Number(args.get("rounds") ?? 1)
  let seed = SEED_BASE
  for (let round = 1; round <= rounds; round++) {
    const t0 = Date.now()
    console.log(`\n=== round ${round}: incumbent ${JSON.stringify(inc)}`)
    const sSeed = seed
    seed += N_SCREEN + 1000
    const alts: { key: string; val: number; cfg: Cfg }[] = []
    for (const [key, vals] of Object.entries(SPACE)) for (const val of vals) if (inc[key] !== val) alts.push({ key, val, cfg: { ...inc, [key]: val } })
    const [base, ...scored] = await Promise.all([evaluate(inc, sSeed, N_SCREEN), ...alts.map((a) => evaluate(a.cfg, sSeed, N_SCREEN))])
    console.log(`screened ${alts.length} changes on ${base.length} matches each; incumbent ${mean(base).toFixed(2)} (${((Date.now() - t0) / 60000).toFixed(1)} min)`)
    const ranked = alts.map((a, i) => ({ ...a, ...paired(base, scored[i]) })).sort((p, q) => q.t - p.t)
    for (const r of ranked) console.log(`  ${r.key}=${r.val}`.padEnd(22) + `${fmt(r.diff)} ± ${r.se.toFixed(2)}  t=${r.t.toFixed(1)}`)

    // confirm the best few on fresh seeds
    const cands = ranked.filter((r) => r.diff > 0.12 && r.t > 1.0).slice(0, TOP)
    if (!cands.length) {
      console.log("nothing promising")
      break
    }
    const cSeed = seed
    seed += N_CONFIRM + 1000
    const [cBase, ...cScored] = await Promise.all([evaluate(inc, cSeed, N_CONFIRM), ...cands.map((c) => evaluate(c.cfg, cSeed, N_CONFIRM))])
    console.log(`confirmation on ${cBase.length} fresh matches; incumbent ${mean(cBase).toFixed(2)}`)
    const kept: typeof cands = []
    cands.forEach((c, i) => {
      const r = paired(cBase, cScored[i])
      const ok = r.diff > 0.1 && r.t > 1.2
      console.log(`  ${c.key}=${c.val}`.padEnd(22) + `screen ${fmt(c.diff)}  confirm ${fmt(r.diff)} ± ${r.se.toFixed(2)} t=${r.t.toFixed(1)}  ${ok ? "KEEP" : "drop"}`)
      if (ok) kept.push({ ...c, diff: r.diff, t: r.t })
    })
    if (!kept.length) {
      console.log("nothing held up")
      break
    }
    // adopt together, one value per parameter, and check the combination on fresh seeds
    const next: Cfg = { ...inc }
    const used = new Set<string>()
    for (const k of kept.sort((p, q) => q.t - p.t)) {
      if (used.has(k.key)) continue
      used.add(k.key)
      next[k.key] = k.val
    }
    const jSeed = seed
    seed += N_CONFIRM + 1000
    const [jBase, jNext] = await Promise.all([evaluate(inc, jSeed, N_CONFIRM), evaluate(next, jSeed, N_CONFIRM)])
    const j = paired(jBase, jNext)
    console.log(`joint check ${JSON.stringify(next)}: ${mean(jBase).toFixed(2)} -> ${mean(jNext).toFixed(2)}  (${fmt(j.diff)} ± ${j.se.toFixed(2)}, t=${j.t.toFixed(1)})`)
    if (j.diff > 0.1) inc = next
    else {
      const best = kept[0]
      console.log(`combination did not hold; keeping only ${best.key}=${best.val}`)
      inc = { ...inc, [best.key]: best.val }
    }
    console.log(`round ${round} done in ${((Date.now() - t0) / 60000).toFixed(1)} min -> ${JSON.stringify(inc)}`)
  }
  console.log(`\nRESULT ${JSON.stringify(inc)}`)
}
main()
