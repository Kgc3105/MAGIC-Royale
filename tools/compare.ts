/**
 * Compare several tunings of the bot on the same simulated matches (paired).
 *
 *   node tools/compare.ts --n 600 --seed 40000 '{}' '{"evN":4}' '{"evN":4,"dodgeW":7}'
 *
 * The first config is the reference; each other one is reported as a paired difference from it.
 */
import { spawn } from "node:child_process"
import { fileURLToPath } from "node:url"

const SIM = fileURLToPath(new URL("./sim.ts", import.meta.url))
const opts = new Map<string, string>()
const cfgs: string[] = []
for (let i = 2; i < process.argv.length; i++) {
  const a = process.argv[i]
  if (a.startsWith("--")) opts.set(a.slice(2), process.argv[++i])
  else cfgs.push(a)
}
const N = Number(opts.get("n") ?? 400)
const SEED = Number(opts.get("seed") ?? 40000)
const WORKERS = Number(opts.get("workers") ?? 14)
const SCENARIO = opts.get("scenario") ?? "gauntlet"
const BLOCK = 50
/** extra simulator arguments, e.g. --simargs "--vision hunters" */
const EXTRA = (opts.get("simargs") ?? "").split(" ").filter(Boolean)

let running = 0
const queue: (() => void)[] = []
const slot = () => (running < WORKERS ? (running++, Promise.resolve()) : new Promise<void>((res) => queue.push(res)))
const release = () => {
  const next = queue.shift()
  if (next) next()
  else running--
}
async function block(cfg: string, seed: number, n: number): Promise<number[]> {
  await slot()
  try {
    return await new Promise<number[]>((resolve) => {
      const p = spawn(process.execPath, [SIM, "--scenario", SCENARIO, "--matches", String(n), "--seed", String(seed), "--me", cfg, ...EXTRA], { stdio: ["ignore", "pipe", "ignore"] })
      let out = ""
      p.stdout.on("data", (d) => (out += d))
      p.on("close", () => resolve([...out.matchAll(/^seed\s+\d+ \| outlived\s+(\d+)/gm)].map((m) => Number(m[1]))))
      p.on("error", () => resolve([]))
    })
  } finally {
    release()
  }
}
async function evaluate(cfg: string): Promise<number[]> {
  const jobs: Promise<number[]>[] = []
  for (let k = 0; k < N; k += BLOCK) jobs.push(block(cfg, SEED + k, Math.min(BLOCK, N - k)))
  return (await Promise.all(jobs)).flat()
}
const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / Math.max(1, v.length)

const t0 = Date.now()
const res = await Promise.all(cfgs.map(evaluate))
console.log(`${SCENARIO}, ${res[0].length} matches each, seeds from ${SEED} (${((Date.now() - t0) / 60000).toFixed(1)} min)`)
res.forEach((r, i) => {
  let line = `${mean(r).toFixed(2)}  ${cfgs[i]}`
  if (i === 0) {
    const m = mean(r)
    const sd = Math.sqrt(r.reduce((s, x) => s + (x - m) * (x - m), 0) / Math.max(1, r.length - 1))
    line += `   (per-match sd ${sd.toFixed(2)}; a 48-match score varies by ±${(sd / Math.sqrt(48)).toFixed(2)})`
  }
  if (i > 0) {
    const n = Math.min(r.length, res[0].length)
    const d = Array.from({ length: n }, (_, k) => r[k] - res[0][k])
    const m = mean(d)
    const se = Math.sqrt(d.reduce((s, x) => s + (x - m) * (x - m), 0) / Math.max(1, n - 1)) / Math.sqrt(n)
    line += `   vs first: ${m >= 0 ? "+" : ""}${m.toFixed(2)} ± ${se.toFixed(2)} (t=${se > 0 ? (m / se).toFixed(1) : "0"})`
  }
  console.log(line)
})
