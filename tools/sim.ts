/**
 * Local sparring simulator. NOT the real server: it follows protocol.md as closely as is practical
 * (movement, walls, zone, guns, lasers, lightning, cubes, upgrades, drops) and pits the bot against
 * scripted hunters whose behaviour is our own guess. Use it to catch broken behaviour and compare
 * versions, not to predict a leaderboard score.
 *
 *   node tools/sim.ts --scenario gauntlet --matches 8 --seed 1
 *   node tools/sim.ts --scenario nav --matches 4
 *   node tools/sim.ts --scenario duel --skill hard --matches 10
 */
import { readFileSync } from "node:fs"
import { gunzipSync } from "node:zlib"
import type { BotAction, Config, Hello, Observation } from "../mechfight.ts"
import { clamp, interceptTime, mulberry32, TAU, wrap } from "../src/aim.ts"
import { Brain } from "../src/brain.ts"
import { Nav } from "../src/nav.ts"
import { World, type Prism } from "../src/world.ts"

// ------------------------------------------------------------------------------------------- args
const args = new Map<string, string>()
for (let i = 2; i < process.argv.length; i++) {
  const a = process.argv[i]
  if (a.startsWith("--")) args.set(a.slice(2), process.argv[i + 1] && !process.argv[i + 1].startsWith("--") ? process.argv[++i] : "1")
}
const KIT = args.get("kit") ?? new URL("../mechfight-kit", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1")
const SCENARIO = args.get("scenario") ?? "gauntlet"
const MATCHES = Number(args.get("matches") ?? 4)
const SEED0 = Number(args.get("seed") ?? 1)
const VERBOSE = args.has("verbose")
const SKILL = args.get("skill") ?? "medium"
const STATIC_ZONE = args.has("static-zone")
/** hunters always know where the bot is (the first, harsher model) instead of sharing vision */
const OMNISCIENT = args.has("omniscient")
const HUNT_DELAY = Number(args.get("delay") ?? 0)
/** when every house bot joins the hunt regardless of kills (our guess at "late in the match") */
const LATE_T = Number(args.get("late") ?? 255)
const TRACE = args.has("trace")

const hello0: Hello = JSON.parse(gunzipSync(readFileSync(`${decodeURIComponent(KIT)}/check/stream.jsonl.gz`)).toString("utf8").split("\n")[0])
hello0.map.dir = `${decodeURIComponent(KIT)}/arena`
const cfg: Config = hello0.config
const world = new World(hello0.map.dir, hello0.map.files)
const npcNav = new Nav(world)
const DT = 1 / (cfg.match.tick_hz ?? 20)
const M = cfg.mech

// ---------------------------------------------------------------------------------------- physics
type Wpn = { kind: string; cooldown: number; ammo: number; reload: number; charge: number; held: number; stats: Record<string, number> }
type Mech = {
  id: number
  alive: boolean
  x: number
  y: number
  z: number
  vx: number
  vy: number
  vz: number
  yaw: number
  pitch: number
  grounded: boolean
  energy: number
  hp: number
  cubes: number
  upgrades: string[]
  stats: Record<string, number>
  weapons: (Wpn | null)[]
  lastCombatT: number
  lastSpendT: number
  cmd: BotAction
  ai: Dummy | null
  memUntil: number
  diedAt: number
  offers: number[]
}
type Bullet = { id: number; kind: "gun" | "mortar"; by: number; x: number; y: number; z: number; vx: number; vy: number; vz: number; left: number; dmg: number }
type Loot = { id: number; kind: string; x: number; y: number; z: number; hp: number }

const prismBox = world.prisms.map((p: Prism) => {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity
  for (let k = 0; k < p.ring.length; k += 2) {
    x0 = Math.min(x0, p.ring[k])
    x1 = Math.max(x1, p.ring[k])
    y0 = Math.min(y0, p.ring[k + 1])
    y1 = Math.max(y1, p.ring[k + 1])
  }
  return { p, x0, x1, y0, y1, rings: [p.ring, ...(p.holes ?? [])] }
})

function insideRings(rings: number[][], x: number, y: number): boolean {
  let c = false
  for (const r of rings) {
    const n = r.length >> 1
    for (let a = 0, b = n - 1; a < n; b = a++) {
      const ya = r[2 * a + 1], yb = r[2 * b + 1]
      if (ya > y !== yb > y && x < ((r[2 * b] - r[2 * a]) * (y - ya)) / (yb - ya) + r[2 * a]) c = !c
    }
  }
  return c
}

/** exact cylinder-vs-prism wall push-out (independent of the bot's rasterised map) */
function collide(m: Mech) {
  const r = m.stats.radius
  const h = m.stats.height
  for (let pass = 0; pass < 2; pass++) {
    for (const b of prismBox) {
      if (m.x < b.x0 - r || m.x > b.x1 + r || m.y < b.y0 - r || m.y > b.y1 + r) continue
      if (m.z >= b.p.top - M.step_up || m.z + h <= b.p.base) continue
      let bd = Infinity, qx = 0, qy = 0
      for (const ring of b.rings) {
        const n = ring.length >> 1
        for (let a = 0, c = n - 1; a < n; c = a++) {
          const ax = ring[2 * c], ay = ring[2 * c + 1], ex = ring[2 * a] - ax, ey = ring[2 * a + 1] - ay
          const L = ex * ex + ey * ey
          const u = L > 0 ? clamp(((m.x - ax) * ex + (m.y - ay) * ey) / L, 0, 1) : 0
          const px = ax + ex * u, py = ay + ey * u
          const d = Math.hypot(m.x - px, m.y - py)
          if (d < bd) {
            bd = d
            qx = px
            qy = py
          }
        }
      }
      const inside = insideRings(b.rings, m.x, m.y)
      if (!inside && bd >= r) continue
      let nx = m.x - qx, ny = m.y - qy
      const d = Math.hypot(nx, ny)
      if (d < 1e-6) continue
      nx /= d
      ny /= d
      if (inside) {
        nx = -nx
        ny = -ny
      }
      m.x = qx + nx * r
      m.y = qy + ny * r
      const vn = m.vx * nx + m.vy * ny
      if (vn < 0) {
        m.vx -= vn * nx
        m.vy -= vn * ny
      }
    }
  }
  m.x = clamp(m.x, world.bMinX + r, world.bMaxX - r)
  m.y = clamp(m.y, world.bMinY + r, world.bMaxY - r)
}

/** highest thing the mech's centre can stand on at its current height */
function surface(m: Mech): number {
  let s = world.groundAt(m.x, m.y)
  for (const b of prismBox) {
    if (m.x < b.x0 || m.x > b.x1 || m.y < b.y0 || m.y > b.y1) continue
    if (b.p.top > s && m.z >= b.p.top - M.step_up - 0.01 && insideRings(b.rings, m.x, m.y)) s = b.p.top
  }
  return s
}

function stepMech(m: Mech, t: number) {
  const c = m.cmd
  const S = m.stats
  // torso
  const want = c.aim ?? [m.yaw, m.pitch]
  m.yaw = wrap(m.yaw + clamp(wrap(want[0] - m.yaw), -S.turn_rate * DT, S.turn_rate * DT))
  m.pitch = clamp(m.pitch + clamp(clamp(want[1], S.min_pitch, S.max_pitch) - m.pitch, -S.pitch_rate * DT, S.pitch_rate * DT), S.min_pitch, S.max_pitch)
  // legs
  let mx = c.move?.[0] ?? 0, my = c.move?.[1] ?? 0
  const ml = Math.hypot(mx, my)
  if (ml > 1) {
    mx /= ml
    my /= ml
  }
  const charging = m.weapons.some((w) => w?.kind === "laser" && w.charge > 0)
  let spent = false
  let speed = S.walk_speed
  if (c.boost && m.grounded && ml > 0.01 && m.energy > 0) {
    speed = S.boost_speed
    m.energy -= S.boost_drain * DT
    spent = true
  }
  if (charging) speed *= S.charge_move_mult
  const acc = (m.grounded ? S.ground_accel : S.air_accel) * DT
  const ex = mx * speed - m.vx, ey = my * speed - m.vy
  const el = Math.hypot(ex, ey)
  if (el <= acc) {
    m.vx = mx * speed
    m.vy = my * speed
  } else {
    m.vx += (ex / el) * acc
    m.vy += (ey / el) * acc
  }
  if (c.jump && m.grounded) {
    m.vz = S.jump_speed
    m.grounded = false
  }
  if (!m.grounded) {
    if (c.boost && m.energy > 0) {
      m.vz = Math.min(S.max_rise_speed, m.vz + S.jet_accel * DT)
      m.energy -= S.jet_drain * DT
      spent = true
    }
    m.vz -= S.gravity * DT
  }
  if (spent) m.lastSpendT = t
  else if (t - m.lastSpendT >= S.energy_regen_delay) m.energy = Math.min(S.energy, m.energy + S.energy_regen * DT)
  m.energy = Math.max(0, m.energy)
  const ox = m.x, oy = m.y
  const g0 = world.groundAt(ox, oy)
  m.x += m.vx * DT
  m.y += m.vy * DT
  collide(m)
  if (m.grounded) {
    // slopes steeper than the limit cannot be walked up
    const g1 = world.groundAt(m.x, m.y)
    const run = Math.hypot(m.x - ox, m.y - oy)
    if (m.z <= g0 + 0.3 && g1 - g0 > Math.tan((M.max_slope_deg * Math.PI) / 180) * run + 0.02) {
      m.x = ox
      m.y = oy
      m.vx = 0
      m.vy = 0
    }
  }
  const s = surface(m)
  if (m.grounded) {
    if (s >= m.z - 0.6) m.z = s
    else {
      m.grounded = false
      m.vz = 0
    }
  } else {
    m.z += m.vz * DT
    if (m.z <= s) {
      m.z = s
      m.vz = 0
      m.grounded = true
    }
  }
}

// ------------------------------------------------------------------------------------- scripted AI
const SKILLS: Record<string, { sigma: number; strafe: boolean; laser: boolean; range: number; lead: boolean; boost: boolean }> = {
  easy: { sigma: 0.05, strafe: false, laser: false, range: 55, lead: false, boost: false },
  medium: { sigma: 0.02, strafe: true, laser: false, range: 70, lead: true, boost: false },
  hard: { sigma: 0.008, strafe: true, laser: true, range: 110, lead: true, boost: false },
  expert: { sigma: 0.002, strafe: true, laser: true, range: 130, lead: true, boost: true },
}

class Dummy {
  skill: (typeof SKILLS)[string]
  rnd: () => number
  path: number[] = []
  pathT = -1e9
  strafe = 1
  flipT = 0
  noise = 0
  noiseT = 0
  active = false
  home: [number, number] = [0, 0]
  search: [number, number] | null = null
  searchT = -1
  lastTgt = -1
  lastTgtT = -1e9
  constructor(skill: string, seed: number) {
    this.skill = SKILLS[skill] ?? SKILLS.medium
    this.rnd = mulberry32(seed)
  }
  private walkTo(m: Mech, gx: number, gy: number, t: number): [number, number] {
    const d = Math.hypot(gx - m.x, gy - m.y)
    if (d < 3) return [0, 0]
    npcNav.need = m.stats.radius + 0.55
    npcNav.now = t
    if (npcNav.segOk(m.x, m.y, gx, gy)) return [(gx - m.x) / d, (gy - m.y) / d]
    if (t - this.pathT > 1.5 || this.path.length === 0) {
      this.path = npcNav.findPath(m.x, m.y, gx, gy, 20000)
      this.pathT = t
    }
    const n = this.path.length >> 1
    let k = 0
    let bd = Infinity
    for (let i = 0; i < n; i++) {
      const dd = Math.hypot(this.path[2 * i] - m.x, this.path[2 * i + 1] - m.y)
      if (dd < bd) {
        bd = dd
        k = i
      }
    }
    let aim = Math.min(n - 1, k + 1)
    for (const look of [10, 6, 3]) {
      const j = Math.min(n - 1, k + look)
      if (npcNav.segOk(m.x, m.y, this.path[2 * j], this.path[2 * j + 1])) {
        aim = j
        break
      }
    }
    if (n === 0) return [(gx - m.x) / d, (gy - m.y) / d]
    const ax = this.path[2 * aim] - m.x, ay = this.path[2 * aim + 1] - m.y
    const al = Math.hypot(ax, ay)
    return al > 1e-6 ? [ax / al, ay / al] : [0, 0]
  }
  /**
   * `seen`: some house bot can see the target right now (they share vision). Otherwise the hunter
   * only has `believed`, the last place the squad saw it, and searches around there.
   */
  decide(m: Mech, tgt: Mech | null, t: number, zone: { cx: number; cy: number; r: number; ncx: number; ncy: number; nr: number; hurry: boolean }, seen = true, believed: [number, number] = [0, 0]): BotAction {
    const out: BotAction = { fire: [false, false] }
    if (tgt && tgt.alive && this.active && !seen) {
      const dzn0 = Math.hypot(m.x - zone.ncx, m.y - zone.ncy)
      let gx = believed[0], gy = believed[1]
      if (zone.hurry && dzn0 > zone.nr - 10) {
        gx = zone.ncx
        gy = zone.ncy
      } else if (Math.hypot(gx - m.x, gy - m.y) < 18) {
        if (!this.search || t > this.searchT || Math.hypot(this.search[0] - m.x, this.search[1] - m.y) < 6) {
          this.search = null
          for (let k = 0; k < 20 && !this.search; k++) {
            const a = this.rnd() * TAU, r = 20 + this.rnd() * 55
            const px = believed[0] + Math.cos(a) * r, py = believed[1] + Math.sin(a) * r
            if (npcNav.free(px, py)) this.search = [px, py]
          }
          this.searchT = t + 8
        }
        if (this.search) {
          gx = this.search[0]
          gy = this.search[1]
        }
      }
      out.move = this.walkTo(m, gx, gy, t)
      const mv = Math.hypot(out.move[0], out.move[1])
      out.aim = [(mv > 0.1 ? Math.atan2(out.move[1], out.move[0]) : m.yaw + 0.3) + 0.9 * Math.sin(t * 1.1 + m.id), 0]
      return out
    }
    const dzn = Math.hypot(m.x - zone.ncx, m.y - zone.ncy)
    if (zone.hurry && dzn > zone.nr - 10) {
      out.move = this.walkTo(m, zone.ncx, zone.ncy, t)
      out.boost = Math.hypot(m.x - zone.cx, m.y - zone.cy) > zone.r
      if (!tgt) return out
    }
    if (!tgt || !tgt.alive || !this.active) {
      if (!out.move) out.move = dzn > zone.nr * 0.6 ? this.walkTo(m, zone.ncx + this.home[0] * zone.nr * 0.5, zone.ncy + this.home[1] * zone.nr * 0.5, t) : [0, 0]
      return out
    }
    const mz = m.z + m.stats.muzzle_height
    const cz = tgt.z + tgt.stats.height * 0.5
    const d = Math.hypot(tgt.x - m.x, tgt.y - m.y)
    const los = world.losClear(m.x, m.y, mz, tgt.x, tgt.y, cz)
    if (!out.move) {
      if (!los || d > this.skill.range + 12) out.move = this.walkTo(m, tgt.x, tgt.y, t)
      else if (d < this.skill.range - 20) out.move = [(m.x - tgt.x) / d, (m.y - tgt.y) / d]
      else if (this.skill.strafe) {
        if (t > this.flipT) {
          this.strafe = -this.strafe
          this.flipT = t + 0.5 + this.rnd() * 0.9
        }
        const px = (-(tgt.y - m.y) / d) * this.strafe, py = ((tgt.x - m.x) / d) * this.strafe
        out.move = npcNav.free(m.x + px * 5, m.y + py * 5) ? [px, py] : [0, 0]
      } else out.move = [0, 0]
      out.boost = this.skill.boost && (!los || d > this.skill.range + 40) && m.energy > 30
    }
    if (t > this.noiseT) {
      this.noise = (this.rnd() + this.rnd() + this.rnd() - 1.5) * 2 * this.skill.sigma
      this.noiseT = t + 0.4
    }
    let dx = tgt.x - m.x, dy = tgt.y - m.y, dz = cz - mz
    const gun = m.weapons[0]
    const laser = m.weapons[1]?.kind === "laser" ? m.weapons[1] : null
    const laserHot = !!laser && laser.charge >= laser.stats.charge - 0.25
    if (this.skill.lead && gun && !laserHot) {
      const tl = interceptTime(dx, dy, dz, tgt.vx, tgt.vy, tgt.vz, gun.stats.speed)
      if (tl > 0) {
        dx += tgt.vx * tl
        dy += tgt.vy * tl
        dz += tgt.vz * tl
      }
    }
    const yaw = Math.atan2(dy, dx) + this.noise
    const pitch = Math.atan2(dz, Math.hypot(dx, dy))
    out.aim = [yaw, pitch]
    const tol = Math.atan2(tgt.stats.radius, Math.max(1, d)) + 0.01
    const onAim = Math.abs(wrap(yaw - m.yaw)) < tol
    if (gun && los && d < gun.stats.range - 5 && onAim) out.fire![0] = true
    if (laser && d < laser.stats.range - 5) {
      let hold = los || laser.charge > 0
      if (laser.charge >= laser.stats.charge - 0.01 && los && Math.abs(wrap(Math.atan2(tgt.y - m.y, tgt.x - m.x) - m.yaw)) < tol) hold = false
      out.fire![1] = hold && laser.cooldown <= 0.05
      if (laser.charge > 0 && !hold) out.fire![1] = false
    }
    return out
  }
}

// ------------------------------------------------------------------------------------------ match
type Result = { score: number; t: number; hp: number; kills: number; dealt: number; taken: number; zoneDmg: number; cubes: number; weapons: string; stuck: number; msMax: number; msP99: number; err: number; modes: Record<string, number>; travelled: number; seen: number; laserT: number; gs: number; gh: number; ls: number; lh: number; losT: number; conT: number; lz2: Record<string, number>; src: Record<string, number>; activeEnd: number; killT: number[]; pend: string; cubesAtEnd: number; roofLaserSeenT: number }

function runMatch(seed: number): Result {
  const rnd = mulberry32(seed * 7919 + 17)
  const gauss = () => (rnd() + rnd() + rnd() + rnd() - 2) * 1.2247
  let nextId = 30000
  const mkWpn = (kind: string, stats?: Record<string, number>): Wpn => {
    const st = { ...(cfg.weapons as Record<string, Record<string, number>>)[kind], ...(stats ?? {}) }
    return { kind, cooldown: 0, ammo: st.mag ?? 0, reload: 0, charge: 0, held: 0, stats: st }
  }
  const zc = cfg.zone.start_center
  const freeSpot = (cx: number, cy: number, r: number): [number, number] => {
    npcNav.need = 3.1
    npcNav.now = 0
    for (let k = 0; k < 400; k++) {
      const a = rnd() * TAU, rr = Math.sqrt(rnd()) * r
      const px = cx + Math.cos(a) * rr, py = cy + Math.sin(a) * rr
      if (px < world.bMinX + 30 || px > world.bMaxX - 30 || py < world.bMinY + 30 || py > world.bMaxY - 30) continue
      if (npcNav.free(px, py)) return [px, py]
    }
    return [cx, cy]
  }
  const mkMech = (id: number, x: number, y: number): Mech => ({
    id, alive: true, x, y, z: world.groundAt(x, y), vx: 0, vy: 0, vz: 0, yaw: rnd() * TAU - Math.PI, pitch: 0, grounded: true, energy: M.energy, hp: M.hp, cubes: 0, upgrades: [],
    stats: { ...M }, weapons: [mkWpn("gun"), null], lastCombatT: -1e9, lastSpendT: -1e9, cmd: {}, ai: null, memUntil: -1, diedAt: Infinity, offers: [],
  })
  // spawns
  const spawns: [number, number][] = []
  const nMech = SCENARIO === "nav" ? 1 : SCENARIO === "duel" ? 2 : 16
  for (let guard = 0; spawns.length < nMech && guard < 5000; guard++) {
    const p = freeSpot(zc[0], zc[1], cfg.zone.start_radius * (cfg.match.spawn_zone_fraction ?? 0.8))
    if (spawns.every((q) => Math.hypot(q[0] - p[0], q[1] - p[1]) >= (SCENARIO === "duel" ? 150 : (cfg.match.spawn_min_separation ?? 130)))) spawns.push(p)
  }
  const mechs: Mech[] = spawns.map((p, i) => mkMech(i, p[0], p[1]))
  const me = mechs[0]
  const skills = ["easy", "easy", "easy", "easy", "medium", "medium", "medium", "medium", "hard", "hard", "hard", "hard", "expert", "expert", "expert"]
  for (let i = 1; i < mechs.length; i++) {
    const sk = SCENARIO === "duel" ? SKILL : skills[i - 1]
    mechs[i].ai = new Dummy(sk, seed * 131 + i)
    const a = rnd() * TAU
    mechs[i].ai!.home = [Math.cos(a), Math.sin(a)]
    if (SKILLS[sk].laser) mechs[i].weapons[1] = mkWpn("laser")
  }
  // loot
  const pickups: Loot[] = []
  const cubes: Loot[] = []
  const crates: Loot[] = []
  const kinds = ["mortar", "laser", "lightning"]
  /** a spot on a real, flat rooftop a mech could jet onto (5-35 m up) */
  const roofSpot = (): [number, number, number] | null => {
    for (let k = 0; k < 600; k++) {
      const a = rnd() * TAU, rr = Math.sqrt(rnd()) * cfg.zone.start_radius * 0.85
      const px = zc[0] + Math.cos(a) * rr, py = zc[1] + Math.sin(a) * rr
      if (px < world.bMinX + 30 || px > world.bMaxX - 30 || py < world.bMinY + 30 || py > world.bMaxY - 30) continue
      const top = world.topAt(px, py)
      const h = top - world.groundAt(px, py)
      if (h < 5 || h > 35) continue
      if ([[3, 0], [-3, 0], [0, 3], [0, -3]].every(([dx, dy]) => Math.abs(world.topAt(px + dx, py + dy) - top) < 1.2)) return [px, py, top]
    }
    return null
  }
  for (let i = 0; i < cfg.loot.floor_count; i++) {
    const kind = kinds[Math.floor(rnd() * 3)]
    const rs = rnd() < cfg.loot.floor_roof_fraction ? roofSpot() : null
    if (rs) pickups.push({ id: 8000 + i, kind, x: rs[0], y: rs[1], z: rs[2], hp: 0 })
    else {
      const [px, py] = freeSpot(zc[0], zc[1], cfg.zone.start_radius * 0.85)
      pickups.push({ id: 8000 + i, kind, x: px, y: py, z: world.groundAt(px, py), hp: 0 })
    }
  }
  for (let i = 0; i < (cfg.crates.count ?? 36); i++) {
    const [px, py] = freeSpot(zc[0], zc[1], cfg.zone.start_radius * 0.85)
    crates.push({ id: 1000 + i, kind: "crate", x: px, y: py, z: world.groundAt(px, py), hp: cfg.crates.hp ?? 300 })
  }
  const drops: { id: number; x: number; y: number; z: number; lands: number; progress: number; holder: number | null; called: number }[] = []
  const dropTimes = [...(cfg.drops?.times ?? [])]

  // zone
  const phases = cfg.zone.phases
  let ph = 0
  let phaseStart = 0
  let cur = { cx: zc[0], cy: zc[1], r: cfg.zone.start_radius }
  const pickNext = (c: { cx: number; cy: number; r: number }, nr: number) => {
    for (let k = 0; k < 200; k++) {
      const a = rnd() * TAU, rr = Math.sqrt(rnd()) * (c.r - nr)
      const nx = c.cx + Math.cos(a) * rr, ny = c.cy + Math.sin(a) * rr
      if (nx > world.bMinX + 40 && nx < world.bMaxX - 40 && ny > world.bMinY + 40 && ny < world.bMaxY - 40) return { cx: nx, cy: ny, r: nr }
    }
    return { cx: c.cx, cy: c.cy, r: nr }
  }
  let next = pickNext(cur, phases[0].radius)

  // what-if switches: start the bot with cubes or a weapon to measure how much they are worth
  if (args.has("weapon")) me.weapons[1] = mkWpn(args.get("weapon")!)
  // --me '{"key":value}' tunes only our bot; --stance forces its posture
  const brain = new Brain({ ...hello0, slot: 0, seed: seed * 2654435761 }, { world, nav: new Nav(world) }, { tune: args.has("me") ? JSON.parse(args.get("me")!) : undefined, stance: args.get("stance") })
  const bullets: Bullet[] = []
  let beams: Observation["beams"] = []
  let events: Observation["events"] = []
  let pendingOffer: { id: number; choices: { id: string; name: string; desc: string }[]; expires: number } | null = null
  let offerId = 1
  // ---- free-for-all rivals: copies of this bot. Most play the practice style (other teams tuned
  // for the practice board and will brawl); every third one adapts to a free-for-all as ours does.
  type Pilot = { brain: Brain; events: Observation["events"]; offer: Observation["offer"]; mem: Map<number, number>; beams: Observation["beams"] }
  const rivals = new Map<number, Pilot>()
  if (SCENARIO === "ffa") {
    const nRival = Math.min(mechs.length - 1, Number(args.get("rivals") ?? 9))
    for (let i = 1; i <= nRival; i++) {
      mechs[i].ai = null
      mechs[i].weapons[1] = null
      rivals.set(i, {
        brain: new Brain({ ...hello0, slot: i, seed: (seed * 7 + i * 1009) >>> 0 }, { world, nav: new Nav(world) }, { stance: i % 3 === 0 ? "auto" : "brawl" }),
        events: [], offer: null, mem: new Map(), beams: [],
      })
    }
  }
  /** can mech o see a point (distance, view cone, line of sight)? */
  const canSee = (o: Mech, x: number, y: number, z1: number, z2: number) => {
    const d = Math.hypot(x - o.x, y - o.y)
    if (d > cfg.vision.range) return false
    if (d > cfg.vision.near_range && Math.abs(wrap(Math.atan2(y - o.y, x - o.x) - o.yaw)) > (cfg.vision.fov_deg * Math.PI) / 360) return false
    const ez = o.z + o.stats.eye_height
    return world.losClear(o.x, o.y, ez, x, y, z1) || world.losClear(o.x, o.y, ez, x, y, z2)
  }
  const doTake = (o: Mech, a: BotAction, ev: Observation["events"]) => {
    if (a.take == null) return
    const l = pickups.find((q) => q.id === a.take)
    if (!l) return void ev.push({ type: "take_failed", id: a.take, reason: "gone" })
    if (Math.hypot(l.x - o.x, l.y - o.y) > cfg.loot.pickup_radius || Math.abs(l.z - o.z) > 4) return void ev.push({ type: "take_failed", id: l.id, reason: "range" })
    if (o.weapons.some((w) => w?.kind === l.kind)) return void ev.push({ type: "take_failed", id: l.id, reason: "have" })
    let slot = o.weapons.findIndex((w) => !w)
    if (slot < 0 && (a.drop === 0 || a.drop === 1)) {
      slot = a.drop
      pickups.push({ id: nextId++, kind: o.weapons[slot]!.kind, x: o.x, y: o.y, z: o.z, hp: 0 })
    }
    if (slot < 0) return void ev.push({ type: "take_failed", id: l.id, reason: "full" })
    const w = mkWpn(l.kind)
    for (const id of o.upgrades) {
      const u = cfg.upgrades.find((q) => q.id === id)
      for (const e of u?.effects ?? []) {
        const path = e.path.split(".")
        if (path[0] === "weapons" && path[1] === l.kind) w.stats[path[2]] = (w.stats[path[2]] ?? 0) * (e.mul ?? 1) + (e.add ?? 0)
      }
    }
    w.cooldown = Math.max(cfg.loot.equip_s, w.stats.cooldown ?? 0)
    o.weapons[slot] = w
    pickups.splice(pickups.indexOf(l), 1)
    ev.push({ type: "took", id: l.id, kind: l.kind as never, slot })
  }
  let kills = 0, dealt = 0, taken = 0, zoneDmg = 0, stuck = 0, err = 0, travelled = 0
  const times: number[] = []
  const modes: Record<string, number> = {}
  let lastPos: [number, number] = [me.x, me.y]
  let believed: [number, number] = [me.x, me.y]
  let knownT = 0
  let seenTicks = 0
  let laserT = -1
  let roofLaserSeenT = -1
  let gs = 0, gh = 0, ls = 0, lh = 0, losT = 0, conT = 0
  const lz2 = { auto: 0, autoHit: 0, rel: 0, relHit: 0, relBlocked: 0, relOffAim: 0 }
  const src: Record<string, number> = {}
  const killT: number[] = []
  let activeEnd = 0
  let cubesAtEnd = -1
  let decTicks = 0
  let stillTicks = 0
  let deadBeforeMe = 0

  const cubeMult = (m: Mech) => 1 + (Number(cfg.cubes.damage_per_cube) || 0) * Math.min(m.cubes, Number(cfg.cubes.stat_cap) || 10)
  const maxHp = (m: Mech) => m.stats.hp * (1 + (Number(cfg.cubes.hp_per_cube) || 0) * Math.min(m.cubes, Number(cfg.cubes.stat_cap) || 10))
  const applyUpgrade = (m: Mech, id: string) => {
    const u = cfg.upgrades.find((q) => q.id === id)
    if (!u) return
    m.upgrades.push(id)
    for (const e of u.effects) {
      const path = e.path.split(".")
      if (path[0] === "mech") {
        const before = maxHp(m)
        m.stats[path[1]] = (m.stats[path[1]] ?? 0) * (e.mul ?? 1) + (e.add ?? 0)
        if (path[1] === "hp") m.hp += maxHp(m) - before
      } else if (path[0] === "weapons") for (const w of m.weapons) if (w && w.kind === path[1]) w.stats[path[2]] = (w.stats[path[2]] ?? 0) * (e.mul ?? 1) + (e.add ?? 0)
    }
  }
  const giveCube = (m: Mech, t: number) => {
    const before = maxHp(m)
    m.cubes++
    const sc = 1 + (Number(cfg.cubes.size_per_cube) || 0) * m.cubes
    m.stats.radius = M.radius * sc
    m.stats.height = M.height * sc
    m.stats.eye_height = M.eye_height * sc
    m.stats.muzzle_height = M.muzzle_height * sc
    m.hp += maxHp(m) - before
    if (m.id === 0) {
      events.push({ type: "pickup", cubes: m.cubes })
      if (cfg.cubes.upgrade_thresholds.includes(m.cubes)) m.offers.push(t)
    }
    const rvc = rivals.get(m.id)
    if (rvc) {
      rvc.events.push({ type: "pickup", cubes: m.cubes })
      if (cfg.cubes.upgrade_thresholds.includes(m.cubes)) m.offers.push(t)
    }
  }
  const hurt = (v: Mech, amount: number, weapon: string, by: Mech | null, sx: number, sy: number, t: number) => {
    if (!v.alive) return
    const bearing = Math.atan2(sy - v.y, sx - v.x)
    const rear = weapon !== "zone" && Math.abs(wrap(bearing - v.yaw)) > ((M.front_arc_deg ?? 180) * Math.PI) / 360
    let a = amount
    if (weapon !== "zone") a *= (rear ? M.rear_mult : 1) * (1 - (v.stats.armor ?? 0))
    v.hp -= a
    v.lastCombatT = t
    if (by) by.lastCombatT = t
    if (v.id === 0) {
      events.push({ type: "damaged", amount: a, weapon, bearing: weapon === "zone" ? null : bearing, by: by ? by.id : null, ...(rear ? { rear: true } : {}) })
      if (weapon === "zone") zoneDmg += a
      else {
        taken += a
        const sk = by?.ai ? (Object.entries(SKILLS).find(([, v]) => v === by.ai!.skill)?.[0] ?? "?") : by && rivals.has(by.id) ? "rival" : "?"
        src[`${sk}/${weapon}`] = (src[`${sk}/${weapon}`] ?? 0) + a
        if (rear) src.rear = (src.rear ?? 0) + a
      }
    }
    const rvV = rivals.get(v.id)
    if (rvV) rvV.events.push({ type: "damaged", amount: a, weapon, bearing: weapon === "zone" ? null : bearing, by: by ? by.id : null, ...(rear ? { rear: true } : {}) })
    const rvB = by ? rivals.get(by.id) : undefined
    if (rvB) rvB.events.push({ type: "hit", target: v.id, amount: a, weapon })
    if (by && by.id === 0) {
      events.push({ type: "hit", target: v.id, amount: a, weapon })
      dealt += a
      if (weapon === "gun") gh++
      else if (weapon === "laser") lh++
    }
    if (v.hp <= 0) {
      v.alive = false
      v.diedAt = t
      if (rvB) rvB.events.push({ type: "kill", target: v.id })
      if (by && by.id === 0) {
        kills++
        killT.push(t)
        events.push({ type: "kill", target: v.id })
      }
      for (let k = 0; k < v.cubes + (Number(cfg.cubes.death_drop_extra) || 0); k++) cubes.push({ id: nextId++, kind: "cube", x: v.x + gauss() * 2, y: v.y + gauss() * 2, z: v.z, hp: 0 })
      for (const w of v.weapons) if (w && w.kind !== "gun") pickups.push({ id: nextId++, kind: w.kind, x: v.x, y: v.y, z: v.z, hp: 0 })
    }
  }
  /** first mech hit by a ray; returns [mech, distance] */
  const rayHit = (by: Mech, ox: number, oy: number, oz: number, ux: number, uy: number, uz: number, range: number): [Mech | null, number] => {
    let best: Mech | null = null
    let bd = range
    for (const v of mechs) {
      if (!v.alive || v === by) continue
      const rx = v.x - ox, ry = v.y - oy
      const hl = Math.hypot(ux, uy)
      if (hl < 1e-6) continue
      const along = (rx * ux + ry * uy) / hl
      if (along <= 0) continue
      const perp = Math.abs(rx * uy - ry * ux) / hl
      if (perp > v.stats.radius) continue
      const dd = along / hl
      const hz = oz + uz * dd
      if (hz < v.z || hz > v.z + v.stats.height) continue
      if (dd < bd) {
        bd = dd
        best = v
      }
    }
    return [best, bd]
  }
  const liveZone = (t: number) => {
    const p = phases[Math.min(ph, phases.length - 1)]
    const ss = phaseStart + p.hold_s
    const f = ph >= phases.length ? 1 : clamp((t - ss) / p.shrink_s, 0, 1)
    return { cx: cur.cx + (next.cx - cur.cx) * f, cy: cur.cy + (next.cy - cur.cy) * f, r: cur.r + (next.r - cur.r) * f, ss, se: ss + p.shrink_s, dps: p.dps }
  }

  for (let k = 0; k < Number(args.get("cubes") ?? 0); k++) giveCube(me, 0)
  let t = 0
  let tick = 0
  const maxT = cfg.match.duration_s
  while (t < maxT && me.alive) {
    // zone phase advance
    if (ph < phases.length) {
      const p = phases[ph]
      if (t >= phaseStart + p.hold_s + p.shrink_s) {
        cur = next
        ph++
        phaseStart = t
        next = ph < phases.length ? pickNext(cur, phases[ph].radius) : cur
      }
    }
    const lz = liveZone(t)
    if (dropTimes.length && t >= dropTimes[0]) {
      dropTimes.shift()
      const [dx, dy] = freeSpot(next.cx, next.cy, next.r * (cfg.drops?.spread ?? 0.7))
      drops.push({ id: 20000 + drops.length, x: dx, y: dy, z: world.groundAt(dx, dy), lands: t + (cfg.drops?.fall_s ?? 25), progress: 0, holder: null, called: t })
    }

    if (tick % 2 === 0) {
      // ---- hunters: one at first, one more per kill, all of them late (our guess at "late")
      const aliveOthers = mechs.filter((m) => m.ai && m.alive)
      const dead = mechs.length - 1 - aliveOthers.length
      // --delay N: what if nobody hunted the bot for the first N seconds (upper bound on farming first)
      const wantActive = SCENARIO === "duel" ? 1 : t < HUNT_DELAY ? 0 : t > LATE_T ? 99 : 1 + dead
      let act = aliveOthers.filter((m) => m.ai!.active).length
      for (const m of aliveOthers) {
        if (act >= wantActive) break
        if (!m.ai!.active) {
          m.ai!.active = true
          act++
        }
      }
      const hurry = t > lz.ss - 25
      // the squad shares what any of them sees; out of everyone's sight they only know where we were
      let seenNow = OMNISCIENT
      if (!seenNow) {
        for (const m of aliveOthers) {
          const d = Math.hypot(me.x - m.x, me.y - m.y)
          if (d > cfg.vision.range) continue
          if (d > cfg.vision.near_range && Math.abs(wrap(Math.atan2(me.y - m.y, me.x - m.x) - m.yaw)) > (cfg.vision.fov_deg * Math.PI) / 360) continue
          const ez = m.z + m.stats.eye_height
          if (world.losClear(m.x, m.y, ez, me.x, me.y, me.z + me.stats.height * 0.5) || world.losClear(m.x, m.y, ez, me.x, me.y, me.z + me.stats.height)) {
            seenNow = true
            break
          }
        }
      }
      if (seenNow) {
        believed = [me.x, me.y]
        knownT = t
      }
      const seen = seenNow || t - knownT < 1
      if (seen) seenTicks++
      decTicks++
      const zinfo = { cx: lz.cx, cy: lz.cy, r: lz.r, ncx: next.cx, ncy: next.cy, nr: next.r, hurry }
      if (SCENARIO === "ffa") {
        // free-for-all: every scripted mech goes for the nearest other mech, us included
        for (const m of aliveOthers) {
          m.ai!.active = true
          let tgt: Mech | null = null
          let bd = 260
          for (const o of mechs) {
            if (o === m || !o.alive) continue
            const d = Math.hypot(o.x - m.x, o.y - m.y)
            if (d < bd && (OMNISCIENT || canSee(m, o.x, o.y, o.z + o.stats.height * 0.5, o.z + o.stats.height))) {
              bd = d
              tgt = o
            }
          }
          const ai = m.ai!
          if (tgt) {
            ai.lastTgt = tgt.id
            ai.lastTgtT = t
          } else if (ai.lastTgt >= 0 && t - ai.lastTgtT < 5 && mechs[ai.lastTgt].alive) tgt = mechs[ai.lastTgt]
          m.cmd = ai.decide(m, tgt, t, zinfo, true, believed)
        }
      } else for (const m of aliveOthers) m.cmd = m.ai!.decide(m, me, t, zinfo, seen, believed)

      // ---- our bot's observation
      const eye = me.z + me.stats.eye_height
      const sees = (x: number, y: number, zc0: number, zt: number) => {
        const d = Math.hypot(x - me.x, y - me.y)
        if (d > cfg.vision.range) return false
        if (d > cfg.vision.near_range && Math.abs(wrap(Math.atan2(y - me.y, x - me.x) - me.yaw)) > (cfg.vision.fov_deg * Math.PI) / 360) return false
        return world.losClear(me.x, me.y, eye, x, y, zc0) || world.losClear(me.x, me.y, eye, x, y, zt)
      }
      const sounds: Observation["sounds"] = []
      const obsMechs: Observation["mechs"] = []
      for (const m of mechs) {
        if (m === me || !m.alive) continue
        if (sees(m.x, m.y, m.z + m.stats.height * 0.5, m.z + m.stats.height)) m.memUntil = t + (cfg.vision.memory_s ?? 5)
        if (t <= m.memUntil)
          obsMechs.push({ id: m.id, kind: "mech", pos: [m.x, m.y, m.z], vel: [m.vx, m.vy, m.vz], aim: [m.yaw, m.pitch], hp: m.hp, max_hp: maxHp(m), cubes: m.cubes, weapons: m.weapons.filter((w) => w).map((w) => w!.kind) as never, charging: m.weapons.some((w) => w?.kind === "laser" && w.charge > 0) })
        else if (Math.hypot(m.x - me.x, m.y - me.y) < 40 && Math.hypot(m.vx, m.vy) > 5 && tick % 12 === 0)
          sounds.push({ kind: "step", bearing: Math.atan2(m.y - me.y, m.x - me.x) + gauss() * 0.12, dist: "near", age: 0.05 })
      }
      const item = (l: Loot) => sees(l.x, l.y, l.z + 1, l.z + 2)
      if (pendingOffer && t > pendingOffer.expires) {
        applyUpgrade(me, pendingOffer.choices[0].id)
        events.push({ type: "upgrade", id: pendingOffer.choices[0].id })
        pendingOffer = null
      }
      if (!pendingOffer && me.offers.length) {
        me.offers.shift()
        const held = new Set(me.weapons.filter((w) => w).map((w) => w!.kind))
        const pool = cfg.upgrades.filter((u) => (!u.requires || held.has(u.requires)) && me.upgrades.filter((q) => q === u.id).length < u.max)
        const ch: typeof pool = []
        while (ch.length < 3 && pool.length) ch.push(pool.splice(Math.floor(rnd() * pool.length), 1)[0])
        if (ch.length) pendingOffer = { id: offerId++, choices: ch.map((u) => ({ id: u.id, name: u.name, desc: u.desc })), expires: t + 5 }
      }
      const p = phases[Math.min(ph, phases.length - 1)]
      const obs: Observation = {
        type: "obs", tick, t: Math.round(t * 100) / 100, alive: mechs.filter((m) => m.alive).length,
        self: {
          id: 0, pos: [me.x, me.y, me.z], vel: [me.vx, me.vy, me.vz], aim: [me.yaw, me.pitch], hp: me.hp, max_hp: maxHp(me), energy: me.energy, max_energy: me.stats.energy, grounded: me.grounded,
          cubes: me.cubes, upgrades: me.upgrades, in_zone: Math.hypot(me.x - lz.cx, me.y - lz.cy) <= lz.r,
          weapons: me.weapons.map((w) => (w ? { kind: w.kind, cooldown: Math.max(0, w.cooldown), ...(w.kind === "gun" ? { ammo: w.ammo, reload: Math.max(0, w.reload) } : {}), ...(w.kind === "laser" ? { charge: Math.min(w.charge, w.stats.charge) } : {}), stats: w.stats } : null)) as never,
          stats: me.stats,
        },
        mechs: obsMechs,
        projectiles: bullets.filter((b) => b.by !== 0 && Math.hypot(b.x - me.x, b.y - me.y) < 200).map((b) => ({ id: b.id, kind: b.kind, pos: [b.x, b.y, b.z], vel: [b.vx, b.vy, b.vz] })),
        crates: crates.filter(item).map((l) => ({ id: l.id, pos: [l.x, l.y, l.z], hp: l.hp })),
        cubes: cubes.filter(item).map((l) => ({ id: l.id, pos: [l.x, l.y, l.z] })),
        pickups: pickups.filter(item).map((l) => ({ id: l.id, kind: l.kind as never, pos: [l.x, l.y, l.z] })),
        sounds, beams, events,
        zone: STATIC_ZONE
          ? { center: [cur.cx, cur.cy], radius: cur.r, next_center: [next.cx, next.cy], next_radius: next.r, shrink_start: lz.ss, shrink_end: lz.se, dps: p.dps }
          : { center: [lz.cx, lz.cy], radius: lz.r, next_center: [next.cx, next.cy], next_radius: next.r, shrink_start: lz.ss, shrink_end: lz.se, dps: p.dps },
        offer: pendingOffer,
        drops: drops.map((d) => ({ id: d.id, pos: [d.x, d.y, d.z], radius: cfg.drops?.radius ?? 8, lands_in: Math.max(0, d.lands - t), capture_s: cfg.drops?.capture_s ?? 4, progress: d.progress, holder: d.holder })),
      }
      events = []
      beams = []
      if (laserT < 0 && me.weapons.some((w) => w?.kind === "laser")) laserT = t
      if (laserT < 0 && roofLaserSeenT < 0 && obs.pickups.some((q) => q.kind === "laser" && q.pos[2] - world.groundAt(q.pos[0], q.pos[1]) > 2.5)) roofLaserSeenT = t
      if (cubesAtEnd < 0 && brain.powerEnded) cubesAtEnd = me.cubes
      if (obsMechs.length) {
        conT++
        if (obsMechs.some((o) => world.losClear(me.x, me.y, me.z + 7, o.pos[0], o.pos[1], o.pos[2] + 4.5))) losT++
      }
      const w0 = performance.now()
      let a: BotAction
      try {
        a = brain.act(obs)
      } catch (e) {
        if (err++ < 3) console.error(`seed ${seed} t ${t.toFixed(1)}: act threw`, e)
        a = {}
      }
      times.push(performance.now() - w0)
      modes[brain.mode] = (modes[brain.mode] ?? 0) + 1
      if (TRACE && tick % 20 === 0)
        console.log(`t ${t.toFixed(1)} ${brain.mode.padEnd(11)} pos ${me.x.toFixed(0)},${me.y.toFixed(0)},${me.z.toFixed(0)} hp ${me.hp.toFixed(0)} en ${me.energy.toFixed(0)} cubes ${me.cubes} wp ${me.weapons.map((w) => w?.kind ?? "-").join("+")} see ${obsMechs.length} zone r ${lz.r.toFixed(0)} d ${(Math.hypot(me.x - lz.cx, me.y - lz.cy)).toFixed(0)}`)
      // one-shot parts
      if (a.pick != null && pendingOffer) {
        const ch = pendingOffer.choices[clamp(a.pick | 0, 0, pendingOffer.choices.length - 1)]
        applyUpgrade(me, ch.id)
        events.push({ type: "upgrade", id: ch.id })
        pendingOffer = null
      }
      if (a.take != null) {
        const i = pickups.findIndex((l) => l.id === a.take)
        if (i < 0) events.push({ type: "take_failed", id: a.take, reason: "gone" })
        else {
          const l = pickups[i]
          if (Math.hypot(l.x - me.x, l.y - me.y) > cfg.loot.pickup_radius || Math.abs(l.z - me.z) > 4) events.push({ type: "take_failed", id: l.id, reason: "range" })
          else if (me.weapons.some((w) => w?.kind === l.kind)) events.push({ type: "take_failed", id: l.id, reason: "have" })
          else {
            let slot = me.weapons.findIndex((w) => !w)
            if (slot < 0 && (a.drop === 0 || a.drop === 1)) {
              slot = a.drop
              const old = me.weapons[slot]!
              pickups.push({ id: nextId++, kind: old.kind, x: me.x, y: me.y, z: me.z, hp: 0 })
            }
            if (slot < 0) events.push({ type: "take_failed", id: l.id, reason: "full" })
            else {
              const w = mkWpn(l.kind)
              for (const id of me.upgrades) {
                const u = cfg.upgrades.find((q) => q.id === id)
                for (const e of u?.effects ?? []) {
                  const path = e.path.split(".")
                  if (path[0] === "weapons" && path[1] === l.kind) w.stats[path[2]] = (w.stats[path[2]] ?? 0) * (e.mul ?? 1) + (e.add ?? 0)
                }
              }
              w.cooldown = Math.max(cfg.loot.equip_s, w.stats.cooldown ?? 0)
              me.weapons[slot] = w
              pickups.splice(pickups.indexOf(l), 1)
              events.push({ type: "took", id: l.id, kind: l.kind as never, slot })
            }
          }
        }
      }
      if (a.reload) {
        const g = me.weapons.find((w) => w?.kind === "gun")
        if (g && g.reload <= 0 && g.ammo < g.stats.mag) g.reload = g.stats.reload
      }
      me.cmd = a
      const mv = Math.hypot(me.x - lastPos[0], me.y - lastPos[1])
      travelled += mv
      const wantMove = Math.hypot(a.move?.[0] ?? 0, a.move?.[1] ?? 0) > 0.6
      if (wantMove && mv < 0.15 && (brain.mode === "zone" || brain.mode === "home" || brain.mode.startsWith("get") || brain.mode === "cube" || brain.mode === "king")) {
        if (++stillTicks === 20) {
          stuck++
          if (VERBOSE) console.log(`  seed ${seed} t ${t.toFixed(1)} STUCK at ${me.x.toFixed(1)},${me.y.toFixed(1)},${me.z.toFixed(1)} mode ${brain.mode}`)
        }
      } else stillTicks = 0
      lastPos = [me.x, me.y]

      // ---- rival bots: each gets its own observation and decides exactly as ours does
      for (const [rid, st] of rivals) {
        const o = mechs[rid]
        if (!o.alive) continue
        const list: Observation["mechs"] = []
        for (const m of mechs) {
          if (m === o || !m.alive) continue
          if (canSee(o, m.x, m.y, m.z + m.stats.height * 0.5, m.z + m.stats.height)) st.mem.set(m.id, t + (cfg.vision.memory_s ?? 5))
          if (t <= (st.mem.get(m.id) ?? -1)) list.push({ id: m.id, kind: "mech", pos: [m.x, m.y, m.z], vel: [m.vx, m.vy, m.vz], aim: [m.yaw, m.pitch], hp: m.hp, max_hp: maxHp(m), cubes: m.cubes, weapons: m.weapons.filter((w) => w).map((w) => w!.kind) as never, charging: m.weapons.some((w) => w?.kind === "laser" && w.charge > 0) })
        }
        const oitem = (l: Loot) => canSee(o, l.x, l.y, l.z + 1, l.z + 2)
        if (st.offer && t > st.offer.expires) {
          applyUpgrade(o, st.offer.choices[0].id)
          st.events.push({ type: "upgrade", id: st.offer.choices[0].id })
          st.offer = null
        }
        if (!st.offer && o.offers.length) {
          o.offers.shift()
          const held = new Set(o.weapons.filter((w) => w).map((w) => w!.kind))
          const pool = cfg.upgrades.filter((u) => (!u.requires || held.has(u.requires)) && o.upgrades.filter((q) => q === u.id).length < u.max)
          const ch: typeof pool = []
          while (ch.length < 3 && pool.length) ch.push(pool.splice(Math.floor(rnd() * pool.length), 1)[0])
          if (ch.length) st.offer = { id: offerId++, choices: ch.map((u) => ({ id: u.id, name: u.name, desc: u.desc })), expires: t + 5 }
        }
        const obsR: Observation = {
          type: "obs", tick, t: Math.round(t * 100) / 100, alive: mechs.filter((m) => m.alive).length,
          self: {
            id: rid, pos: [o.x, o.y, o.z], vel: [o.vx, o.vy, o.vz], aim: [o.yaw, o.pitch], hp: o.hp, max_hp: maxHp(o), energy: o.energy, max_energy: o.stats.energy, grounded: o.grounded,
            cubes: o.cubes, upgrades: o.upgrades, in_zone: Math.hypot(o.x - lz.cx, o.y - lz.cy) <= lz.r,
            weapons: o.weapons.map((w) => (w ? { kind: w.kind, cooldown: Math.max(0, w.cooldown), ...(w.kind === "gun" ? { ammo: w.ammo, reload: Math.max(0, w.reload) } : {}), ...(w.kind === "laser" ? { charge: Math.min(w.charge, w.stats.charge) } : {}), stats: w.stats } : null)) as never,
            stats: o.stats,
          },
          mechs: list,
          projectiles: bullets.filter((b) => b.by !== rid && Math.hypot(b.x - o.x, b.y - o.y) < 200).map((b) => ({ id: b.id, kind: b.kind, pos: [b.x, b.y, b.z], vel: [b.vx, b.vy, b.vz] })),
          crates: crates.filter(oitem).map((l) => ({ id: l.id, pos: [l.x, l.y, l.z], hp: l.hp })),
          cubes: cubes.filter(oitem).map((l) => ({ id: l.id, pos: [l.x, l.y, l.z] })),
          pickups: pickups.filter(oitem).map((l) => ({ id: l.id, kind: l.kind as never, pos: [l.x, l.y, l.z] })),
          sounds: [], beams: st.beams, events: st.events,
          zone: { center: [lz.cx, lz.cy], radius: lz.r, next_center: [next.cx, next.cy], next_radius: next.r, shrink_start: lz.ss, shrink_end: lz.se, dps: p.dps },
          offer: st.offer,
          drops: drops.map((d) => ({ id: d.id, pos: [d.x, d.y, d.z], radius: cfg.drops?.radius ?? 8, lands_in: Math.max(0, d.lands - t), capture_s: cfg.drops?.capture_s ?? 4, progress: d.progress, holder: d.holder })),
        }
        st.events = []
        st.beams = []
        let ar: BotAction
        try {
          ar = st.brain.act(obsR)
        } catch (e) {
          if (err++ < 3) console.error(`seed ${seed} t ${t.toFixed(1)}: rival ${rid} threw`, e)
          ar = {}
        }
        if (ar.pick != null && st.offer) {
          const ch = st.offer.choices[clamp(ar.pick | 0, 0, st.offer.choices.length - 1)]
          applyUpgrade(o, ch.id)
          st.events.push({ type: "upgrade", id: ch.id })
          st.offer = null
        }
        doTake(o, ar, st.events)
        if (ar.reload) {
          const g = o.weapons.find((w) => w?.kind === "gun")
          if (g && g.reload <= 0 && g.ammo < g.stats.mag) g.reload = g.stats.reload
        }
        o.cmd = ar
      }
    }

    // ---- simulate one tick
    for (const m of mechs) {
      if (!m.alive) continue
      stepMech(m, t)
      const mzl = m.z + m.stats.muzzle_height
      const ux = Math.cos(m.pitch) * Math.cos(m.yaw), uy = Math.cos(m.pitch) * Math.sin(m.yaw), uz = Math.sin(m.pitch)
      const off = m.stats.radius + 0.5
      for (let s = 0; s < 2; s++) {
        const w = m.weapons[s]
        if (!w) continue
        w.cooldown -= DT
        const trig = !!m.cmd.fire?.[s]
        if (w.kind === "gun") {
          if (w.reload > 0) {
            w.reload -= DT
            if (w.reload <= 0) w.ammo = w.stats.mag
          } else if (trig && w.cooldown <= 0 && w.ammo > 0) {
            const sp = (w.stats.spread_deg * Math.PI) / 180
            const yy = m.yaw + gauss() * sp, pp = m.pitch + gauss() * sp
            const bx = Math.cos(pp) * Math.cos(yy), by = Math.cos(pp) * Math.sin(yy), bz = Math.sin(pp)
            bullets.push({ id: nextId++, kind: "gun", by: m.id, x: m.x + ux * off, y: m.y + uy * off, z: mzl + uz * off, vx: bx * w.stats.speed, vy: by * w.stats.speed, vz: bz * w.stats.speed, left: w.stats.range, dmg: w.stats.damage * cubeMult(m) })
            w.ammo--
            if (m.id === 0) gs++
            w.cooldown = w.stats.cooldown
            if (w.ammo <= 0) w.reload = w.stats.reload
          }
        } else if (w.kind === "laser") {
          let fireNow = false
          let auto = false
          if (trig && w.cooldown <= 0) {
            w.charge += DT
            if (w.charge >= w.stats.charge + w.stats.hold_s) {
              fireNow = true
              auto = true
            }
          } else if (w.charge > 0) {
            if (w.charge >= w.stats.min_charge) fireNow = true
            else w.charge += DT
          }
          if (fireNow) {
            const frac = w.stats.min_frac + (1 - w.stats.min_frac) * Math.min(1, w.charge / w.stats.charge)
            if (m.id === 0) ls++
            const [v, dd] = rayHit(m, m.x, m.y, mzl, ux, uy, uz, w.stats.range)
            // our bot sees a beam when either end is in view; one that hits it always is
            {
              const ex = m.x + ux * dd, ey = m.y + uy * dd, ezz = mzl + uz * dd
              const beam = { by: m.id, kind: "laser" as const, path: [[m.x, m.y, mzl], [ex, ey, ezz]] as [number, number, number][], age: 0.05 }
              if (m.id !== 0 && me.alive && (v === me || canSee(me, m.x, m.y, mzl, mzl) || canSee(me, ex, ey, ezz, ezz))) beams.push(beam)
              for (const [rid, st] of rivals) {
                const o = mechs[rid]
                if (rid !== m.id && o.alive && (v === o || canSee(o, m.x, m.y, mzl, mzl) || canSee(o, ex, ey, ezz, ezz))) st.beams.push(beam)
              }
            }
            const clearShot = !!v && world.losClear(m.x, m.y, mzl, m.x + ux * dd, m.y + uy * dd, mzl + uz * dd)
            // a beam that meets no mech can still break a crate
            let crateHit = false
            if (!v) {
              const hl = Math.hypot(ux, uy)
              for (const c of crates) {
                const rx = c.x - m.x, ry = c.y - m.y
                const along = hl > 1e-6 ? (rx * ux + ry * uy) / hl : -1
                if (along <= 0 || along / hl > w.stats.range) continue
                if (Math.abs(rx * uy - ry * ux) / hl > (cfg.crates.radius ?? 1.4)) continue
                const hz = mzl + uz * (along / hl)
                if (hz < c.z || hz > c.z + (cfg.crates.height ?? 2.4)) continue
                if (!world.losClear(m.x, m.y, mzl, c.x, c.y, hz)) continue
                c.hp -= w.stats.damage * frac * cubeMult(m)
                if (m.id === 0) events.push({ type: "hit", target: c.id, amount: w.stats.damage * frac, weapon: "laser" })
                crateHit = true
                break
              }
            }
            if (m.id === 0 && !crateHit) {
              // why our laser shots miss: fired itself after the hold ran out, wall in the way, or off aim
              if (auto) {
                lz2.auto++
                if (clearShot) lz2.autoHit++
              } else {
                lz2.rel++
                if (clearShot) lz2.relHit++
                else if (v) lz2.relBlocked++
                else lz2.relOffAim++
              }
            }
            if (v && clearShot) {
              let dmg = w.stats.damage * frac * cubeMult(m)
              if (v.weapons.some((q) => q?.kind === "mortar")) dmg *= 1 + w.stats.cookoff
              hurt(v, dmg, "laser", m, m.x, m.y, t)
            }
            w.charge = 0
            w.cooldown = w.stats.cooldown
          }
        } else if (w.kind === "lightning") {
          if (trig && w.cooldown <= 0 && m.energy >= w.stats.energy) {
            m.energy -= w.stats.energy
            m.lastSpendT = t
            w.cooldown = w.stats.cooldown
            let best: Mech | null = null
            let bd = w.stats.range
            for (const v of mechs) {
              if (!v.alive || v === m) continue
              const d = Math.hypot(v.x - m.x, v.y - m.y, v.z - m.z)
              if (d > bd || Math.abs(wrap(Math.atan2(v.y - m.y, v.x - m.x) - m.yaw)) > (w.stats.cone_deg * Math.PI) / 180) continue
              if (!world.losClear(m.x, m.y, mzl, v.x, v.y, v.z + v.stats.height * 0.5)) continue
              bd = d
              best = v
            }
            if (best) hurt(best, w.stats.damage * cubeMult(m), "lightning", m, m.x, m.y, t)
          }
        } else if (w.kind === "mortar") {
          if (trig && w.cooldown <= 0) {
            w.cooldown = w.stats.cooldown
            bullets.push({ id: nextId++, kind: "mortar", by: m.id, x: m.x + ux * off, y: m.y + uy * off, z: m.z + m.stats.height + 0.5 + uz * off, vx: ux * w.stats.speed, vy: uy * w.stats.speed, vz: uz * w.stats.speed, left: w.stats.ttl * w.stats.speed, dmg: w.stats.splash * cubeMult(m) })
          }
        }
      }
      // zone, regen, cubes
      if (Math.hypot(m.x - lz.cx, m.y - lz.cy) > lz.r) hurt(m, lz.dps * DT, "zone", null, 0, 0, t)
      else if (t - m.lastCombatT >= m.stats.regen_delay) m.hp = Math.min(maxHp(m), m.hp + m.stats.regen_per_s * DT)
      for (let i = cubes.length - 1; i >= 0; i--) {
        const c = cubes[i]
        if (Math.hypot(c.x - m.x, c.y - m.y) <= Number(cfg.cubes.pickup_radius) && Math.abs(c.z - m.z) <= 4) {
          cubes.splice(i, 1)
          giveCube(m, t)
        }
      }
    }
    // bullets
    for (let i = bullets.length - 1; i >= 0; i--) {
      const b = bullets[i]
      const by = mechs[b.by]
      if (b.kind === "mortar") b.vz -= cfg.weapons.mortar.gravity * DT
      const nx = b.x + b.vx * DT, ny = b.y + b.vy * DT, nz = b.z + b.vz * DT
      const sp = Math.hypot(b.vx, b.vy, b.vz)
      let gone = false
      if (b.kind === "gun") {
        const [v, dd] = rayHit(by, b.x, b.y, b.z, b.vx / sp, b.vy / sp, b.vz / sp, sp * DT)
        if (v) {
          hurt(v, b.dmg, "gun", by, by.x, by.y, t)
          gone = true
        } else {
          for (const c of crates) {
            if (Math.hypot(c.x - nx, c.y - ny) < (cfg.crates.radius ?? 1.4) + 0.5 && nz > c.z - 0.5 && nz < c.z + (cfg.crates.height ?? 2.4) + 0.5) {
              c.hp -= b.dmg
              if (b.by === 0) events.push({ type: "hit", target: c.id, amount: b.dmg, weapon: "gun" })
              gone = true
              break
            }
          }
        }
        void dd
      }
      if (!gone && nz < world.topAt(nx, ny)) {
        gone = true
        if (b.kind === "mortar") {
          const R = cfg.weapons.mortar.splash_radius
          for (const v of mechs) {
            if (!v.alive) continue
            const d = Math.hypot(v.x - nx, v.y - ny)
            if (d > R + v.stats.radius) continue
            const f = 1 - (1 - cfg.weapons.mortar.splash_edge) * clamp(d / R, 0, 1)
            hurt(v, b.dmg * f * (v === by ? M.self_damage_mult : 1), "mortar", by, nx, ny, t)
          }
        }
      }
      b.x = nx
      b.y = ny
      b.z = nz
      b.left -= sp * DT
      if (gone || b.left <= 0) bullets.splice(i, 1)
    }
    for (let i = crates.length - 1; i >= 0; i--) {
      if (crates[i].hp <= 0) {
        const c = crates[i]
        crates.splice(i, 1)
        cubes.push({ id: nextId++, kind: "cube", x: c.x, y: c.y, z: c.z, hp: 0 })
        if (rnd() < cfg.loot.crate_weapon_chance) pickups.push({ id: nextId++, kind: kinds[Math.floor(rnd() * 3)], x: c.x + 1, y: c.y, z: c.z, hp: 0 })
      }
    }
    // supply drops
    for (let i = drops.length - 1; i >= 0; i--) {
      const d = drops[i]
      if (t < d.lands) continue
      const inside = mechs.filter((m) => m.alive && Math.hypot(m.x - d.x, m.y - d.y) <= (cfg.drops?.radius ?? 8))
      if (inside.length === 1) {
        if (d.holder !== inside[0].id) {
          d.holder = inside[0].id
          d.progress = 0
        }
        d.progress += DT
        if (d.progress >= (cfg.drops?.capture_s ?? 4)) {
          const m = inside[0]
          for (let k = 0; k < (cfg.drops?.cubes ?? 3); k++) giveCube(m, t)
          m.hp = maxHp(m)
          m.energy = m.stats.energy
          if (m.id === 0) events.push({ type: "drop", id: d.id, cubes: m.cubes })
          drops.splice(i, 1)
        }
      } else if (inside.length === 0) {
        d.holder = null
        d.progress = 0
      }
    }
    if (!me.alive) {
      deadBeforeMe = mechs.filter((m) => m !== me && !m.alive).length
      activeEnd = mechs.filter((m) => m.ai && m.alive && m.ai.active).length
    }
    t += DT
    tick++
    if (mechs.filter((m) => m.alive).length <= 1 && SCENARIO !== "nav") break
  }
  if (me.alive) deadBeforeMe = SCENARIO === "nav" ? 0 : mechs.filter((m) => m !== me && !m.alive).length
  times.sort((a, b) => a - b)
  return {
    score: deadBeforeMe, t, hp: Math.max(0, me.hp), kills, dealt, taken, zoneDmg, cubes: me.cubes, weapons: me.weapons.map((w) => w?.kind ?? "-").join("+"), stuck,
    msMax: times[times.length - 1] ?? 0, msP99: times[Math.floor(times.length * 0.99)] ?? 0, err, modes, travelled, seen: decTicks ? seenTicks / decTicks : 0, laserT, gs, gh, ls, lh, losT, conT, lz2, src, activeEnd, killT, pend: brain.powerEnded, cubesAtEnd, roofLaserSeenT,
  }
}

// -------------------------------------------------------------------------------------------- run
const all: Result[] = []
for (let k = 0; k < MATCHES; k++) {
  const seed = SEED0 + k
  const r = runMatch(seed)
  all.push(r)
  const top = Object.entries(r.modes).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([m, n]) => `${m}:${n}`).join(" ")
  console.log(
    `seed ${String(seed).padStart(3)} | outlived ${String(r.score).padStart(2)} | died/ended t=${r.t.toFixed(0).padStart(3)}s hp ${r.hp.toFixed(0).padStart(4)} | kills ${r.kills} dealt ${r.dealt.toFixed(0)} taken ${r.taken.toFixed(0)} zone ${r.zoneDmg.toFixed(0)} | cubes ${r.cubes} ${r.weapons} | stuck ${r.stuck} | ms p99 ${r.msP99.toFixed(1)} max ${r.msMax.toFixed(1)}${r.err ? ` | ERRORS ${r.err}` : ""} | ${top}`,
  )
}
const mean = (f: (r: Result) => number) => all.reduce((s, r) => s + f(r), 0) / Math.max(1, all.length)
const sum = (f: (r: Result) => number) => all.reduce((a, r) => a + f(r), 0)
const pct = (a: number, b: number) => ((100 * a) / Math.max(1, b)).toFixed(0)
console.log(
  `shooting: gun ${sum((r) => r.gh)}/${sum((r) => r.gs)} hits (${pct(sum((r) => r.gh), sum((r) => r.gs))}%), ${(sum((r) => r.gs) / all.length).toFixed(0)} shots a match | laser ${sum((r) => r.lh)}/${sum((r) => r.ls)} hits (${pct(sum((r) => r.lh), sum((r) => r.ls))}%), ${(sum((r) => r.ls) / all.length).toFixed(1)} shots a match | enemy listed ${(sum((r) => r.conT) / all.length / 10).toFixed(0)}s a match, clear line ${pct(sum((r) => r.losT), sum((r) => r.conT))}% of that`,
)
{
  const tot: Record<string, number> = {}
  for (const r of all) for (const [k, v] of Object.entries(r.src)) tot[k] = (tot[k] ?? 0) + v
  const allDmg = Object.entries(tot).filter(([k]) => k !== 'rear').reduce((a, [, v]) => a + v, 0)
  console.log(`damage taken by source: ${Object.entries(tot).filter(([k]) => k !== 'rear').sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} ${pct(v, allDmg)}%`).join(', ')} | from behind ${pct(tot.rear ?? 0, allDmg)}% | hunters on us at death ${mean((r) => r.activeEnd).toFixed(1)} | kill times ${[0, 1, 2, 3, 4, 5].map((i) => { const v = all.filter((r) => r.killT.length > i).map((r) => r.killT[i]); return v.length ? `#${i + 1}@${(v.reduce((a, b) => a + b, 0) / v.length).toFixed(0)}s(${v.length})` : '' }).join(' ')}`)
}
{
  const why: Record<string, number[]> = {}
  for (const r of all) {
    const [k, at] = (r.pend || 'never').split('@')
    ;(why[k] ??= []).push(parseFloat(at) || 0)
  }
  console.log(`build-up phase ended: ${Object.entries(why).map(([k, v]) => `${k} x${v.length} (avg ${(v.reduce((a, b) => a + b, 0) / v.length).toFixed(0)}s)`).join(', ')} | cubes when it ended ${mean((r) => Math.max(0, r.cubesAtEnd)).toFixed(1)}`)
}
console.log(`roof lasers: seen from the ground while we had no laser in ${pct(all.filter((r) => r.roofLaserSeenT >= 0).length, all.length)}% of matches (first seen at ${(all.filter((r) => r.roofLaserSeenT >= 0).reduce((a, r) => a + r.roofLaserSeenT, 0) / Math.max(1, all.filter((r) => r.roofLaserSeenT >= 0).length)).toFixed(0)}s on average); of the ${all.filter((r) => r.laserT < 0).length} matches that never got a laser, ${all.filter((r) => r.laserT < 0 && r.roofLaserSeenT >= 0).length} had seen one on a roof`)
console.log(`laser detail: released ${sum((r) => r.lz2.rel)} (hit ${sum((r) => r.lz2.relHit)}, wall in the way ${sum((r) => r.lz2.relBlocked)}, off aim ${sum((r) => r.lz2.relOffAim)}) | fired itself after the hold ${sum((r) => r.lz2.auto)} (hit ${sum((r) => r.lz2.autoHit)})`)
console.log(
  `\n${SCENARIO}: ${all.length} matches | mean outlived ${mean((r) => r.score).toFixed(2)} | mean survival ${mean((r) => r.t).toFixed(0)}s | kills ${mean((r) => r.kills).toFixed(2)} | dealt ${mean((r) => r.dealt).toFixed(0)} taken ${mean((r) => r.taken).toFixed(0)} zone dmg ${mean((r) => r.zoneDmg).toFixed(0)} | cubes ${mean((r) => r.cubes).toFixed(1)} | laser ${(100 * mean((r) => (r.laserT >= 0 ? 1 : 0))).toFixed(0)}% @${(all.filter((r) => r.laserT >= 0).reduce((a, r) => a + r.laserT, 0) / Math.max(1, all.filter((r) => r.laserT >= 0).length)).toFixed(0)}s | in sight ${(100 * mean((r) => r.seen)).toFixed(0)}% | stuck ${mean((r) => r.stuck).toFixed(2)} | worst decision ${Math.max(...all.map((r) => r.msMax)).toFixed(1)} ms | errors ${all.reduce((s, r) => s + r.err, 0)}`,
)
