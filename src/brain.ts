/**
 * The bot's decisions. Each observation: update memory, decide a mode (zone / final point / evade /
 * fight / loot / hold), pick a place to go, steer there while dodging and using cover, then aim and fire.
 */
import type { BotAction, Config, Hello, Observation, Vec3, WeaponObs } from "../mechfight.ts"
import { ballisticPitch, clamp, interceptTime, mulberry32, TAU, wrap } from "./aim.ts"
import { Nav } from "./nav.ts"
import { World } from "./world.ts"

type Track = {
  id: number
  kind: "mech" | "npc"
  x: number
  y: number
  z: number
  vx: number
  vy: number
  vz: number
  yaw: number
  hp: number
  maxHp: number
  cubes: number
  weapons: string[]
  charging: boolean
  /** match time of the last observation that listed it */
  seen: number
  r: number
  h: number
  /** position is a guess (from a hit's bearing or a beam), not a sighting */
  phantom: boolean
}
type Item ={ id: number; x: number; y: number; z: number; kind: string; hp: number; seen: number; roof: boolean }
type Aimable = { id: number; kind: "mech" | "npc" | "crate"; x: number; y: number; z: number; vx: number; vy: number; vz: number; r: number; h: number; hp: number; maxHp: number; weapons: string[]; aimFrac?: number }
type Goal = { x: number; y: number; arrive: number; why: string; boost: boolean; travel: boolean }
type Circle = { cx: number; cy: number; r: number }
type Ctx = {
  x: number
  y: number
  z: number
  t: number
  radius: number
  height: number
  mh: number
  hpFrac: number
  walk: number
  zone: Circle
  insideNext: boolean
  mustGo: boolean
  king: boolean
  evade: boolean
  fight: boolean
  /** hide-first posture (free-for-all) rather than take-the-fight posture (practice squad) */
  ambush: boolean
  /** building up before the first kill: no shooting at mechs, keep away from them, farm */
  power: boolean
  /** sniper posture: we hold a laser, so fight from a corner at range and let them come to us */
  camp: boolean
  /** fight-from-cover mechanics are on (either of the two postures above) */
  cover: boolean
  threats: Track[]
  target: Aimable | null
  targetLos: boolean
  targetDist: number
  wantLos: number
  sGun: number
  sLaser: number
  sLight: number
  sMortar: number
  exposed: boolean
}

/** how much we like holding each weapon (the gun is never dropped) */
const WEAPON_VALUE: Record<string, number> = { laser: 4, lightning: 2.4, gun: 2, mortar: 1.2 }
/** upgrade preference, best first: survive the final burn, then win duels */
const UPGRADE_ORDER = [
  "plating", "nanites", "armor", "servos", "laser_damage", "gun_damage", "jets", "reactor", "capacitor", "gun_rate", "laser_charge",
  "lightning_efficiency", "lightning_range", "lightning_chain", "gun_mag", "gun_velocity", "gyros", "sensors", "laser_pierce",
  "mortar_reload", "mortar_splash", "mortar_velocity",
]
const CAMP_DEFAULT = "0"
/** numeric overrides for local experiments: MECH_TUNE='{"lmin":100}' */
const TUNE: Record<string, number> = (() => {
  try {
    return JSON.parse((typeof process !== "undefined" && process.env?.MECH_TUNE) || "{}")
  } catch {
    return {}
  }
})()
/** overrides of the instance that is deciding right now (the simulator runs several bots in one process) */
let CUR: Record<string, number> | null = null
const tv = (k: string, d: number) => (CUR && typeof CUR[k] === "number" ? CUR[k] : typeof TUNE[k] === "number" ? TUNE[k] : d)
const envFlag = (name: string, dflt: string) => ((typeof process !== "undefined" && process.env?.[name]) || dflt) === "1"
/** farm before fighting (practice only); MECH_POWER=0/1 overrides for local testing */
const POWER = envFlag("MECH_POWER", "0")
/** rehearse decisions before the match so the runtime compiles them early; MECH_REHEARSE=0/1 */
const REHEARSE = envFlag("MECH_REHEARSE", "1")
/** go for a supply drop even with mechs in contact; MECH_DROP=0/1 overrides for local testing */
const DROP_UNDER_FIRE = envFlag("MECH_DROP", "0")
/** keep charging the laser while running from several mechs; MECH_RUNCHARGE=0/1 overrides for local testing */
const CHARGE_ON_RUN = ((typeof process !== "undefined" && process.env?.MECH_RUNCHARGE) || "1") === "1"
/** "auto" picks the posture from what the match looks like; the others force one (for local testing) */
const STANCE = (typeof process !== "undefined" && process.env?.MECH_STANCE) || "auto"
/** sniper posture when holding a laser; MECH_CAMP=0/1 overrides for local testing */
const CAMP = ((typeof process !== "undefined" && process.env?.MECH_CAMP) || CAMP_DEFAULT) === "1"
/** alternative upgrade orders tried in sparring (tune "upg") */
const UPGRADE_ALT: string[][] = [
  UPGRADE_ORDER,
  // 1: damage first
  ["laser_damage", "gun_damage", "plating", "gun_rate", "armor", "laser_charge", "gun_mag", "nanites", "servos", "jets", "reactor", "capacitor", "gun_velocity"],
  // 2: toughness then damage, no repair or speed early
  ["plating", "armor", "laser_damage", "gun_damage", "gun_rate", "gun_mag", "laser_charge", "nanites", "servos", "jets", "reactor", "capacitor", "gun_velocity"],
  // 3: toughness, damage and magazine interleaved
  ["plating", "laser_damage", "armor", "gun_mag", "gun_damage", "gun_rate", "nanites", "laser_charge", "servos", "jets", "reactor", "capacitor", "gun_velocity"],
]
const NDIR = 16
const DIRX = new Float64Array(NDIR)
const DIRY = new Float64Array(NDIR)
for (let i = 0; i < NDIR; i++) {
  DIRX[i] = Math.cos((i * TAU) / NDIR)
  DIRY[i] = Math.sin((i * TAU) / NDIR)
}
const dist2 = (ax: number, ay: number, bx: number, by: number) => Math.hypot(ax - bx, ay - by)

export class Brain {
  name = "magic-royale"
  cfg: Config
  world: World
  nav: Nav
  /** last mode, for debugging / the harness */
  mode = "init"
  private rng: () => number
  private tracks = new Map<number, Track>()
  private pickups = new Map<number, Item>()
  private crates = new Map<number, Item>()
  private cubes = new Map<number, Item>()
  /** pickup id -> match time until which we leave it alone */
  private ignore = new Map<number, number>()

  private path: number[] = []
  private pathI = 0
  private pathGX = NaN
  private pathGY = NaN
  private pathT = -1e9
  private pathPartial = false
  private replan = false

  private hx = new Float64Array(12)
  private hy = new Float64Array(12)
  private hWant = new Uint8Array(12)
  private hN = 0
  private stuckCount = 0
  private hopUntil = -1
  private lastStuckT = -1e9

  private targetId = -1
  private strafe = 1
  private strafeFlipT = 0
  private evading = false
  private coverX = NaN
  private coverY = NaN
  private coverT = -1e9
  private coverPeek = false
  /** the match looks like a free-for-all (mechs are dying that we did not kill) */
  ffa = false
  private myKills = 0
  private dropsTaken = 0
  private chaseId = -1
  private chaseBest = Infinity
  private chaseT = 0
  private alertYaw = 0
  private alertUntil = -1
  private lastHurtT = -1e9
  private lastHurtBy = -1
  /** last time a mech (not a drone or the zone) damaged us */
  private lastMechHurtT = -1e9
  private lastThreatT = -1e9
  /** jet crossing out of (or into) a walled-off area: plan, and whether we are in the air on it */
  private hopPlanT = -1e9
  private hopPlan: { lx: number; ly: number; tx: number; ty: number; energy: number; rise: number } | null = null
  private hopTarget = -1
  private hopFly = false
  private hopFlyT = 0
  private hopFails = 0
  private hopBan = -1e9
  /** the rooftop laser we are going up for */
  private roofItem: Item | null = null
  private scanStartT = 0
  private scanUntil = 1.8
  private scanYaw0 = NaN
  private nextScanT = 12
  private lastDX = 0
  private lastDY = 0
  private idleX = NaN
  private idleY = NaN
  private idleKey = ""
  private zKey = NaN
  private zHoldR = NaN
  private zSame = 0
  private zStatic = false
  private lastSawTargetT = -1e9
  /** still building up: avoiding fights and farming (see updatePower) */
  power = true
  powerEnded = ""
  /** mech id -> last time it damaged us */
  private hurtBy = new Map<number, number>()
  private visitT = new Float32Array(0)
  private exploreX = NaN
  private exploreY = NaN
  private exploreT = -1e9

  private stance: string | null
  private tuneOv: Record<string, number> | null

  constructor(hello: Hello, shared?: { world: World; nav: Nav }, opts?: { stance?: string; tune?: Record<string, number> }) {
    this.stance = opts?.stance ?? null
    this.tuneOv = opts?.tune ?? null
    this.cfg = hello.config
    this.world = shared?.world ?? new World(hello.map.dir, hello.map.files)
    this.nav = shared?.nav ?? new Nav(this.world)
    this.rng = mulberry32((hello.seed ^ 0x9e3779b9) >>> 0)
    if (!shared) {
      this.warmUp()
      try {
        if (REHEARSE) this.rehearse(hello)
      } catch {
        // rehearsal is only for speed; never let it stop the bot starting
      }
      // the throwaway copy shared our navigator: drop anything it marked as blocked
      this.nav.clearBlocks()
      this.nav.now = 0
    }
  }

  /**
   * Play a few hundred made-up decisions on a throwaway copy of ourselves before the match starts.
   * With one CPU, the runtime compiling the decision code mid-match shows up as a slow answer;
   * doing it here spends the start-up budget instead.
   */
  private rehearse(hello: Hello) {
    const w = this.world
    const cfg = this.cfg
    const t0 = performance.now()
    const rnd = mulberry32(777)
    const spot = (): [number, number] => {
      for (let k = 0; k < 50; k++) {
        const px = w.bMinX + 60 + rnd() * (w.bMaxX - w.bMinX - 120)
        const py = w.bMinY + 60 + rnd() * (w.bMaxY - w.bMinY - 120)
        if (this.nav.clearAt(px, py) > 4) return [px, py]
      }
      return [cfg.zone.start_center[0], cfg.zone.start_center[1]]
    }
    const kinds = ["laser", "lightning", "mortar"] as const
    const wstats = cfg.weapons as unknown as Record<string, Record<string, number>>
    const ph = cfg.zone.phases[0]
    for (let round = 0; round < 6 && performance.now() - t0 < 2500; round++) {
      const twin = new Brain(hello, { world: w, nav: this.nav })
      let [x, y] = spot()
      const second = round === 0 ? null : kinds[round % 3]
      const foes = Array.from({ length: round % 4 }, (_, i) => ({ id: 1 + i, p: spot(), w: kinds[(round + i) % 3] }))
      for (const f of foes) {
        const a = rnd() * TAU
        const r = 40 + rnd() * 150
        f.p = [x + Math.cos(a) * r, y + Math.sin(a) * r]
      }
      for (let i = 0; i < 70 && performance.now() - t0 < 2500; i++) {
        const t = i * 0.1
        const z = w.groundAt(x, y)
        const hp = cfg.mech.hp * (round % 2 ? 0.3 + 0.7 * rnd() : 1)
        const obs = {
          type: "obs", tick: i * 2, t, alive: 16 - (round > 3 ? 4 : 0),
          self: {
            id: 0, pos: [x, y, z], vel: [rnd() * 10 - 5, rnd() * 10 - 5, 0], aim: [rnd() * 6 - 3, 0], hp, max_hp: cfg.mech.hp, energy: 100 * rnd(), max_energy: cfg.mech.energy,
            grounded: true, cubes: round, upgrades: [], in_zone: round !== 5,
            weapons: [
              { kind: "gun", cooldown: 0, ammo: i % 13, reload: 0, stats: wstats.gun },
              second ? { kind: second, cooldown: 0, ...(second === "laser" ? { charge: (i % 9) * 0.1 } : {}), stats: wstats[second] } : null,
            ],
            stats: cfg.mech,
          },
          mechs: foes.map((f) => ({ id: f.id, kind: "mech", pos: [f.p[0], f.p[1], w.groundAt(f.p[0], f.p[1])], vel: [rnd() * 20 - 10, rnd() * 20 - 10, 0], aim: [rnd() * 6 - 3, 0], hp: 300 + rnd() * 1100, max_hp: 1400, cubes: 0, weapons: ["gun", f.w], charging: i % 7 === 0 })),
          projectiles: foes.slice(0, 2).map((f, k) => ({ id: 900 + k, kind: k ? "mortar" : "gun", pos: [f.p[0], f.p[1], z + 7], vel: [(x - f.p[0]) * 1.5, (y - f.p[1]) * 1.5, k ? 20 : 0] })),
          crates: [{ id: 1000, pos: [x + 30, y + 10, w.groundAt(x + 30, y + 10)], hp: 300 }],
          cubes: i % 3 ? [{ id: 5000, pos: [x - 20, y + 15, w.groundAt(x - 20, y + 15)] }] : [],
          pickups: [{ id: 8000, kind: kinds[i % 3], pos: [x + 40, y - 30, w.groundAt(x + 40, y - 30)] }],
          sounds: i % 5 ? [] : [{ kind: "step", bearing: 1, dist: "near", age: 0.05 }],
          beams: [],
          events: i % 6 || !foes.length ? [] : [{ type: "damaged", amount: 45, weapon: "gun", bearing: rnd() * 6 - 3, by: foes[0].id }],
          zone: { center: cfg.zone.start_center, radius: cfg.zone.start_radius, next_center: [cfg.zone.start_center[0] - 60, cfg.zone.start_center[1] + 60], next_radius: ph.radius, shrink_start: ph.hold_s, shrink_end: ph.hold_s + ph.shrink_s, dps: ph.dps },
          offer: i === 10 ? { id: 1, expires: t + 5, choices: cfg.upgrades.slice(0, 3).map((u) => ({ id: u.id, name: u.name, desc: u.desc })) } : null,
          drops: round === 4 ? [{ id: 20000, pos: [x + 50, y + 50, z], radius: 8, lands_in: 5, capture_s: 4, progress: 0, holder: null }] : [],
        } as unknown as Observation
        const a = twin.act(obs)
        const mv = a.move ?? [0, 0]
        const nx = x + mv[0] * 1.1
        const ny = y + mv[1] * 1.1
        if (this.nav.clearAt(nx, ny) > 3) {
          x = nx
          y = ny
        }
      }
    }
  }

  /** run the hot paths a few times so the JIT has compiled them before the first real decision */
  private warmUp() {
    const w = this.world
    const t0 = performance.now()
    const rnd = mulberry32(12345)
    const px = () => w.bMinX + 40 + rnd() * (w.bMaxX - w.bMinX - 80)
    const py = () => w.bMinY + 40 + rnd() * (w.bMaxY - w.bMinY - 80)
    for (let k = 0; k < 40 && performance.now() - t0 < 1200; k++) {
      const ax = px()
      const ay = py()
      const bx = px()
      const by = py()
      this.nav.findPath(ax, ay, bx, by)
      for (let i = 0; i < 300; i++) w.losClear(ax, ay, w.groundAt(ax, ay) + 7, px(), py(), 5)
      this.nav.segOk(ax, ay, ax + 10, ay + 10)
    }
    // also the planner for jetting out of walled-off areas, so its first real use is not a slow one
    const nav = this.nav
    const seen = new Set<number>([nav.mainComp])
    for (let cc = 0; cc < nav.comp.length && seen.size < 6 && performance.now() - t0 < 1500; cc += 7) {
      const id = nav.comp[cc]
      if (id < 0 || seen.has(id)) continue
      seen.add(id)
      for (let k = 0; k < 3; k++) nav.hopPlan(nav.cellX(cc), nav.cellY(cc), nav.mainComp)
    }
  }

  // ------------------------------------------------------------------------------------------ main
  act(obs: Observation): BotAction {
    CUR = this.tuneOv
    const me = obs.self
    const S = me.stats
    const radius = S.radius ?? this.cfg.mech.radius ?? 2.5
    const height = S.height ?? this.cfg.mech.height ?? 9
    this.nav.now = obs.t
    this.nav.need = radius + 0.55
    this.nav.clear = height > 10.8 ? this.world.clearTall : this.world.clearLow
    this.remember(obs)
    const out: BotAction = {}
    if (obs.offer) out.pick = this.pickUpgrade(obs)
    const c = this.assess(obs, radius, height)
    const goal = this.chooseGoal(obs, c)
    this.mode = goal.why
    if (this.hop(obs, c, goal, out)) this.mode = "hop"
    else this.move(obs, c, goal, out)
    this.aimAndFire(obs, c, out)
    this.loot(obs, c, out)
    return out
  }

  // ---------------------------------------------------------------------------------------- memory
  private remember(obs: Observation) {
    const t = obs.t
    const me = obs.self
    const [x, y, z] = me.pos
    for (const m of obs.mechs) {
      const cubes = m.cubes ?? 0
      const scale = 1 + (Number(this.cfg.cubes.size_per_cube) || 0) * cubes
      const npc = m.kind === "npc"
      this.tracks.set(m.id, {
        id: m.id,
        kind: m.kind,
        x: m.pos[0],
        y: m.pos[1],
        z: m.pos[2],
        vx: m.vel[0],
        vy: m.vel[1],
        vz: m.vel[2],
        yaw: m.aim[0],
        hp: m.hp ?? m.max_hp ?? 1400,
        maxHp: m.max_hp ?? 1400,
        cubes,
        weapons: m.weapons ?? [],
        charging: !!m.charging,
        seen: t,
        r: npc ? (this.cfg.npc.radius ?? 1.8) : (this.cfg.mech.radius ?? 2.5) * scale,
        h: npc ? (this.cfg.npc.height ?? 3.5) : (this.cfg.mech.height ?? 9) * scale,
        phantom: false,
      })
    }
    // a mech we cannot see gave itself away: place it from the beam's origin or the hit's bearing
    const guess = (id: number, px: number, py: number, weapon: string, fresh: number) => {
      const k = this.tracks.get(id)
      if (k && t - k.seen <= fresh) return
      const weapons = k?.weapons.length ? (k.weapons.includes(weapon) ? k.weapons : [...k.weapons, weapon]) : [weapon]
      this.tracks.set(id, {
        id, kind: "mech", x: px, y: py, z: this.world.groundAt(px, py), vx: 0, vy: 0, vz: 0, yaw: Math.atan2(y - py, x - px),
        hp: k?.hp ?? 1400, maxHp: k?.maxHp ?? 1400, cubes: k?.cubes ?? 0, weapons, charging: false, seen: t,
        r: this.cfg.mech.radius ?? 2.5, h: this.cfg.mech.height ?? 9, phantom: true,
      })
    }
    for (const b of obs.beams ?? []) {
      if (b.by === me.id || !b.path?.length) continue
      guess(b.by, b.path[0][0], b.path[0][1], b.kind, 0.5)
    }
    for (const [id, k] of this.tracks) if (t - k.seen > 25) this.tracks.delete(id)

    const eyeZ = z + (me.stats.eye_height ?? 8)
    const refresh = (mem: Map<number, Item>, list: { id: number; pos: Vec3; hp?: number; kind?: string }[], ttl: number) => {
      const now = new Set<number>()
      for (const it of list) {
        now.add(it.id)
        const roof = it.pos[2] - this.world.groundAt(it.pos[0], it.pos[1]) > 2.5
        mem.set(it.id, { id: it.id, x: it.pos[0], y: it.pos[1], z: it.pos[2], kind: it.kind ?? "", hp: it.hp ?? 0, seen: t, roof })
      }
      for (const [id, it] of mem) {
        if (now.has(id)) continue
        // forget what is no longer there: we are close with a clear view and it is not listed
        const d = dist2(x, y, it.x, it.y)
        if (t - it.seen > ttl || (d < 38 && t - it.seen > 0.4 && this.world.losClear(x, y, eyeZ, it.x, it.y, it.z + 1))) mem.delete(id)
      }
    }
    refresh(this.pickups, obs.pickups, 150)
    refresh(this.crates, obs.crates, 150)
    refresh(this.cubes, obs.cubes, 60)

    for (const e of obs.events) {
      if (e.type === "damaged") {
        if (e.weapon !== "zone") {
          this.lastHurtT = t
          this.lastHurtBy = e.by ?? -1
          if (e.bearing != null) {
            this.alertYaw = e.bearing
            this.alertUntil = t + 2.5
            if (e.by != null && e.by >= 0 && e.by < 100) {
              this.lastMechHurtT = t
              this.hurtBy.set(e.by, t)
              const k = this.tracks.get(e.by)
              const D = k ? clamp(dist2(x, y, k.x, k.y), 30, 250) : e.weapon === "lightning" ? 35 : 110
              guess(e.by, x + Math.cos(e.bearing) * D, y + Math.sin(e.bearing) * D, e.weapon, 1)
            }
          }
        }
      } else if (e.type === "took") this.pickups.delete(e.id)
      else if (e.type === "take_failed") {
        if (e.reason === "gone") this.pickups.delete(e.id)
        else this.ignore.set(e.id, t + (e.reason === "range" ? 1 : 20))
      } else if (e.type === "drop") this.dropsTaken++
      else if (e.type === "kill") {
        this.tracks.delete(e.target)
        this.myKills++
      }
    }
    // Mechs dying that we did not kill means the others are fighting each other: a free-for-all.
    // (The practice squad never fights itself; the margin allows for a zone or drone accident.)
    if (!this.ffa && (this.cfg.match.mechs ?? 16) - obs.alive - this.myKills >= 3) this.ffa = true
    // someone close that we cannot see: look that way
    if (this.alertUntil < t) {
      for (const s of obs.sounds) {
        if (s.dist === "near" && (s.kind === "step" || s.kind === "boost" || s.kind === "reload" || s.kind === "jet" || s.kind === "laser_charge" || s.kind === "lightning" || s.kind === "gun")) {
          this.alertYaw = s.bearing
          this.alertUntil = t + 1.5
          break
        }
      }
    }
    // zone bookkeeping: does `zone.radius` move during a shrink, or stay at the phase's start value?
    const zn = obs.zone
    if (zn.shrink_end !== this.zKey) {
      this.zKey = zn.shrink_end
      this.zHoldR = zn.radius
      this.zSame = 0
      this.zStatic = false
    } else if (t > zn.shrink_start + 0.3 && t < zn.shrink_end - 0.3 && Math.abs(zn.radius - zn.next_radius) > 0.5) {
      if (Math.abs(zn.radius - this.zHoldR) < 1e-3) {
        if (++this.zSame >= 3) this.zStatic = true
      } else {
        this.zSame = 0
        this.zStatic = false
      }
    }
  }

  // ------------------------------------------------------------------------------------------ zone
  /** the zone circle at match time tt (>= now) */
  private zoneAt(obs: Observation, tt: number): Circle {
    const z = obs.zone
    let f = 0
    if (tt > z.shrink_start && z.shrink_end > z.shrink_start) {
      if (this.zStatic) f = clamp((tt - z.shrink_start) / (z.shrink_end - z.shrink_start), 0, 1)
      else {
        const t0 = Math.max(obs.t, z.shrink_start)
        f = z.shrink_end > t0 ? clamp((tt - t0) / (z.shrink_end - t0), 0, 1) : 1
      }
    }
    return {
      cx: z.center[0] + (z.next_center[0] - z.center[0]) * f,
      cy: z.center[1] + (z.next_center[1] - z.center[1]) * f,
      r: z.radius + (z.next_radius - z.radius) * f,
    }
  }

  /** match time at which the zone edge passes this point (Infinity if it stays inside) */
  private edgeTime(obs: Observation, px: number, py: number): number {
    const z = obs.zone
    const start = Math.max(obs.t, z.shrink_start)
    for (let tt = start; ; tt += 1) {
      const last = tt >= z.shrink_end
      const c = this.zoneAt(obs, last ? z.shrink_end : tt)
      if (dist2(px, py, c.cx, c.cy) > c.r - 4) return Math.min(tt, z.shrink_end)
      if (last) return Infinity
    }
  }

  /** a walkable spot inside the next circle to fall back to */
  private homePoint(obs: Observation, c: Ctx): [number, number] {
    const z = obs.zone
    const frac = z.next_radius >= 140 ? 0.5 : z.next_radius >= 60 ? 0.35 : 0
    const R = z.next_radius * frac
    const [ncx, ncy] = z.next_center
    const d = dist2(c.x, c.y, ncx, ncy)
    let hx = ncx
    let hy = ncy
    if (d <= R) {
      hx = c.x
      hy = c.y
    } else if (R > 0) {
      hx = ncx + ((c.x - ncx) / d) * R
      hy = ncy + ((c.y - ncy) / d) * R
    }
    const cell = this.nav.nearestWalkable(hx, hy, 16)
    if (cell >= 0) return [this.nav.cellX(cell), this.nav.cellY(cell)]
    const cell2 = this.nav.nearestWalkable(ncx, ncy, 30)
    return cell2 >= 0 ? [this.nav.cellX(cell2), this.nav.cellY(cell2)] : [ncx, ncy]
  }

  /** a spot next to a wall near (hx,hy): cover is one step away when someone shows up */
  private idleSpot(hx: number, hy: number, z: Observation["zone"]): [number, number] {
    // one resting spot per zone phase, kept for as long as it stays inside the next circle
    const key = String(z.shrink_end)
    if (key === this.idleKey && Number.isFinite(this.idleX) && dist2(this.idleX, this.idleY, z.next_center[0], z.next_center[1]) <= z.next_radius - 8) return [this.idleX, this.idleY]
    this.idleKey = key
    const w = this.world
    let best = Infinity
    let bx = hx
    let by = hy
    const need = this.nav.need
    for (let r = 0; r <= 48; r += 6) {
      const n = r === 0 ? 1 : NDIR
      for (let k = 0; k < n; k++) {
        const px = hx + DIRX[k] * r
        const py = hy + DIRY[k] * r
        if (dist2(px, py, z.next_center[0], z.next_center[1]) > z.next_radius - 8) continue
        const cl = this.nav.clearAt(px, py)
        if (cl < need + 0.3 || cl > need + 6 || !this.nav.free(px, py)) continue
        // how many directions can a mech 70 m away see this spot from? fewer is better
        const gz = w.groundAt(px, py) + 5
        let vis = 0
        for (let q = 0; q < NDIR; q++) {
          const ox = px + DIRX[q] * 70
          const oy = py + DIRY[q] * 70
          if (w.losClear(ox, oy, w.topAt(ox, oy) + 8, px, py, gz)) vis++
        }
        const s = vis + r / 12
        if (s < best) {
          best = s
          bx = px
          by = py
        }
      }
    }
    this.idleX = bx
    this.idleY = by
    return [bx, by]
  }

  // ---------------------------------------------------------------------------------------- assess
  private assess(obs: Observation, radius: number, height: number): Ctx {
    const me = obs.self
    const t = obs.t
    const [x, y, z] = me.pos
    const S = me.stats
    const mh = S.muzzle_height ?? 7
    const walk = S.walk_speed ?? 11.7
    const hpFrac = me.hp / Math.max(1, me.max_hp)
    const slot = (kind: string) => me.weapons.findIndex((w) => w?.kind === kind)
    const sGun = slot("gun")
    const sLaser = slot("laser")
    const sLight = slot("lightning")
    const sMortar = slot("mortar")
    const zn = obs.zone

    const zone = this.zoneAt(obs, t)
    const dNext = dist2(x, y, zn.next_center[0], zn.next_center[1]) - zn.next_radius
    const insideNext = dNext < -6
    let mustGo = !me.in_zone
    if (!insideNext) {
      const tEdge = this.edgeTime(obs, x, y)
      if (tEdge - t < 5 + (Math.max(0, dNext) / walk) * 0.35) mustGo = true
      if (zn.dps >= 60 && t > zn.shrink_start - (Math.max(0, dNext) / (walk * 0.6) + 4)) mustGo = true
    }
    const king = zn.next_radius <= 25.5

    // other mechs
    const threats: Track[] = []
    let exposed = false
    for (const k of this.tracks.values()) {
      if (k.kind !== "mech" || t - k.seen > 3) continue
      if (dist2(x, y, k.x, k.y) > 300) continue
      threats.push(k)
    }
    if (threats.length) this.lastThreatT = t

    const underFire = t - this.lastMechHurtT < 3
    let target: Aimable | null = null
    let targetLos = false
    let targetDist = Infinity
    let bestScore = -1e9
    let unseenShooter = false
    /** the chosen target has its back or side to us: it is fighting someone else */
    let targetAway = false
    for (const k of threats) {
      if (k.phantom) {
        if (t - k.seen < 1.5) {
          exposed = true
          unseenShooter = true
        }
        continue
      }
      if (k.seen < t - 0.01) continue
      const d = Math.hypot(k.x - x, k.y - y, k.z - z)
      // aim at the chest, or at the shoulders if only they show above cover
      let frac = 0.5
      let los = this.world.losClear(x, y, z + mh, k.x, k.y, k.z + k.h * 0.5)
      if (!los && this.world.losClear(x, y, z + mh, k.x, k.y, k.z + k.h * 0.86)) {
        los = true
        frac = 0.86
      }
      if (los) exposed = true
      let s = -d / tv("tsD", 60) - (k.hp / Math.max(1, k.maxHp)) * tv("tsHp", 1.5) + (los ? tv("tsLos", 2.5) : 0)
      if (k.id === this.targetId) s += tv("tsStick", 0.6)
      if (k.id === this.lastHurtBy && t - this.lastHurtT < 4) s += tv("tsAtk", 1)
      const away = Math.abs(wrap(Math.atan2(y - k.y, x - k.x) - k.yaw)) > 1.3
      if (away && tv("tp", 0) > 0) s += 1.2 + (1 - k.hp / Math.max(1, k.maxHp))
      if (s > bestScore) {
        bestScore = s
        target = { ...k, aimFrac: frac }
        targetLos = los
        targetDist = d
        targetAway = away
      }
    }
    const tk0 = target as Aimable | null
    if (tk0) {
      this.targetId = tk0.id
      if (targetLos) this.lastSawTargetT = t
    } else this.targetId = -1

    // ---- posture. Any mech in contact puts us behind cover. From there we either trade shots
    // (lean out when a weapon is ready) or stay hidden and repair. Hysteresis stops flapping.
    const nT = threats.length
    // once a mech is clearly losing, keep hitting it: if we let go it repairs and the damage is wasted
    const finishable = !!tk0 && (tk0.hp < me.hp * 0.5 || (tk0.hp < 0.55 * tk0.maxHp && me.hp > tk0.hp * 1.3))
    // a laser out-trades a gun at range: do not trade with one unless we hold one too
    const outgunned = !!tk0 && tk0.weapons.includes("laser") && sLaser < 0 && targetDist > 55
    const ambush = this.ambushStance()
    const camp = !ambush && CAMP && sLaser >= 0
    const hurtByMech = this.lastHurtBy >= 0 && this.lastHurtBy < 100 && t - this.lastHurtT < 4
    let hide = this.evading
    let fight: boolean
    if (ambush) {
      // free-for-all: placement is what counts and the others thin each other out. Stay out of
      // sight, and only trade shots with a mech that is close, is shooting us, or is nearly dead.
      if (!hide) {
        if (nT > 0 && hpFrac < (nT >= 2 ? 0.7 : 0.45) && !finishable) hide = true
        if (nT >= 3) hide = true
        if (nT >= 2 && tv("ffa2hide", 0) > 0 && !finishable && !(targetAway && !underFire)) hide = true
        if (unseenShooter && underFire && !tk0) hide = true
        if (outgunned && hpFrac < 0.85 && !finishable) hide = true
      } else if (nT === 0) {
        if (hpFrac > 0.85 || t - this.lastThreatT > 10) hide = false
      } else if (nT === 1 && ((hpFrac > 0.85 && !unseenShooter && !outgunned) || finishable)) hide = false
      const freeShot = tv("tp", 0) > 0 && targetAway && targetLos && !underFire && targetDist <= (sLaser >= 0 ? 250 : 150)
      fight = nT > 0 && !hide && !!tk0 && (targetDist <= 120 || hurtByMech || finishable || freeShot)
    } else {
      // practice squad: they never fight each other, so a house bot only dies if we kill it.
      // Take the fight while healthy, break off when losing.
      if (!hide) {
        if (nT > 0 && hpFrac < tv("evHp", 0.4) && !finishable) hide = true
        if (nT >= tv("evN", 6)) hide = true
        if (nT === 2 && hpFrac < tv("evHp2", 0.5)) hide = true
        if (unseenShooter && underFire && !tk0) hide = true
        if (outgunned && hpFrac < tv("ogHp", 0.8) && !finishable) hide = true
      } else if ((nT === 0 && (hpFrac > tv("exHp0", 0.75) || t - this.lastThreatT > 8)) || (nT === 1 && hpFrac > tv("exHp1", 0.85) && !unseenShooter && !outgunned) || (finishable && nT === 1)) hide = false
      fight = !hide && !!tk0 && (camp || (targetLos && targetDist <= tv("engR", 175)) || targetDist <= tv("engNear", 60) || hurtByMech)
    }
    if (king) hide = false
    const power = this.updatePower(obs, hpFrac, ambush, king)
    if (power) {
      // not yet: leave the hunter alone, farm, and let the escalation wait
      hide = false
      fight = false
      // still shoot back at a mech that is on top of us or already hitting us
      const defend = !!tk0 && (hurtByMech || targetDist < tv("pDefend", 60))
      if (!defend) {
        target = null
        targetLos = false
        targetDist = Infinity
      }
    }
    this.evading = hide
    if (hide) fight = false
    // in a free-for-all, a mech we are not fighting is one to stay hidden from
    const evade = hide || (ambush && nT > 0 && !fight && !king)

    // nothing hostile in the way: shoot drones that shoot us, then crates, then drones for cubes
    if (!target) {
      let bd = Infinity
      const consider = (a: Aimable, maxD: number, bonus: number) => {
        const d = Math.hypot(a.x - x, a.y - y, a.z - z)
        if (d > maxD) return
        if (!this.world.losClear(x, y, z + mh, a.x, a.y, a.z + a.h * 0.5)) return
        if (d - bonus < bd) {
          bd = d - bonus
          target = a
          targetLos = true
          targetDist = d
        }
      }
      // shooting anything restarts the repair timer, so only farm at full health and with nobody about
      let nearMech = Infinity
      for (const k of threats) nearMech = Math.min(nearMech, dist2(x, y, k.x, k.y))
      const calm = power
        ? me.in_zone && nearMech > 60
        : nT === 0 && !underFire && !hide && hpFrac >= tv("farmHp", 0.9) && t - this.lastThreatT > tv("farmWait", 6) && me.cubes < 14 && me.in_zone
      for (const k of this.tracks.values()) {
        if (k.kind !== "npc" || k.seen < t - 0.01) continue
        const angry = k.id === this.lastHurtBy && t - this.lastHurtT < 6
        if (angry) consider({ ...k, kind: "npc" }, 200, 1000)
        else if (calm && hpFrac > 0.6) consider({ ...k, kind: "npc" }, tv("npcR", 110), 0)
      }
      if (calm) {
        const cr = this.cfg.crates
        for (const it of obs.crates) consider({ id: it.id, kind: "crate", x: it.pos[0], y: it.pos[1], z: it.pos[2], vx: 0, vy: 0, vz: 0, r: cr.radius ?? 1.4, h: cr.height ?? 2.4, hp: it.hp ?? 300, maxHp: cr.hp ?? 300, weapons: [] }, tv("crateR", 150), 60)
      }
    }

    // peek logic: want a line of fire when a weapon is ready, want to be hidden when none is
    let wantLos = 0
    if (!power && nT > 0 && tk0 && (ambush || camp || fight)) {
      if (!fight) wantLos = -1.6
      else {
        let ready = false
        const gun = sGun >= 0 ? me.weapons[sGun] : null
        const laser = sLaser >= 0 ? me.weapons[sLaser] : null
        const light = sLight >= 0 ? me.weapons[sLight] : null
        // how far each ready weapon can usefully reach: beyond that, wait behind cover
        if (gun && (gun.ammo ?? 1) > 0 && (gun.reload ?? 0) <= 0.25 && (!(ambush || camp) || targetDist <= Math.min((gun.stats.range ?? 230) - 10, (gun.stats.speed ?? 252) * 0.8))) ready = true
        let laserHot = false
        if (laser && (laser.charge ?? 0) >= (laser.stats.charge ?? 0.75) - 0.3 && targetDist <= (laser.stats.range ?? 280) - 5) {
          ready = true
          laserHot = true
        }
        if (light && targetDist <= (light.stats.range ?? 45) + 6 && me.energy >= (light.stats.energy ?? 3) * 3) ready = true
        let duck = false
        for (const k of threats) {
          if (!k.charging || k.seen < t - 0.3) continue
          const toMe = Math.atan2(y - k.y, x - k.x)
          if (Math.abs(wrap(toMe - k.yaw)) < 0.3 && dist2(x, y, k.x, k.y) < 300) duck = true
        }
        wantLos = duck && !laserHot ? -1.6 : ready ? 1 : -tv("hideW", 1)
      }
    }

    return { x, y, z, t, radius, height, mh, hpFrac, walk, zone, insideNext, mustGo, king, evade, fight, ambush, power, camp, cover: ambush || camp, threats, target, targetLos, targetDist, wantLos, sGun, sLaser, sLight, sMortar, exposed }
  }

  /**
   * Practice only. Every house bot we kill brings another hunter, so the escalation should start
   * when we are strong, not at spawn. Until then: do not shoot mechs, keep away from them, and
   * collect cubes, a laser and supply drops. Ends for good on the first of: strong enough, out of
   * time, hurt, shot at by several mechs, a kill, or the match turning out to be a free-for-all.
   */
  private updatePower(obs: Observation, hpFrac: number, ambush: boolean, king: boolean): boolean {
    // the rat posture can begin late (once the match is recognised as a free-for-all)
    if (!this.power && !king && tv("rat", 0) > 0 && (ambush || this.ffa)) this.power = true
    if (!this.power) return false
    const t = obs.t
    let attackers = 0
    for (const at of this.hurtBy.values()) if (t - at < 8) attackers++
    // free-for-all "rat" play: keep out of everyone's way until the last circles (tune rat=1)
    const rat = tv("rat", 0) > 0 && (ambush || this.ffa)
    let end = ""
    if (tv("power", POWER ? 1 : 0) <= 0 && !rat) end = "off"
    else if (king) end = "endgame"
    else if (rat) end = ""
    else if (ambush || this.ffa) end = "free-for-all"
    else if (this.myKills > 0) end = "kill"
    else if (obs.self.cubes >= tv("pCubes", 10)) end = "strong"
    else if (t >= tv("pT", 120)) end = "time"
    else if (hpFrac < tv("pHp", 0.5)) end = "hurt"
    else if (attackers >= 2) end = "several attackers"
    if (end) {
      this.power = false
      this.powerEnded = `${end}@${t.toFixed(0)}s`
    }
    return this.power
  }

  /** a spot a good walk away that puts distance (and ideally a building) between us and the mechs */
  private fleePoint(obs: Observation, c: Ctx, foes: Track[]): [number, number] | null {
    const { x, y, t } = c
    const w = this.world
    const soon = this.zoneAt(obs, t + 8)
    const zn = obs.zone
    const late = t > zn.shrink_start - 30
    let cur = Infinity
    for (const k of foes) cur = Math.min(cur, dist2(x, y, k.x, k.y))
    let best = -1e9
    let out: [number, number] | null = null
    for (const r of [45, 80, 120]) {
      for (let i = 0; i < NDIR; i++) {
        const px = x + DIRX[i] * r
        const py = y + DIRY[i] * r
        if (!this.nav.free(px, py)) continue
        if (dist2(px, py, soon.cx, soon.cy) > soon.r - 10) continue
        if (late && dist2(px, py, zn.next_center[0], zn.next_center[1]) > zn.next_radius - 8) continue
        let near = Infinity
        let seen = 0
        const gz = w.groundAt(px, py) + c.height * 0.5
        for (const k of foes) {
          near = Math.min(near, dist2(px, py, k.x, k.y))
          if (w.losClear(k.x, k.y, k.z + k.h * 0.89, px, py, gz)) seen++
        }
        const sc = near - cur - seen * 35 - (this.nav.segOk(x, y, px, py) ? 0 : 15)
        if (sc > best) {
          best = sc
          out = [px, py]
        }
      }
    }
    return out
  }

  /** somewhere inside the zone we have not looked at lately, away from the mechs */
  private explorePoint(obs: Observation, c: Ctx, foes: Track[], ok: (px: number, py: number) => boolean): [number, number] {
    const { x, y, t } = c
    const w = this.world
    const G = 40
    const gw = Math.ceil((w.bMaxX - w.bMinX) / G)
    const gh = Math.ceil((w.bMaxY - w.bMinY) / G)
    if (this.visitT.length !== gw * gh) this.visitT = new Float32Array(gw * gh).fill(-1e9)
    // what we can see from here counts as looked at
    const ci = Math.floor((x - w.bMinX) / G)
    const cj = Math.floor((y - w.bMinY) / G)
    for (let dj = -2; dj <= 2; dj++) {
      for (let di = -2; di <= 2; di++) {
        const i = ci + di
        const j = cj + dj
        if (i >= 0 && j >= 0 && i < gw && j < gh) this.visitT[j * gw + i] = t
      }
    }
    if (Number.isFinite(this.exploreX) && t - this.exploreT < 10 && dist2(x, y, this.exploreX, this.exploreY) > 25 && ok(this.exploreX, this.exploreY)) return [this.exploreX, this.exploreY]
    let best = -1e9
    let bx = obs.zone.next_center[0]
    let by = obs.zone.next_center[1]
    for (let j = 0; j < gh; j++) {
      for (let i = 0; i < gw; i++) {
        const px = w.bMinX + (i + 0.5) * G
        const py = w.bMinY + (j + 0.5) * G
        if (!ok(px, py)) continue
        const d = dist2(x, y, px, py)
        if (d < 50 || d > 320) continue
        if (this.nav.clearAt(px, py) < this.nav.need + 1) continue
        let sc = Math.min(3, (t - this.visitT[j * gw + i]) / 60) - d / 160
        for (const k of foes) {
          const kd = dist2(px, py, k.x, k.y)
          if (kd < 160) sc -= 3
          const fd = dist2(x, y, k.x, k.y)
          if (fd < 320 && ((px - x) * (k.x - x) + (py - y) * (k.y - y)) / (Math.max(1, d) * Math.max(1, fd)) > 0.4) sc -= 1.5
        }
        if (sc > best) {
          best = sc
          bx = px
          by = py
        }
      }
    }
    this.exploreX = bx
    this.exploreY = by
    this.exploreT = t
    return [bx, by]
  }

  /** hide-first posture? Forced by MECH_STANCE=ambush|brawl, otherwise on once the match looks like a free-for-all. */
  private ambushStance(): boolean {
    const st = this.stance ?? STANCE
    return st === "ambush" ? true : st === "brawl" ? false : this.ffa
  }

  // ------------------------------------------------------------------------------------------ goal
  private chooseGoal(obs: Observation, c: Ctx): Goal {
    const me = obs.self
    const zn = obs.zone
    const { x, y, t } = c
    const stay: Goal = { x, y, arrive: 1e9, why: "hold", boost: false, travel: false }
    const live = c.zone
    const late = t > zn.shrink_start - 20
    // loot in a walkable area other than the one we stand in cannot be walked to: leave it
    const myArea = tv("hop", 1) > 0 && me.grounded ? this.nav.areaAt(x, y, 4) : -1
    const zoneOk = (px: number, py: number) => (late ? dist2(px, py, zn.next_center[0], zn.next_center[1]) < zn.next_radius - 4 : dist2(px, py, live.cx, live.cy) < live.r - 6)
    const okSpot = (px: number, py: number) => zoneOk(px, py) && (myArea < 0 || this.nav.areaAt(px, py, 5) === myArea)
    const skip = (id: number) => (this.ignore.get(id) ?? -1) > t
    // give up on something we are not getting closer to (on a ledge, behind a fence, in a pit)
    const chase = (id: number, d: number, limit: number): boolean => {
      if (id !== this.chaseId || d < this.chaseBest - 2.5) {
        this.chaseId = id
        this.chaseBest = d
        this.chaseT = t
      }
      if (t - this.chaseT <= limit) return true
      this.ignore.set(id, t + 30)
      this.chaseId = -1
      return false
    }

    // a laser lying on a roof, when we hold none: worth jetting up for (see hop)
    const roofLaser = (reach: number): Goal | null => {
      // switched off: in the simulator it rescued only a few matches and cost about as much elsewhere
      if (tv("roof", 0) <= 0 || c.sLaser >= 0) return null
      let bi: Item | null = null
      let bd = reach
      for (const it of this.pickups.values()) {
        if (!it.roof || it.kind !== "laser" || skip(it.id) || !zoneOk(it.x, it.y)) continue
        const d = dist2(x, y, it.x, it.y)
        if (d < bd) {
          bd = d
          bi = it
        }
      }
      if (!bi || !chase(bi.id, bd, 18)) return null
      this.roofItem = bi
      return { x: bi.x, y: bi.y, arrive: 1.2, why: "roof-laser", boost: false, travel: true }
    }

    if (c.mustGo) {
      const [hx, hy] = this.homePoint(obs, c)
      return { x: hx, y: hy, arrive: 3, why: "zone", boost: !me.in_zone, travel: true }
    }
    if (c.king) {
      const cell = this.nav.nearestWalkable(zn.next_center[0], zn.next_center[1], 12)
      const kx = cell >= 0 && !this.nav.free(zn.next_center[0], zn.next_center[1], 0.5) ? this.nav.cellX(cell) : zn.next_center[0]
      const ky = cell >= 0 && !this.nav.free(zn.next_center[0], zn.next_center[1], 0.5) ? this.nav.cellY(cell) : zn.next_center[1]
      return { x: kx, y: ky, arrive: 0.5, why: "king", boost: dist2(x, y, kx, ky) > zn.next_radius + 10, travel: true }
    }
    if (c.power) {
      const foes: Track[] = []
      // which mechs to keep away from: all of them (pAvoid 2), only ones that have shot us (1), none (0)
      const avoid = tv("pAvoid", 1)
      for (const k of this.tracks.values()) {
        if (k.kind !== "mech" || t - k.seen >= 8 || avoid === 0) continue
        if (avoid >= 2 || t - (this.hurtBy.get(k.id) ?? -1e9) < 20) foes.push(k)
      }
      let fd = Infinity
      for (const k of foes) fd = Math.min(fd, dist2(x, y, k.x, k.y))
      // a mech is getting close: open the distance before anything else (gun fire can be dodged
      // beyond about 110 m and does not reach past 230 m)
      const keep = tv("pKeep", 150)
      if (fd < keep) {
        const away = this.fleePoint(obs, c, foes)
        if (away) return { x: away[0], y: away[1], arrive: 3, why: "power-away", boost: fd < keep - 40 && me.energy > 25, travel: true }
      }
      type Pick = { id: number; x: number; y: number; arrive: number; why: string; lim: number }
      const picks: Pick[] = []
      let ps = 0
      const consider = (id: number, px: number, py: number, value: number, arrive: number, why: string, lim: number) => {
        if (skip(id) || !okSpot(px, py)) return
        const d = dist2(x, y, px, py)
        if (d > 360) return
        let sc = value / (d + 40)
        for (const k of foes) {
          const kd = dist2(px, py, k.x, k.y)
          if (kd < 110) return
          if (kd < d + 30) sc *= 0.3
          const f = dist2(x, y, k.x, k.y)
          if (f < 300 && ((px - x) * (k.x - x) + (py - y) * (k.y - y)) / (Math.max(1, d) * Math.max(1, f)) > 0.5) sc *= 0.4
        }
        if (sc > ps) {
          ps = sc
          picks[0] = { id, x: px, y: py, arrive, why, lim }
        }
      }
      for (const d of obs.drops) {
        const dd = dist2(x, y, d.pos[0], d.pos[1])
        if (d.lands_in <= dd / (c.walk * 0.75) + 12) consider(d.id, d.pos[0], d.pos[1], 600, 1.5, "power-drop", 18)
      }
      for (const it of this.cubes.values()) if (!it.roof) consider(it.id, it.x, it.y, 160, 0.8, "power-cube", 7)
      for (const it of this.pickups.values()) {
        if (it.roof) continue
        const gain = this.pickupGain(me.weapons, it.kind, live.r)
        if (gain > 0) consider(it.id, it.x, it.y, it.kind === "laser" ? 420 : 50 * gain, 1.2, `power-${it.kind}`, 9)
      }
      for (const it of this.crates.values()) if (!it.roof) consider(it.id, it.x, it.y, 110, 5, "power-crate", 12)
      if (c.hpFrac > 0.6) for (const k of this.tracks.values()) if (k.kind === "npc" && t - k.seen < 10) consider(k.id, k.x, k.y, 190, 70, "power-drone", 16)
      const pk = picks[0]
      if (pk && chase(pk.id, dist2(x, y, pk.x, pk.y), pk.lim)) return { x: pk.x, y: pk.y, arrive: pk.arrive, why: pk.why, boost: false, travel: true }
      const [ex, ey] = this.explorePoint(obs, c, foes, okSpot)
      return { x: ex, y: ey, arrive: 8, why: "power-explore", boost: false, travel: true }
    }

    // ---- a supply drop is a second life (full repair, full energy, 3 cubes): go for it under fire.
    // It only needs us alone in its ring for a few seconds; being shot at does not stop the capture.
    if ((DROP_UNDER_FIRE || tv("drop", 0) > 0) && c.threats.length > 0) {
      for (const d of obs.drops) {
        if (skip(d.id)) continue
        const dd = dist2(x, y, d.pos[0], d.pos[1])
        if (dd > tv("dropReach", 240)) continue
        // set off in time to be standing there when it lands, not earlier
        if (d.lands_in > dd / (c.walk * 0.75) + tv("dropLead", 6)) continue
        if (d.holder != null && d.holder !== me.id && d.progress > 1.5) continue
        if (dd > 8 && !chase(d.id, dd, 16)) continue
        return { x: d.pos[0], y: d.pos[1], arrive: 2.5, why: "grab-drop", boost: dd > 30 && me.energy > 35, travel: true }
      }
    }

    // ---- power beats position. A laser is worth about a kill a match and every cube is 4% more HP
    // and damage, so with a mech about we still go for one that is close (the torso keeps shooting).
    if (c.threats.length > 0 && c.hpFrac > tv("grabHp", 0.35)) {
      // not something an enemy is standing nearer to than we are
      const contested = (px: number, py: number, d: number) => c.threats.some((k) => dist2(px, py, k.x, k.y) + 10 < d)
      let gc: Item | null = null
      let gd = c.ambush ? 25 : tv("grabR", 45)
      for (const it of this.cubes.values()) {
        if (it.roof || skip(it.id) || !okSpot(it.x, it.y)) continue
        const d = dist2(x, y, it.x, it.y)
        if (d < gd && !contested(it.x, it.y, d)) {
          gd = d
          gc = it
        }
      }
      if (gc && chase(gc.id, gd, 6)) return { x: gc.x, y: gc.y, arrive: 0.8, why: "grab-cube", boost: false, travel: true }
      let gw: Item | null = null
      let gs = 0
      for (const it of this.pickups.values()) {
        if (it.roof || skip(it.id) || !okSpot(it.x, it.y)) continue
        const gain = this.pickupGain(me.weapons, it.kind, live.r)
        const d = dist2(x, y, it.x, it.y)
        // a laser is worth a long run; anything else only if we hold nothing and it is on the way
        const reach = (it.kind === "laser" ? tv("grabLaser", 120) : gain >= 2 ? 60 : 0) * (c.ambush ? 0.6 : 1)
        if (gain <= 0 || d > reach || contested(it.x, it.y, d)) continue
        const s = gain / (d + 25)
        if (s > gs) {
          gs = s
          gw = it
        }
      }
      if (gw && chase(gw.id, dist2(x, y, gw.x, gw.y), 9)) return { x: gw.x, y: gw.y, arrive: 1.2, why: `grab-${gw.kind}`, boost: c.exposed && me.energy > 50, travel: true }
      if (c.threats.length <= tv("roofN", 2) && c.hpFrac > tv("roofHp", 0.5)) {
        const g = roofLaser(tv("roofR", 90))
        if (g) return g
      }
    }

    if (!c.cover) {
      // ---- practice posture: break off when losing, otherwise hold a range that suits our weapons
      if (c.evade) {
        const [cx, cy] = this.coverSpot(obs, c, false)
        return { x: cx, y: cy, arrive: 1.5, why: "evade", boost: c.exposed && me.energy > 30, travel: true }
      }
      if (c.fight && c.target) {
        const k = c.target
        const d = dist2(x, y, k.x, k.y)
        let rmin = tv("gmin", 45)
        let rmax = tv("gmax", 85)
        if (c.sLaser >= 0) {
          rmin = tv("lmin", 125)
          rmax = tv("lmax", 185)
        } else if (c.sLight >= 0 && me.energy > 25) {
          rmin = 12
          rmax = 36
        } else if (k.weapons.includes("laser")) {
          rmin = 30
          rmax = 60
        }
        if (k.weapons.includes("lightning") && c.sLight < 0) rmin = Math.max(rmin, 60)
        if (k.hp < 0.3 * k.maxHp && c.hpFrac > 0.5) rmax = Math.min(rmax, 45)
        rmax = Math.max(rmax, rmin + 15)
        const lost = t - this.lastSawTargetT > 4
        const inZone = dist2(k.x, k.y, zn.next_center[0], zn.next_center[1]) < zn.next_radius + 20 || t < zn.shrink_start - 25
        if (d > rmax + 4 && !lost && inZone) return { x: k.x, y: k.y, arrive: rmax, why: "fight-close", boost: false, travel: true }
        if (d < rmin - 4 && d > 0.5) {
          const back = this.retreatPoint(obs, c, k)
          if (back) return { x: back[0], y: back[1], arrive: 2, why: "fight-back", boost: d < rmin - tv("kboost", 15) && me.energy > 45, travel: false }
        }
        return { ...stay, why: "fight" }
      }
    }
    // ---- free-for-all posture: get behind cover and deal with a mech from there. We never walk
    // at an enemy across open ground; it has to come to us, and shows itself when it does.
    if (c.cover && (c.evade || c.fight) && c.threats.length > 0) {
      const k = c.fight ? c.target : null
      if (k && k.kind === "mech" && k.hp < 0.25 * k.maxHp && c.hpFrac > 0.5 && c.threats.length === 1 && c.targetDist < 130) {
        // one shot from dead and alone: go and finish it
        return { x: k.x, y: k.y, arrive: 25, why: "finish", boost: me.energy > 40, travel: true }
      }
      const [cx, cy] = this.coverSpot(obs, c, c.fight)
      const far = dist2(x, y, cx, cy) > 10
      return { x: cx, y: cy, arrive: 1.5, why: c.fight ? (c.camp ? "camp" : "ambush") : "hide", boost: c.exposed && far && me.energy > (c.fight ? 45 : 25), travel: true }
    }
    // just lost contact: stay put behind the same cover for a few seconds before wandering off
    if (t - this.lastThreatT < 6 && Number.isFinite(this.coverX) && t - this.coverT < 12 && dist2(x, y, this.coverX, this.coverY) < 40) {
      return { x: this.coverX, y: this.coverY, arrive: 2, why: "wary", boost: false, travel: false }
    }

    // ---- nobody to deal with: improve
    const quiet = c.threats.length === 0

    // supply drop: 3 cubes and a full repair
    for (const d of obs.drops) {
      if (skip(d.id)) continue
      const dd = dist2(x, y, d.pos[0], d.pos[1])
      if (dd > 280 || d.lands_in > dd / c.walk + 16) continue
      if (!okSpot(d.pos[0], d.pos[1]) && d.lands_in > 0) continue
      if (!quiet && c.hpFrac < 0.5) continue
      if (dd > 6 && !chase(d.id, dd, 14)) continue
      return { x: d.pos[0], y: d.pos[1], arrive: 1.5, why: "drop", boost: false, travel: true }
    }
    // cubes lying around
    let best: Item | null = null
    let bd = tv("cubeR", 90)
    for (const it of this.cubes.values()) {
      if (it.roof || !okSpot(it.x, it.y) || skip(it.id)) continue
      const d = dist2(x, y, it.x, it.y)
      if (d < bd) {
        bd = d
        best = it
      }
    }
    if (best && chase(best.id, bd, 7)) return { x: best.x, y: best.y, arrive: 0.8, why: "cube", boost: false, travel: true }

    // a better weapon
    let bw: Item | null = null
    let bs = 0
    for (const it of this.pickups.values()) {
      if (it.roof || !okSpot(it.x, it.y) || (this.ignore.get(it.id) ?? -1) > t) continue
      const gain = this.pickupGain(me.weapons, it.kind, live.r)
      if (gain <= 0) continue
      const d = dist2(x, y, it.x, it.y)
      if (d > (it.kind === "laser" || !(me.weapons[1] && me.weapons[0]) ? 320 : 150)) continue
      const s = gain / (d + 25)
      if (s > bs) {
        bs = s
        bw = it
      }
    }
    if (!bw || bw.kind !== "laser") {
      const g = roofLaser(tv("roofRq", 160))
      if (g) return g
    }
    if (bw && chase(bw.id, dist2(x, y, bw.x, bw.y), 9)) return { x: bw.x, y: bw.y, arrive: 1.2, why: `get-${bw.kind}`, boost: false, travel: true }

    // crates: a cube each and maybe a weapon (only at full health: shooting stops our repair)
    if (quiet && me.cubes < 14 && c.hpFrac >= 0.9 && !c.evade) {
      let bc: Item | null = null
      let bcd = 200
      for (const it of this.crates.values()) {
        if (it.roof || !okSpot(it.x, it.y) || skip(it.id)) continue
        const d = dist2(x, y, it.x, it.y)
        if (d < bcd) {
          bcd = d
          bc = it
        }
      }
      if (bc) {
        if (c.target && c.target.id === bc.id) return { ...stay, why: "crate" }
        // walk up until we have a clear shot at it
        if (chase(bc.id, bcd, 10)) return { x: bc.x, y: bc.y, arrive: 6, why: "to-crate", boost: false, travel: true }
      }
    }
    if (c.target) return { ...stay, why: "farm" }

    const [hx, hy] = this.homePoint(obs, c)
    const [ix, iy] = this.idleSpot(hx, hy, zn)
    return { x: ix, y: iy, arrive: 2, why: "home", boost: false, travel: true }
  }

  /** a reachable spot a short walk away that opens the distance to the target (and the other mechs) */
  private retreatPoint(obs: Observation, c: Ctx, k: Aimable): [number, number] | null {
    const { x, y, t } = c
    const soon = this.zoneAt(obs, t + 5)
    const zn = obs.zone
    const late = t > zn.shrink_start - 25
    let best = -1e9
    let out: [number, number] | null = null
    for (const r of [22, 38]) {
      for (let i = 0; i < NDIR; i++) {
        const px = x + DIRX[i] * r
        const py = y + DIRY[i] * r
        if (!this.nav.free(px, py) || !this.nav.segOk(x, y, px, py)) continue
        if (dist2(px, py, soon.cx, soon.cy) > soon.r - 8) continue
        if (late && dist2(px, py, zn.next_center[0], zn.next_center[1]) > zn.next_radius - 6) continue
        let s = dist2(px, py, k.x, k.y)
        let near = Infinity
        for (const th of c.threats) if (th.id !== k.id) near = Math.min(near, dist2(px, py, th.x, th.y))
        if (near < 90) s -= (90 - near) * 0.8
        if (s > best) {
          best = s
          out = [px, py]
        }
      }
    }
    return out && best > dist2(x, y, k.x, k.y) + 6 ? out : null
  }

  /** value of swapping/adding this pickup kind; <= 0 means leave it */
  private pickupGain(weapons: [WeaponObs | null, WeaponObs | null], kind: string, zoneR: number): number {
    const val = (k: string) => (k === "lightning" && zoneR <= 150 ? 3.3 : (WEAPON_VALUE[k] ?? 0))
    if (weapons.some((w) => w?.kind === kind)) return 0
    if (kind === "gun") return 0
    if (weapons.some((w) => !w)) return val(kind)
    let worst = Infinity
    for (const w of weapons) if (w && w.kind !== "gun") worst = Math.min(worst, val(w.kind))
    return worst === Infinity ? 0 : val(kind) - worst - 0.5
  }

  /**
   * Best nearby spot hidden from everyone hunting us. With `peek`, prefer a corner: hidden where we
   * stand, with a line of fire to the target a few steps to the side.
   */
  private coverSpot(obs: Observation, c: Ctx, peek: boolean): [number, number] {
    const { x, y, t } = c
    const w = this.world
    const seenBy = (px: number, py: number): number => {
      const gz = w.groundAt(px, py)
      let seen = 0
      for (const k of c.threats) {
        const ez = k.z + k.h * 0.89
        if (w.losClear(k.x, k.y, ez, px, py, gz + c.height * 0.5) || w.losClear(k.x, k.y, ez, px, py, gz + c.height - 0.5)) seen++
      }
      return seen
    }
    const have = Number.isFinite(this.coverX)
    // keep the spot we chose while it still hides us; look again at most a few times a second
    if (have && t - this.coverT < 0.3) return [this.coverX, this.coverY]
    if (have && t - this.coverT < 1.5 && this.coverPeek === peek && seenBy(this.coverX, this.coverY) === 0) return [this.coverX, this.coverY]
    const zn = obs.zone
    const soon = this.zoneAt(obs, t + 6)
    const tk = peek && c.target && c.target.kind === "mech" ? c.target : null
    // as the shrink nears, cover outside the next circle is a trap: the zone will flush us out
    const lead = zn.shrink_start - t
    const outPen = lead > 40 ? 0.5 : lead > 0 ? 0.5 + ((40 - lead) / 40) * 3.5 : 6
    let best = -1e9
    let bx = x
    let by = y
    let bestSeen = 99
    const consider = (px: number, py: number, r: number, bonus: number) => {
      if (!this.nav.free(px, py)) return
      if (dist2(px, py, soon.cx, soon.cy) > soon.r - 5) return
      const seen = seenBy(px, py)
      let s = -3 * seen - r / 35 + bonus
      let near = Infinity
      for (const th of c.threats) near = Math.min(near, dist2(px, py, th.x, th.y))
      if (near < Infinity) s += Math.min(near, 150) / 75
      if (dist2(px, py, zn.next_center[0], zn.next_center[1]) > zn.next_radius - 5) s -= outPen
      if (tk && seen === 0) {
        const tz = tk.z + tk.h * 0.5
        for (let k = 0; k < NDIR; k += 2) {
          const qx = px + DIRX[k] * 7
          const qy = py + DIRY[k] * 7
          if (this.nav.free(qx, qy) && w.losClear(qx, qy, w.groundAt(qx, qy) + c.mh, tk.x, tk.y, tz)) {
            s += 1.2
            break
          }
        }
      }
      if (s > best) {
        best = s
        bx = px
        by = py
        bestSeen = seen
      }
    }
    if (have) consider(this.coverX, this.coverY, dist2(x, y, this.coverX, this.coverY), 0.8)
    const ring = (r: number) => {
      const n = r === 0 ? 1 : NDIR
      for (let k = 0; k < n; k++) consider(x + DIRX[k] * r, y + DIRY[k] * r, r, 0)
    }
    for (const r of [0, 6, 12, 20, 30, 42, 56, 72]) ring(r)
    // nothing close hides us: look further before settling for the least bad spot
    if (bestSeen > 0) for (const r of [95, 125]) ring(r)
    this.coverX = bx
    this.coverY = by
    this.coverT = t
    this.coverPeek = peek
    return [bx, by]
  }

  // ---------------------------------------------------------------------------------------- moving
  /** unit direction to walk along the planned path toward (gx,gy) */
  private follow(x: number, y: number, gx: number, gy: number, t: number, onGround: boolean): [number, number] {
    const direct = (): [number, number] => {
      const d = dist2(x, y, gx, gy)
      return d > 1e-6 ? [(gx - x) / d, (gy - y) / d] : [0, 0]
    }
    const dGoal = dist2(x, y, gx, gy)
    if (dGoal < 5 || this.nav.segOk(x, y, gx, gy)) {
      this.path.length = 0
      return direct()
    }
    const moved = dist2(gx, gy, this.pathGX, this.pathGY)
    let need = this.replan || this.path.length === 0 || moved > Math.max(4, 0.12 * dGoal) || t - this.pathT > (this.pathPartial ? 1.2 : 4)
    if (!need && this.path.length) {
      const px = this.path[2 * this.pathI]
      const py = this.path[2 * this.pathI + 1]
      if (dist2(x, y, px, py) > 14) need = true
    }
    if (need && t - this.pathT > 0.25) {
      this.path = this.nav.findPath(x, y, gx, gy)
      this.pathPartial = this.nav.lastPartial
      this.pathI = 0
      this.pathGX = gx
      this.pathGY = gy
      this.pathT = t
      this.replan = false
    }
    const n = this.path.length >> 1
    if (n === 0) return direct()
    // advance past points we have reached, and snap forward to a nearer later point
    while (this.pathI < n - 1 && dist2(x, y, this.path[2 * this.pathI], this.path[2 * this.pathI + 1]) < 3) this.pathI++
    let bestI = this.pathI
    let bestD = dist2(x, y, this.path[2 * bestI], this.path[2 * bestI + 1])
    for (let k = this.pathI + 1; k < Math.min(n, this.pathI + 8); k++) {
      const d = dist2(x, y, this.path[2 * k], this.path[2 * k + 1])
      if (d < bestD) {
        bestD = d
        bestI = k
      }
    }
    this.pathI = bestI
    let aim = this.pathI
    if (onGround) {
      for (const look of [18, 12, 8, 5, 3, 1]) {
        const j = Math.min(n - 1, this.pathI + look)
        if (j > aim && this.nav.segOk(x, y, this.path[2 * j], this.path[2 * j + 1])) {
          aim = j
          break
        }
      }
    } else aim = Math.min(n - 1, this.pathI + 1)
    const ax = this.path[2 * aim] - x
    const ay = this.path[2 * aim + 1] - y
    const d = Math.hypot(ax, ay)
    return d > 1e-6 ? [ax / d, ay / d] : direct()
  }

  /**
   * Some parts of the map (the stadium bowl, a few sunken courts and terraces) have no walking way
   * out: the planner sees a wall all round. When where we need to be is in a different walkable
   * area, walk to the cheapest crossing, wait for the energy, then jump and jet over.
   * Returns true when it has set the movement for this decision.
   */
  private hop(obs: Observation, c: Ctx, goal: Goal, out: BotAction): boolean {
    if (tv("hop", 1) <= 0) return false
    const me = obs.self
    const { x, y, z, t } = c
    const nav = this.nav
    const w = this.world
    if (this.hopFly) {
      const p = this.hopPlan
      if (!p || t - this.hopFlyT > 14) {
        this.hopFly = false
        return false
      }
      if (me.grounded && t - this.hopFlyT > 0.5) {
        // down again: did we make it across?
        this.hopFly = false
        this.hopPlanT = -1e9
        this.replan = true
        this.path.length = 0
        if (this.hopTarget < -1) return false // a roof: the goal logic takes it from here
        if (nav.areaAt(x, y, 3) === this.hopTarget) this.hopFails = 0
        else if (++this.hopFails >= 2) {
          this.hopBan = t + 25
          this.hopFails = 0
        }
        return false
      }
      const dx = p.tx - x
      const dy = p.ty - y
      const d = Math.hypot(dx, dy)
      const ux = d > 1e-6 ? dx / d : 0
      const uy = d > 1e-6 ? dy / d : 0
      // stay above whatever stands on the next stretch of the way
      let need = w.topAt(x, y)
      for (let s = 1; s <= Math.min(d, 30); s++) need = Math.max(need, w.topAt(x + ux * s, y + uy * s))
      // ease off near the landing point so we do not sail past it
      const sp = Math.min(1, (2.2 * d) / Math.max(1, c.walk))
      out.move = d > 0.6 ? [ux * sp, uy * sp] : [0, 0]
      out.boost = z < need + 2.5 && me.energy > 1
      return true
    }
    if (goal.why === "roof-laser") {
      const it = this.roofItem
      if (!it || !me.grounded) return false
      if (z - w.groundAt(x, y) > 3 && Math.abs(z - it.z) < 3) {
        // already up there: walk straight to it
        const d = dist2(x, y, it.x, it.y)
        out.move = d > 0.5 ? [(it.x - x) / d, (it.y - y) / d] : [0, 0]
        out.boost = false
        return true
      }
      if (!this.hopPlan || this.hopTarget !== -2 - it.id || t - this.hopPlanT > 4) {
        this.hopPlan = this.nav.roofPlan(x, y, it.x, it.y, it.z)
        this.hopTarget = -2 - it.id
        this.hopPlanT = t
      }
      const p = this.hopPlan
      if (!p) {
        this.ignore.set(it.id, t + 60)
        return false
      }
      const dl = dist2(x, y, p.lx, p.ly)
      if (dl > 2.5) {
        this.move(obs, c, { x: p.lx, y: p.ly, arrive: 1, why: goal.why, boost: goal.boost && dl > 25, travel: true }, out)
        return true
      }
      out.move = [0, 0]
      out.boost = false
      if (me.energy < Math.min((me.stats.energy ?? 100) - 4, p.energy + 6)) return true
      this.hopFly = true
      this.hopFlyT = t
      out.jump = true
      const d = dist2(x, y, p.tx, p.ty)
      if (d > 1e-6) out.move = [(p.tx - x) / d, (p.ty - y) / d]
      return true
    }
    if (t < this.hopBan) return false
    if (goal.why !== "zone" && goal.why !== "king" && goal.why !== "home") return false
    if (!me.grounded || z - w.groundAt(x, y) > 3) return false
    const mine = nav.areaAt(x, y, 4)
    if (mine < 0) return false
    let target = nav.areaAt(goal.x, goal.y, 14)
    // nothing to do in a walled-off area: do not sit in it until the zone forces the issue
    if (goal.why === "home" && mine !== nav.mainComp) target = nav.mainComp
    if (target < 0 || target === mine) {
      this.hopPlan = null
      return false
    }
    if (!this.hopPlan || this.hopTarget !== target || t - this.hopPlanT > 6) {
      this.hopPlan = nav.hopPlan(x, y, target)
      this.hopTarget = target
      this.hopPlanT = t
    }
    const p = this.hopPlan
    if (!p) return false
    const dl = dist2(x, y, p.lx, p.ly)
    if (dl > 2.5) {
      this.move(obs, c, { x: p.lx, y: p.ly, arrive: 1, why: goal.why, boost: goal.boost && dl > 25, travel: true }, out)
      return true
    }
    out.move = [0, 0]
    out.boost = false
    if (me.energy < Math.min((me.stats.energy ?? 100) - 4, p.energy + 6)) return true // wait for the jets
    this.hopFly = true
    this.hopFlyT = t
    out.jump = true
    const d = dist2(x, y, p.tx, p.ty)
    if (d > 1e-6) out.move = [(p.tx - x) / d, (p.ty - y) / d]
    return true
  }

  private move(obs: Observation, c: Ctx, goal: Goal, out: BotAction) {
    const me = obs.self
    const S = me.stats
    const { x, y, z, t } = c
    const w = this.world
    const vx = me.vel[0]
    const vy = me.vel[1]
    const accel = S.ground_accel ?? 58.5
    const laser = c.sLaser >= 0 ? me.weapons[c.sLaser] : null
    const charging = !!laser && (laser.charge ?? 0) > 0
    const onGround = z - w.groundAt(x, y) < 3
    let boost = goal.boost && me.energy > 6 && !charging
    const speed = (boost ? (S.boost_speed ?? 19.5) : c.walk) * (charging ? (S.charge_move_mult ?? 0.5) : 1)

    const gdist = dist2(x, y, goal.x, goal.y)
    let gdx = 0
    let gdy = 0
    if (gdist > goal.arrive) [gdx, gdy] = this.follow(x, y, goal.x, goal.y, t, onGround)

    // ---- stuck detection: we meant to travel but went nowhere
    const want = goal.travel && gdist > goal.arrive + 4 ? 1 : 0
    const hi = this.hN % 12
    this.hx[hi] = x
    this.hy[hi] = y
    this.hWant[hi] = want
    this.hN++
    if (this.hN >= 12) {
      const old = this.hN % 12
      let all = 1
      for (let k = 0; k < 12; k++) all &= this.hWant[k]
      const moved = dist2(x, y, this.hx[old], this.hy[old])
      if (moved > 6) this.stuckCount = 0
      if (all && moved < 1.8 && t - this.lastStuckT > 1) {
        this.lastStuckT = t
        this.stuckCount++
        this.nav.block(x + gdx * 3.5, y + gdy * 3.5, 3.5, t + 20)
        this.replan = true
        this.pathT = -1e9
        if (this.stuckCount >= 2 || !onGround) this.hopUntil = t + 0.9
        this.hWant.fill(0)
      }
    }
    if (t < this.hopUntil) {
      // hop over whatever the map model missed
      out.move = [gdx, gdy]
      out.jump = me.grounded
      out.boost = me.energy > 10
      return
    }

    // ---- incoming fire
    const shots: { px: number; py: number; ux: number; uy: number; th: number; w: number; land: boolean; lx: number; ly: number; lr: number }[] = []
    const kingFinal = c.king && obs.zone.next_radius <= 0.5 && t > obs.zone.shrink_start
    if (!kingFinal) {
      for (const p of obs.projectiles) {
        const rx = p.pos[0] - x
        const ry = p.pos[1] - y
        const sp2 = p.vel[0] * p.vel[0] + p.vel[1] * p.vel[1]
        if (sp2 < 1) continue
        const sp = Math.sqrt(sp2)
        if (p.kind === "mortar") {
          const g = this.cfg.weapons.mortar.gravity ?? 22
          const dz = p.pos[2] - z
          const tl = (p.vel[2] + Math.sqrt(Math.max(0, p.vel[2] * p.vel[2] + 2 * g * dz))) / g
          const lx = p.pos[0] + p.vel[0] * tl
          const ly = p.pos[1] + p.vel[1] * tl
          const lr = (this.cfg.weapons.mortar.splash_radius ?? 7) + c.radius + 2.5
          if (dist2(lx, ly, x, y) < lr + 14) shots.push({ px: 0, py: 0, ux: 0, uy: 0, th: Math.min(tl, 1.4), w: 2.5, land: true, lx, ly, lr })
          continue
        }
        const th = -(rx * p.vel[0] + ry * p.vel[1]) / sp2
        if (th <= 0 || th > 1.4) continue
        const miss = Math.abs(rx * (p.vel[1] / sp) - ry * (p.vel[0] / sp))
        if (miss > 14) continue
        const bz = p.pos[2] + p.vel[2] * th
        if (bz < z - 2 || bz > z + c.height + 2) continue
        shots.push({ px: p.pos[0], py: p.pos[1], ux: p.vel[0] / sp, uy: p.vel[1] / sp, th, w: p.kind === "npc" ? 0.6 : 1, land: false, lx: 0, ly: 0, lr: 0 })
      }
    }

    // ---- strafe direction against the current target
    // the zone outranks cover: outside it we cannot repair and it only gets worse
    const forced = goal.why === "zone" || goal.why === "king"
    const grabbing = goal.why.startsWith("grab")
    const goalW = forced ? 4 : grabbing ? 2.5 : goal.why === "fight-back" ? tv("kw", 2.5) : 1
    const coverW = forced ? 0.3 : grabbing ? 0.5 : 1
    const mechTarget = c.target && c.target.kind === "mech" ? c.target : null
    const tk = c.cover ? (c.fight || c.evade ? mechTarget : null) : c.fight ? mechTarget : null
    // hide-first: side-step only while it can actually see us, and stand still behind cover
    const jink = !tk ? 0 : c.cover ? (c.targetLos ? (c.fight ? 0.7 : 0.4) : 0) : c.targetLos ? tv("jinkLos", 0.7) : tv("jinkNo", 0.2)
    let sx = 0
    let sy = 0
    if (tk) {
      if (t >= this.strafeFlipT) {
        this.strafe = -this.strafe
        this.strafeFlipT = t + tv("flipA", 0.45) + this.rng() * tv("flipB", 0.7)
      }
      const d = dist2(x, y, tk.x, tk.y)
      if (d > 0.5) {
        sx = (-(tk.y - y) / d) * this.strafe
        sy = ((tk.x - x) / d) * this.strafe
      }
    }
    const zSoon = this.zoneAt(obs, t + 1)
    const insideNow = dist2(x, y, zSoon.cx, zSoon.cy) < zSoon.r - 2
    const curClear = this.nav.clearAt(x, y)
    const zn = obs.zone

    let best = -1e9
    let bx = 0
    let by = 0
    const pred = (dx: number, dy: number, sp: number, tau: number): [number, number] => {
      const ex = dx * sp - vx
      const ey = dy * sp - vy
      const dv = Math.hypot(ex, ey)
      const tr = dv / accel
      if (tr < 1e-6) return [x + vx * tau, y + vy * tau]
      if (tau <= tr) return [x + vx * tau + 0.5 * (ex / tr) * tau * tau, y + vy * tau + 0.5 * (ey / tr) * tau * tau]
      return [x + (vx + 0.5 * ex) * tr + dx * sp * (tau - tr), y + (vy + 0.5 * ey) * tr + dy * sp * (tau - tr)]
    }
    const evalDir = (dx: number, dy: number, moving: boolean) => {
      const sp = moving ? speed : 0
      const [p1x, p1y] = pred(dx, dy, sp, tv("h1", 0.45))
      const [p2x, p2y] = pred(dx, dy, sp, 1.0)
      let s = (dx * gdx + dy * gdy) * goalW
      if (onGround && moving && !this.nav.segOk(x, y, p1x, p1y, 0.5)) {
        // walking into a wall; allowed only if it gets us out of a tight spot
        if (!(curClear < this.nav.need && this.nav.clearAt(p1x, p1y) > curClear)) s -= 6
      }
      for (const sh of shots) {
        const [mx, my] = pred(dx, dy, sp, sh.th)
        if (sh.land) {
          const d = dist2(mx, my, sh.lx, sh.ly)
          if (d < sh.lr) s -= tv("dodgeW", 10) * sh.w * (1 - d / sh.lr)
        } else {
          const miss = Math.abs((mx - sh.px) * sh.uy - (my - sh.py) * sh.ux)
          const safe = c.radius + tv("dodgeR", 2)
          if (miss < safe) {
            const q = 1 - miss / safe
            s -= tv("dodgeW", 10) * sh.w * q * q
          }
        }
      }
      if (c.king) s -= dist2(p1x, p1y, zn.next_center[0], zn.next_center[1]) * 0.8
      else if (insideNow && dist2(p2x, p2y, zSoon.cx, zSoon.cy) > zSoon.r - 2) s -= 2.5
      if (tk) {
        s += jink * (dx * sx + dy * sy)
        if (c.wantLos !== 0) {
          const tz = tk.z + tk.h * 0.5
          const l1 = w.losClear(p1x, p1y, w.groundAt(p1x, p1y) + c.mh, tk.x, tk.y, tz)
          const l2 = w.losClear(p2x, p2y, w.groundAt(p2x, p2y) + c.mh, tk.x, tk.y, tz)
          s += coverW * c.wantLos * ((l1 ? 1 : -1) * tv("losW1", 1.3) + (l2 ? 0.6 : -0.6))
        }
      }
      if (c.evade || c.power || c.threats.length > 1) {
        const wgt = (c.evade ? 1.2 : c.power ? 0.7 : tv("multiCover", 0.45)) * coverW
        const g1 = w.groundAt(p1x, p1y) + c.height * 0.5
        let n = 0
        for (const k of c.threats) {
          if (tk && k.id === tk.id) continue
          if (++n > 6) break
          if (w.losClear(k.x, k.y, k.z + k.h * 0.89, p1x, p1y, g1)) s -= wgt
        }
      }
      s += tv("smooth", 0.15) * (dx * this.lastDX + dy * this.lastDY)
      if (s > best) {
        best = s
        bx = moving ? dx : 0
        by = moving ? dy : 0
      }
    }
    if (gdx !== 0 || gdy !== 0) evalDir(gdx, gdy, true)
    for (let k = 0; k < NDIR; k++) evalDir(DIRX[k], DIRY[k], true)
    evalDir(0, 0, false)

    // ease into a precise stop
    let mag = 1
    if (!tk && shots.length === 0 && gdist <= Math.max(goal.arrive, 0) + 3 && goal.arrive < 100) mag = clamp((gdist - goal.arrive * 0.5) / 3, 0, 1)
    if (c.king && gdist < 3) mag = clamp(gdist / 2, 0.15, 1)
    this.lastDX = bx
    this.lastDY = by
    if (c.evade && c.exposed && me.energy > 30 && !charging) boost = true
    out.move = [bx * mag, by * mag]
    out.boost = boost && (bx !== 0 || by !== 0)
  }

  // ---------------------------------------------------------------------------------- aim and fire
  private aimAndFire(obs: Observation, c: Ctx, out: BotAction) {
    const me = obs.self
    const S = me.stats
    const { x, y, z, t } = c
    const w = this.world
    const mz = z + c.mh
    const fire: [boolean, boolean] = [false, false]
    const minPitch = S.min_pitch ?? -0.7
    const maxPitch = S.max_pitch ?? 1.3
    const T = c.target
    const cubeMult = 1 + (Number(this.cfg.cubes.damage_per_cube) || 0) * Math.min(me.cubes, Number(this.cfg.cubes.stat_cap) || 10)
    let yaw = me.aim[0]
    let pitch = 0
    let haveAim = false

    // while hiding to repair, do not shoot unless we are being hit anyway or the kill is there:
    // dealing damage restarts the 5 s regeneration timer just like taking it
    const mechT = !!T && T.kind === "mech"
    const repairing = mechT && c.evade && c.hpFrac < 0.85 && t - this.lastHurtT > 1 && T!.hp > 0.25 * T!.maxHp && c.targetDist > 45
    // free-for-all: do not pick a fight with a distant mech that is leaving us alone
    const unprovoked = mechT && c.ambush && !c.fight && t - this.lastMechHurtT > 3 && T!.hp > 0.35 * T!.maxHp
    // optional pacing: each kill brings another hunter, so after paceK kills wait for the first supply
    // drop (a full repair) before making more, unless the mech is nearly dead or on top of us
    const pacing = mechT && tv("paceK", 0) > 0 && this.myKills >= tv("paceK", 0) && this.dropsTaken === 0 && t < tv("paceT", 104) && !c.ambush && T!.hp > 0.3 * T!.maxHp && c.targetDist > 50 && c.hpFrac > 0.45
    const holdFire = repairing || unprovoked || pacing
    // a charging laser halves our speed, but in sparring the shots were worth far more than the
    // lost metres, so keep charging on the run unless we are nearly dead
    const mayCharge = !c.evade || c.threats.length <= 1 || (CHARGE_ON_RUN && c.hpFrac > tv("runChargeHp", 0))

    if (T) {
      const frac = T.aimFrac ?? 0.5
      const cz = T.z + T.h * frac
      // where the target will be relative to our muzzle a moment from now: both of us are moving
      const ext = (lat: number) => [T.x + (T.vx - me.vel[0]) * lat - x, T.y + (T.vy - me.vel[1]) * lat - y, cz + (T.vz - me.vel[2]) * lat - mz]
      const [dx, dy, dz] = ext(0.05)
      const dist = Math.hypot(dx, dy, dz)
      const dYaw = Math.atan2(dy, dx)
      const dPitch = Math.atan2(dz, Math.hypot(dx, dy))
      const tolYaw = Math.atan2(T.r * tv("tolYaw", 0.6), dist) + 0.002
      // room above and below the aim point on its body (little above when we aim at the shoulders)
      const tolPitch = Math.atan2(T.h * 0.8 * Math.min(frac, 1 - frac), dist) + 0.002
      const errYaw = Math.abs(wrap(dYaw - me.aim[0]))
      const errPitch = Math.abs(dPitch - me.aim[1])
      const los = c.targetLos
      const [ax, ay, az] = ext(0.1)
      yaw = Math.atan2(ay, ax)
      pitch = Math.atan2(az, Math.hypot(ax, ay))
      haveAim = true
      let prio = 0

      for (let s = 0; s < 2; s++) {
        const wp = me.weapons[s]
        if (!wp) continue
        const st = wp.stats
        if (wp.kind === "gun") {
          const [lx0, ly0, lz0] = ext(tv("lat", 0.075))
          const tl = interceptTime(lx0, ly0, lz0, T.vx, T.vy, T.vz, st.speed ?? 252)
          if (tl <= 0) continue
          const lx = lx0 + T.vx * tl
          const ly = ly0 + T.vy * tl
          const lz = lz0 + T.vz * tl
          const gy = Math.atan2(ly, lx)
          const gp = Math.atan2(lz, Math.hypot(lx, ly))
          const ready = (wp.ammo ?? 1) > 0 && (wp.reload ?? 0) <= 0
          const inRange = (st.speed ?? 252) * tl <= (st.range ?? 230) - 4
          const worth = T.kind !== "mech" || tl <= tv("gunT", 0.8) || Math.hypot(T.vx, T.vy) < 3
          let onAim = Math.abs(wrap(gy - me.aim[0])) <= tolYaw && Math.abs(gp - me.aim[1]) <= tolPitch
          if (!onAim && tv("predAim", 1) > 0) {
            const turn = (S.turn_rate ?? 4) * 0.05
            const tilt = (S.pitch_rate ?? 3) * 0.05
            const ey = wrap(gy - me.aim[0])
            const ep = gp - me.aim[1]
            onAim = Math.abs(ey - clamp(ey, -turn, turn)) <= tolYaw && Math.abs(ep - clamp(ep, -tilt, tilt)) <= tolPitch
          }
          if (ready && inRange && worth && los && onAim && !holdFire) fire[s] = true
          if (ready && inRange && prio < 1) {
            prio = 1
            yaw = gy
            pitch = gp
          }
        } else if (wp.kind === "laser") {
          const full = st.charge ?? 0.75
          const ch = wp.charge ?? 0
          const inRange = dist <= (st.range ?? 280) - 3
          // charge ahead of time when it was just in view, or is walking up to our corner
          const closing = dist > 1 ? -(dx * T.vx + dy * T.vy) / dist : 0
          const recently = t - this.lastSawTargetT < tv("preCharge", 1.5) || (dist < 70 && closing > 2)
          let hold = false
          if (inRange && (ch > 0 || (wp.cooldown <= 0.05 && mayCharge && !holdFire)) && (los || recently || ch > 0)) hold = true
          if (ch > 0) {
            const minFrac = st.min_frac ?? 0.35
            let dmg = (st.damage ?? 315) * (minFrac + (1 - minFrac) * Math.min(1, ch / full)) * cubeMult
            if (T.weapons.includes("mortar")) dmg *= 1 + (st.cookoff ?? 0)
            const onAim = los && inRange && errYaw <= tolYaw && errPitch <= tolPitch
            if (onAim && ch >= (st.min_charge ?? 0.15) && (ch >= full - 0.02 || dmg >= T.hp)) hold = false
            if (ch >= full - 0.3 && prio < 3) {
              prio = 3
              yaw = Math.atan2(ay, ax)
              pitch = Math.atan2(az, Math.hypot(ax, ay))
            }
          }
          fire[s] = hold
        } else if (wp.kind === "lightning") {
          const range = st.range ?? 45
          const cone = ((st.cone_deg ?? 30) * Math.PI) / 180
          const reserve = T.kind === "mech" ? 0 : 40
          if (dist <= range - 2 && los && me.energy >= (st.energy ?? 3) + reserve && errYaw <= cone * 0.45 && !holdFire) fire[s] = true
        } else if (wp.kind === "mortar") {
          const v = st.speed ?? 72
          const g = st.gravity ?? 22
          const tz0 = z + c.height + 0.5
          let px = T.x
          let py = T.y
          let sol: number | null = null
          let tf = 0
          for (let it = 0; it < 2; it++) {
            const d = dist2(x, y, px, py)
            sol = ballisticPitch(d, T.z + 1 - tz0, v, g, false)
            if (sol == null) break
            tf = d / (v * Math.cos(sol))
            px = T.x + T.vx * Math.min(tf, 1.5)
            py = T.y + T.vy * Math.min(tf, 1.5)
          }
          const d = dist2(x, y, px, py)
          const slow = Math.hypot(T.vx, T.vy) < 4
          const gun = c.sGun >= 0 ? me.weapons[c.sGun] : null
          const gunBusy = !gun || (gun.ammo ?? 1) <= 0 || (gun.reload ?? 0) > 0
          if (sol != null && sol <= maxPitch - 0.02 && wp.cooldown <= 0.15 && d >= (st.splash_radius ?? 7) + c.radius + 5 && (slow || tf < 0.85) && los && T.kind !== "crate") {
            const my = Math.atan2(py - y, px - x)
            if ((gunBusy || prio < 1) && prio < 2) {
              prio = 2
              yaw = my
              pitch = sol
            }
            if (Math.abs(wrap(my - me.aim[0])) < 0.025 && Math.abs(sol - me.aim[1]) < 0.02 && !holdFire) fire[s] = true
          }
        }
      }
    }

    if (!haveAim) {
      // nothing to shoot: look where trouble is, else sweep so we spot loot and mechs
      const moving = Math.hypot(me.vel[0], me.vel[1]) > 2
      if (!Number.isFinite(this.scanYaw0)) this.scanYaw0 = me.aim[0]
      if (t >= this.nextScanT && c.threats.length === 0) {
        this.scanStartT = t
        this.scanUntil = t + 1.8
        this.scanYaw0 = me.aim[0]
        this.nextScanT = t + 11
      }
      let recent: Track | null = null
      for (const k of this.tracks.values()) if (k.kind === "mech" && t - k.seen < 12 && (!recent || k.seen > recent.seen)) recent = k
      if (this.alertUntil > t) yaw = this.alertYaw
      else if (t < this.scanUntil) yaw = this.scanYaw0 + (t - this.scanStartT + 0.1) * (S.turn_rate ?? 4) * 0.9
      else if (recent) yaw = Math.atan2(recent.y - y, recent.x - x)
      else if (moving) yaw = Math.atan2(me.vel[1], me.vel[0]) + 0.9 * Math.sin(t * 1.3)
      else yaw = me.aim[0] + 0.35
      pitch = 0
    }
    out.aim = [wrap(yaw), clamp(pitch, minPitch, maxPitch)]
    out.fire = fire

    // reload when empty, or top up while nobody is in sight
    const gun = c.sGun >= 0 ? me.weapons[c.sGun] : null
    if (gun && (gun.reload ?? 0) <= 0) {
      const ammo = gun.ammo ?? 1
      const mag = gun.stats.mag ?? 12
      if (ammo <= 0 || (!c.target && ammo < mag * tv("relIdle", 0.7) && t - this.lastSawTargetT > 1.5) || (c.fight && !c.targetLos && ammo < mag * tv("relFight", 0.34) && t - this.lastSawTargetT > 0.8)) out.reload = true
    }
  }

  // ------------------------------------------------------------------------------- upgrades / loot
  private pickUpgrade(obs: Observation): number {
    const ch = obs.offer?.choices ?? []
    let best = 0
    let rank = Infinity
    for (let i = 0; i < ch.length; i++) {
      let r = (UPGRADE_ALT[tv("upg", 0)] ?? UPGRADE_ORDER).indexOf(ch[i].id)
      if (r < 0) r = 50
      if (r < rank) {
        rank = r
        best = i
      }
    }
    return best
  }

  private loot(obs: Observation, c: Ctx, out: BotAction) {
    const me = obs.self
    const reach = (this.cfg.loot.pickup_radius ?? 4) - 0.3
    let best = 0
    let pick: { id: number; kind: string } | null = null
    for (const p of obs.pickups) {
      if (dist2(c.x, c.y, p.pos[0], p.pos[1]) > reach || Math.abs(p.pos[2] - c.z) > 3.7) continue
      if ((this.ignore.get(p.id) ?? -1) > c.t) continue
      const g = this.pickupGain(me.weapons, p.kind, c.zone.r)
      if (g > best) {
        best = g
        pick = p
      }
    }
    if (!pick) return
    if (me.weapons.some((w) => !w)) {
      out.take = pick.id
      return
    }
    // swap out the weakest non-gun weapon (a charging laser cannot be dropped)
    let slot = -1
    let worst = Infinity
    for (let s = 0; s < 2; s++) {
      const w = me.weapons[s]
      if (!w || w.kind === "gun") continue
      if (w.kind === "laser" && (w.charge ?? 0) > 0) continue
      const v = WEAPON_VALUE[w.kind] ?? 0
      if (v < worst) {
        worst = v
        slot = s
      }
    }
    if (slot >= 0) {
      out.take = pick.id
      out.drop = slot
    }
  }
}
