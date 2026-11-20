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
const WEAPON_VALUE: Record<string, number> = { laser: 3, lightning: 2.2, gun: 2, mortar: 1 }
/** upgrade preference, best first: survive the final burn, then win duels */
const UPGRADE_ORDER = [
  "plating", "nanites", "armor", "servos", "laser_damage", "gun_damage", "jets", "reactor", "capacitor", "gun_rate", "laser_charge",
  "lightning_efficiency", "lightning_range", "lightning_chain", "gun_mag", "gun_velocity", "gyros", "sensors", "laser_pierce",
  "mortar_reload", "mortar_splash", "mortar_velocity",
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
  private alertYaw = 0
  private alertUntil = -1
  private lastHurtT = -1e9
  private lastHurtBy = -1
  /** last time a mech (not a drone or the zone) damaged us */
  private lastMechHurtT = -1e9
  private lastThreatT = -1e9
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

  constructor(hello: Hello) {
    this.cfg = hello.config
    this.world = new World(hello.map.dir, hello.map.files)
    this.nav = new Nav(this.world)
    this.rng = mulberry32((hello.seed ^ 0x9e3779b9) >>> 0)
    this.warmUp()
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
  }

  // ------------------------------------------------------------------------------------------ main
  act(obs: Observation): BotAction {
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
    this.move(obs, c, goal, out)
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
      } else if (e.type === "kill") this.tracks.delete(e.target)
    }
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
    const key = `${Math.round(hx / 10)},${Math.round(hy / 10)},${z.shrink_end}`
    if (key === this.idleKey && Number.isFinite(this.idleX)) return [this.idleX, this.idleY]
    this.idleKey = key
    let best = Infinity
    let bx = hx
    let by = hy
    const need = this.nav.need
    for (let r = 0; r <= 44; r += 4) {
      const n = r === 0 ? 1 : NDIR
      for (let k = 0; k < n; k++) {
        const px = hx + DIRX[k] * r
        const py = hy + DIRY[k] * r
        if (dist2(px, py, z.next_center[0], z.next_center[1]) > z.next_radius - 8) continue
        const cl = this.nav.clearAt(px, py)
        if (cl < need + 0.3 || cl > need + 3.5 || !this.nav.free(px, py)) continue
        if (r < best) {
          best = r
          bx = px
          by = py
        }
      }
      if (best < Infinity) break
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
      let s = -d / 60 - (k.hp / Math.max(1, k.maxHp)) * 1.5 + (los ? 2.5 : 0)
      if (k.id === this.targetId) s += 0.6
      if (k.id === this.lastHurtBy && t - this.lastHurtT < 4) s += 1
      if (s > bestScore) {
        bestScore = s
        target = { ...k, aimFrac: frac }
        targetLos = los
        targetDist = d
      }
    }
    let fight = false
    if (target) {
      this.targetId = target.id
      const hurtByMech = this.lastHurtBy >= 0 && this.lastHurtBy < 100 && t - this.lastHurtT < 4
      fight = (targetLos && targetDist <= 175) || targetDist <= 60 || hurtByMech
      if (targetLos) this.lastSawTargetT = t
    } else this.targetId = -1

    // fight or run (with hysteresis)
    const nT = threats.length
    let evade = this.evading
    const tk0 = target as Aimable | null
    const finishable = !!tk0 && tk0.hp < me.hp * 0.5
    // a laser out-trades a gun at range: do not stand in the open against one
    const outgunned = !!tk0 && tk0.weapons.includes("laser") && sLaser < 0 && targetDist > 55
    if (!evade) {
      if (nT > 0 && hpFrac < 0.4 && !finishable) evade = true
      if (nT >= 3) evade = true
      if (nT === 2 && hpFrac < 0.65) evade = true
      if (unseenShooter && underFire && !tk0) evade = true
      if (outgunned && hpFrac < 0.8 && !finishable) evade = true
    } else if ((nT === 0 && (hpFrac > 0.75 || t - this.lastThreatT > 8)) || (nT === 1 && hpFrac > 0.85 && !unseenShooter && !outgunned) || (finishable && nT === 1)) evade = false
    if (king) evade = false
    this.evading = evade
    if (evade) fight = false

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
      const calm = nT === 0 && !underFire && me.cubes < 14 && me.in_zone
      for (const k of this.tracks.values()) {
        if (k.kind !== "npc" || k.seen < t - 0.01) continue
        const angry = k.id === this.lastHurtBy && t - this.lastHurtT < 6
        if (angry) consider({ ...k, kind: "npc" }, 200, 1000)
        else if (calm && hpFrac > 0.6) consider({ ...k, kind: "npc" }, 110, 0)
      }
      if (calm) {
        const cr = this.cfg.crates
        for (const it of obs.crates) consider({ id: it.id, kind: "crate", x: it.pos[0], y: it.pos[1], z: it.pos[2], vx: 0, vy: 0, vz: 0, r: cr.radius ?? 1.4, h: cr.height ?? 2.4, hp: it.hp ?? 300, maxHp: cr.hp ?? 300, weapons: [] }, 150, 60)
      }
    }

    // peek logic: want a line of fire when a weapon is ready, want to be hidden when none is
    let wantLos = 0
    if (fight && target) {
      let ready = false
      const gun = sGun >= 0 ? me.weapons[sGun] : null
      const laser = sLaser >= 0 ? me.weapons[sLaser] : null
      const light = sLight >= 0 ? me.weapons[sLight] : null
      if (gun && (gun.ammo ?? 1) > 0 && (gun.reload ?? 0) <= 0.25) ready = true
      let laserHot = false
      if (laser && (laser.charge ?? 0) >= (laser.stats.charge ?? 0.75) - 0.3) {
        ready = true
        laserHot = true
      }
      if (light && targetDist <= (light.stats.range ?? 45) + 6 && me.energy >= (light.stats.energy ?? 3) * 3) ready = true
      let hide = false
      for (const k of threats) {
        if (!k.charging || k.seen < t - 0.3) continue
        const toMe = Math.atan2(y - k.y, x - k.x)
        if (Math.abs(wrap(toMe - k.yaw)) < 0.3 && dist2(x, y, k.x, k.y) < 300) hide = true
      }
      wantLos = hide && !laserHot ? -1.6 : ready ? 1 : -1
    }

    return { x, y, z, t, radius, height, mh, hpFrac, walk, zone, insideNext, mustGo, king, evade, fight, threats, target, targetLos, targetDist, wantLos, sGun, sLaser, sLight, sMortar, exposed }
  }

  // ------------------------------------------------------------------------------------------ goal
  private chooseGoal(obs: Observation, c: Ctx): Goal {
    const me = obs.self
    const zn = obs.zone
    const { x, y, t } = c
    const stay: Goal = { x, y, arrive: 1e9, why: "hold", boost: false, travel: false }

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
    if (c.evade) {
      const [cx, cy] = this.coverSpot(obs, c)
      return { x: cx, y: cy, arrive: 1.5, why: "evade", boost: c.exposed && me.energy > 30, travel: true }
    }
    if (c.fight && c.target) {
      const k = c.target
      const d = dist2(x, y, k.x, k.y)
      let rmin = 45
      let rmax = 85
      if (c.sLaser >= 0) {
        rmin = 70
        rmax = 140
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
        const bx = x + ((x - k.x) / d) * 18
        const by = y + ((y - k.y) / d) * 18
        if (this.nav.free(bx, by) && dist2(bx, by, c.zone.cx, c.zone.cy) < c.zone.r - 6) return { x: bx, y: by, arrive: 2, why: "fight-back", boost: false, travel: false }
      }
      return { ...stay, why: "fight" }
    }

    // ---- nobody to deal with: improve
    const live = c.zone
    const late = t > zn.shrink_start - 20
    const okSpot = (px: number, py: number) => (late ? dist2(px, py, zn.next_center[0], zn.next_center[1]) < zn.next_radius - 4 : dist2(px, py, live.cx, live.cy) < live.r - 6)
    const quiet = c.threats.length === 0

    // supply drop: 3 cubes and a full repair
    for (const d of obs.drops) {
      const dd = dist2(x, y, d.pos[0], d.pos[1])
      if (dd > 280 || d.lands_in > dd / c.walk + 16) continue
      if (!okSpot(d.pos[0], d.pos[1]) && d.lands_in > 0) continue
      if (!quiet && c.hpFrac < 0.5) continue
      return { x: d.pos[0], y: d.pos[1], arrive: 1.5, why: "drop", boost: false, travel: true }
    }
    // cubes lying around
    let best: Item | null = null
    let bd = 90
    for (const it of this.cubes.values()) {
      if (it.roof || !okSpot(it.x, it.y)) continue
      const d = dist2(x, y, it.x, it.y)
      if (d < bd) {
        bd = d
        best = it
      }
    }
    if (best) return { x: best.x, y: best.y, arrive: 0.8, why: "cube", boost: false, travel: true }

    // a better weapon
    let bw: Item | null = null
    let bs = 0
    for (const it of this.pickups.values()) {
      if (it.roof || !okSpot(it.x, it.y) || (this.ignore.get(it.id) ?? -1) > t) continue
      const gain = this.pickupGain(me.weapons, it.kind, live.r)
      if (gain <= 0) continue
      const d = dist2(x, y, it.x, it.y)
      if (d > (me.weapons[1] && me.weapons[0] ? 150 : 320)) continue
      const s = gain / (d + 25)
      if (s > bs) {
        bs = s
        bw = it
      }
    }
    if (bw) return { x: bw.x, y: bw.y, arrive: 1.2, why: `get-${bw.kind}`, boost: false, travel: true }

    // crates: a cube each and maybe a weapon
    if (quiet && me.cubes < 14) {
      let bc: Item | null = null
      let bcd = 200
      for (const it of this.crates.values()) {
        if (it.roof || !okSpot(it.x, it.y)) continue
        const d = dist2(x, y, it.x, it.y)
        if (d < bcd) {
          bcd = d
          bc = it
        }
      }
      if (bc) {
        if (c.target && c.target.id === bc.id) return { ...stay, why: "crate" }
        return { x: bc.x, y: bc.y, arrive: 35, why: "to-crate", boost: false, travel: true }
      }
    }
    if (c.target) return { ...stay, why: "farm" }

    const [hx, hy] = this.homePoint(obs, c)
    const [ix, iy] = this.idleSpot(hx, hy, zn)
    return { x: ix, y: iy, arrive: 2, why: "home", boost: false, travel: true }
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

  /** best nearby spot hidden from everyone hunting us */
  private coverSpot(obs: Observation, c: Ctx): [number, number] {
    const { x, y, t } = c
    const w = this.world
    const hiddenFrom = (px: number, py: number): number => {
      const gz = w.groundAt(px, py)
      let seen = 0
      for (const k of c.threats) {
        const ez = k.z + k.h * 0.89
        if (w.losClear(k.x, k.y, ez, px, py, gz + c.height * 0.5) || w.losClear(k.x, k.y, ez, px, py, gz + c.height - 0.5)) seen++
      }
      return seen
    }
    if (Number.isFinite(this.coverX) && t - this.coverT < 1.2 && hiddenFrom(this.coverX, this.coverY) === 0) return [this.coverX, this.coverY]
    const zn = obs.zone
    const soon = this.zoneAt(obs, t + 6)
    let best = -1e9
    let bx = x
    let by = y
    const rings = [0, 8, 16, 26, 38, 52, 70]
    for (const r of rings) {
      const n = r === 0 ? 1 : NDIR
      for (let k = 0; k < n; k++) {
        const px = x + DIRX[k] * r
        const py = y + DIRY[k] * r
        if (!this.nav.free(px, py)) continue
        if (dist2(px, py, soon.cx, soon.cy) > soon.r - 5) continue
        let s = -3 * hiddenFrom(px, py) - r / 35
        let near = Infinity
        for (const th of c.threats) near = Math.min(near, dist2(px, py, th.x, th.y))
        if (near < Infinity) s += Math.min(near, 150) / 60
        if (dist2(px, py, zn.next_center[0], zn.next_center[1]) > zn.next_radius - 5) s -= 1
        if (s > best) {
          best = s
          bx = px
          by = py
        }
      }
    }
    this.coverX = bx
    this.coverY = by
    this.coverT = t
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
    const tk = c.fight ? c.target : null
    let sx = 0
    let sy = 0
    if (tk) {
      if (t >= this.strafeFlipT) {
        this.strafe = -this.strafe
        this.strafeFlipT = t + 0.45 + this.rng() * 0.7
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
      const [p1x, p1y] = pred(dx, dy, sp, 0.45)
      const [p2x, p2y] = pred(dx, dy, sp, 1.0)
      let s = dx * gdx + dy * gdy
      if (onGround && moving && !this.nav.segOk(x, y, p1x, p1y, 0.5)) {
        // walking into a wall; allowed only if it gets us out of a tight spot
        if (!(curClear < this.nav.need && this.nav.clearAt(p1x, p1y) > curClear)) s -= 6
      }
      for (const sh of shots) {
        const [mx, my] = pred(dx, dy, sp, sh.th)
        if (sh.land) {
          const d = dist2(mx, my, sh.lx, sh.ly)
          if (d < sh.lr) s -= 4 * sh.w * (1 - d / sh.lr)
        } else {
          const miss = Math.abs((mx - sh.px) * sh.uy - (my - sh.py) * sh.ux)
          const safe = c.radius + 2
          if (miss < safe) {
            const q = 1 - miss / safe
            s -= 4 * sh.w * q * q
          }
        }
      }
      if (c.king) s -= dist2(p1x, p1y, zn.next_center[0], zn.next_center[1]) * 0.8
      else if (insideNow && dist2(p2x, p2y, zSoon.cx, zSoon.cy) > zSoon.r - 2) s -= 2.5
      if (tk) {
        s += (c.targetLos ? 0.7 : 0.2) * (dx * sx + dy * sy)
        if (c.wantLos !== 0) {
          const tz = tk.z + tk.h * 0.5
          const l1 = w.losClear(p1x, p1y, w.groundAt(p1x, p1y) + c.mh, tk.x, tk.y, tz)
          const l2 = w.losClear(p2x, p2y, w.groundAt(p2x, p2y) + c.mh, tk.x, tk.y, tz)
          s += c.wantLos * ((l1 ? 1.3 : -1.3) + (l2 ? 0.6 : -0.6))
        }
      }
      if (c.evade || c.threats.length > 1) {
        const wgt = c.evade ? 1.2 : 0.45
        const g1 = w.groundAt(p1x, p1y) + c.height * 0.5
        let n = 0
        for (const k of c.threats) {
          if (tk && k.id === tk.id) continue
          if (++n > 6) break
          if (w.losClear(k.x, k.y, k.z + k.h * 0.89, p1x, p1y, g1)) s -= wgt
        }
      }
      s += 0.15 * (dx * this.lastDX + dy * this.lastDY)
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
    const holdFire = mechT && c.evade && c.hpFrac < 0.8 && t - this.lastHurtT > 1 && T!.hp > 0.25 * T!.maxHp
    // a charging laser halves our speed: when running from several mechs keep the legs free
    const mayCharge = !c.evade || c.threats.length <= 1

    if (T) {
      const cz = T.z + T.h * (T.aimFrac ?? 0.5)
      const ext = (lat: number) => [T.x + T.vx * lat - x, T.y + T.vy * lat - y, cz + T.vz * lat - mz]
      const [dx, dy, dz] = ext(0.05)
      const dist = Math.hypot(dx, dy, dz)
      const dYaw = Math.atan2(dy, dx)
      const dPitch = Math.atan2(dz, Math.hypot(dx, dy))
      const tolYaw = Math.atan2(T.r * 0.75, dist) + 0.004
      const tolPitch = Math.atan2(T.h * 0.4, dist) + 0.004
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
          const [lx0, ly0, lz0] = ext(0.075)
          const tl = interceptTime(lx0, ly0, lz0, T.vx, T.vy, T.vz, st.speed ?? 252)
          if (tl <= 0) continue
          const lx = lx0 + T.vx * tl
          const ly = ly0 + T.vy * tl
          const lz = lz0 + T.vz * tl
          const gy = Math.atan2(ly, lx)
          const gp = Math.atan2(lz, Math.hypot(lx, ly))
          const ready = (wp.ammo ?? 1) > 0 && (wp.reload ?? 0) <= 0
          const inRange = (st.speed ?? 252) * tl <= (st.range ?? 230) - 4
          const worth = T.kind !== "mech" || tl <= 0.8 || Math.hypot(T.vx, T.vy) < 3
          const onAim = Math.abs(wrap(gy - me.aim[0])) <= tolYaw && Math.abs(gp - me.aim[1]) <= tolPitch
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
          const recently = t - this.lastSawTargetT < 1.5
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
      if (ammo <= 0 || (!c.target && ammo < mag * 0.7 && t - this.lastSawTargetT > 1.5) || (c.fight && !c.targetLos && ammo < mag * 0.34 && t - this.lastSawTargetT > 0.8)) out.reload = true
    }
  }

  // ------------------------------------------------------------------------------- upgrades / loot
  private pickUpgrade(obs: Observation): number {
    const ch = obs.offer?.choices ?? []
    let best = 0
    let rank = Infinity
    for (let i = 0; i < ch.length; i++) {
      let r = UPGRADE_ORDER.indexOf(ch[i].id)
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
