/** Entry point: mechfight/1 bot. Run with `bun bot.ts` or `node bot.ts`. */
import { type BotAction, type Hello, log, type Observation, run } from "./mechfight.ts"
import { Brain } from "./src/brain.ts"

/** Minimal behaviour used only if the real bot fails: head for the zone and shoot what is in sight. */
function fallback(obs: Observation): BotAction {
  const me = obs.self
  const [x, y, z] = me.pos
  const g = obs.zone.next_center
  const d = Math.hypot(g[0] - x, g[1] - y)
  const act: BotAction = { move: d > 3 ? [(g[0] - x) / d, (g[1] - y) / d] : [0, 0], boost: !me.in_zone }
  let best: Observation["mechs"][number] | null = null
  let bd = Infinity
  for (const m of obs.mechs) {
    if (m.kind !== "mech") continue
    const dd = Math.hypot(m.pos[0] - x, m.pos[1] - y)
    if (dd < bd) {
      bd = dd
      best = m
    }
  }
  if (best) {
    act.aim = [Math.atan2(best.pos[1] - y, best.pos[0] - x), Math.atan2(best.pos[2] - z - 2.5, bd)]
    act.fire = [true, false]
  }
  if (obs.offer) act.pick = 0
  return act
}

run((hello: Hello) => {
  let brain: Brain | null = null
  try {
    const t0 = performance.now()
    brain = new Brain(hello)
    log(`ready: map loaded in ${(performance.now() - t0).toFixed(0)} ms`)
  } catch (e) {
    log("brain failed to start, using fallback:", e)
  }
  let errors = 0
  return {
    name: "magic-royale",
    act(obs: Observation): BotAction {
      if (brain) {
        try {
          return brain.act(obs)
        } catch (e) {
          if (++errors <= 5 || errors % 200 === 0) log(`act failed at tick ${obs.tick} (${errors}):`, e)
        }
      }
      return fallback(obs)
    },
  }
})
