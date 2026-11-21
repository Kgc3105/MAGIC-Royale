# magic-royale

A bot for the Builders Mech Royale track (`mechfight/1`): it pilots one mech in a 16-mech battle royale, answering ten observations a second with a move, an aim and what to fire.

## How it plays

- **Map.** At start-up it rasterises the terrain and every building onto a 1 m grid. That grid answers "can I walk there", "can I see or shoot that" and "where is cover".
- **Movement.** A* over a 2 m grid routes around buildings. If the mech stops making progress it marks the spot as blocked, replans, and hops with its jets.
- **Zone.** It heads for the next circle before the current one closes, and in the last phases goes to the exact centre, where every metre away costs HP.
- **Fighting.** It leads gun shots, charges the laser and releases only when on target, steps out of cover when a weapon is ready and back in when none is, and dodges bullets and mortar shells it can see. A mech that shoots it from out of view is located from the hit's bearing or the beam.
- **Surviving.** When outnumbered, out-gunned or low on HP it breaks line of sight, holds fire so repair can start, and comes back when healed.
- **Growing.** It collects weapons (laser first), cubes, crates and supply drops, and takes HP and repair upgrades first. A cube or a laser close by is grabbed even mid-fight, because early power decides more than position does.
- **Two postures.** Against the practice squad, which never fights itself, a house bot only dies if we kill it, so the bot takes fights while healthy. Once mechs start dying that it did not kill, the match is a free-for-all: it switches to hide-first, stays behind cover, and only trades shots with a mech that is close, is shooting it, or is nearly dead.

## Layout

| Path | What it is |
| --- | --- |
| `bot.ts` | Entry point, with a minimal fallback if the main logic fails |
| `src/world.ts` | Map model: ground height, line of sight, wall clearance |
| `src/nav.ts` | A* path planning |
| `src/aim.ts` | Intercept and ballistic maths |
| `src/brain.ts` | Decisions: zone, fight, evade, loot, aim and fire |
| `mechfight.ts` | The organisers' TypeScript wrapper (stdin/stdout loop and types) |
| `tools/sim.ts` | Local sparring simulator (our own approximation, not the real server) |

## Run it

The organisers' kit (`mechfight-kit/`) is not in this repository; download it from the track page and extract it here.

```sh
npm install                 # only for the type checker
npx tsc --noEmit
python mechfight-kit/check/check.py -- node bot.ts
node tools/sim.ts --scenario gauntlet --matches 8
docker build --platform linux/amd64 -t magic-royale .
```

Needs Node 22.18+ locally (it runs TypeScript directly). The image runs on Bun, pinned by digest, with no runtime dependencies.

## Credits

The wrapper `mechfight.ts` and the Dockerfile layout come from the organisers' starter kit. A few tactics (holding fire while repairing, aiming at a target's shoulders when its centre is behind cover) were informed by reading the public MIT-licensed Astra bot; no code was copied from it. The idea of recognising walled-off parts of the map and jetting out of them came from reading the public MIT-licensed giddygiddyboy bot (as packaged by Team Kanyaraasi); the implementation here is our own.
