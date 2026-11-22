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

**Code we brought in.** The folder `giddy/` is the open-source **giddygiddyboy** Mech Royale controller (Go, MIT licence, copyright (c) 2026 giddygiddyboy), imported from the public copy at https://github.com/blanc-team/try_again at commit `f81f264568868340e6724e6ae81e51f112fcd7f5`. Its licence is kept in `giddy/LICENSE` and is copied into the image. Our changes to it are small and marked "our addition" in the source: `giddy/mech/tune.go` (settings overrides and a salt read from the environment), three hook lines in `giddy/mech/brain.go`, the bot name in `giddy/cmd/bot/main.go`, and `giddy/cmd/wasm` (a test build for our simulator, not part of any image). Some submitted commits build that controller (their `Dockerfile` is `docker/Dockerfile.go`); others build our own TypeScript bot (`docker/Dockerfile.ts`). The commit message of each variant says which.

The wrapper `mechfight.ts` and the Dockerfile layout come from the organisers' starter kit. A few tactics (holding fire while repairing, aiming at a target's shoulders when its centre is behind cover) were informed by reading the public MIT-licensed Astra bot; no code was copied from it. The idea of recognising walled-off parts of the map and jetting out of them came from reading the public MIT-licensed giddygiddyboy bot (as packaged by Team Kanyaraasi); the implementation here is our own. The outnumbered rule (short fighting retreats to cover when several hunters are close, no waiting to heal) and one of the optional upgrade orders also follow that bot's practice settings; again the code is ours.
