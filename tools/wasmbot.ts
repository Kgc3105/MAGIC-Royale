/**
 * Runs a bot built as a WebAssembly module (exports alloc/handle, one protocol line per call) inside
 * the simulator, so another controller can be measured on the same matches as ours. Local testing
 * only; no such module is part of this repository or of any submitted image.
 */
import { readFileSync } from "node:fs"
import { WASI } from "node:wasi"
import type { BotAction, Hello, Observation } from "../mechfight.ts"

// eslint-disable-next-line @typescript-eslint/no-explicit-any
declare const WebAssembly: any
let compiled: any = null
let compiledPath = ""
const enc = new TextEncoder()
const dec = new TextDecoder()

export class WasmBot {
  mode = "go"
  powerEnded = ""
  // the simulator's trace pokes at these on our own bot; harmless stand-ins here
  pickups = new Map<number, never>()
  ignore = new Map<number, number>()
  nav = { areaAt: () => 0 }
  errors = 0
  private exp: { memory: { buffer: ArrayBuffer }; alloc: (n: number) => number; handle: (n: number) => bigint }

  constructor(wasmPath: string, arenaDir: string, hello: Hello, env: Record<string, string>) {
    if (!compiled || compiledPath !== wasmPath) {
      compiled = new WebAssembly.Module(readFileSync(wasmPath))
      compiledPath = wasmPath
    }
    const wasi = new WASI({ version: "preview1", args: ["bot"], env, preopens: { "/arena": arenaDir } })
    const inst = new WebAssembly.Instance(compiled, wasi.getImportObject())
    wasi.initialize(inst)
    this.exp = inst.exports as never
    const r = this.send(JSON.stringify({ ...hello, type: "hello", map: { ...hello.map, dir: "/arena" } }))
    if (!r || !r.includes("ready")) throw new Error(`wasm bot did not start: ${r}`)
  }

  private send(line: string): string | null {
    const bytes = enc.encode(line)
    const ptr = this.exp.alloc(bytes.length)
    new Uint8Array(this.exp.memory.buffer, ptr, bytes.length).set(bytes)
    const r = this.exp.handle(bytes.length)
    if (r === 0n) return null
    const p = Number(r >> 32n)
    const n = Number(r & 0xffffffffn)
    return dec.decode(new Uint8Array(this.exp.memory.buffer, p, n))
  }

  act(obs: Observation): BotAction {
    const r = this.send(JSON.stringify(obs))
    if (!r) return {}
    const a = JSON.parse(r)
    if (a.error) {
      this.errors++
      return {}
    }
    if (typeof a.mode === "string" && a.mode) this.mode = a.mode.split(":")[0]
    delete a.mode
    return a as BotAction
  }
}
