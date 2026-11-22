//go:build wasip1

// Local testing only (not part of the submitted image): the controller built as a WebAssembly
// module so our TypeScript simulator can call it in-process. The host writes one protocol line
// into the buffer returned by alloc and calls handle; the reply line comes back as (pointer<<32 | length).
package main

import (
	"encoding/json"
	"unsafe"

	"mechbot/mech"
)

var (
	brain  *mech.Brain
	inBuf  []byte
	outBuf []byte
)

//go:wasmexport alloc
func alloc(n int32) int32 {
	if n < 1 {
		n = 1
	}
	inBuf = make([]byte, n)
	return int32(uintptr(unsafe.Pointer(&inBuf[0])))
}

// handle returns 0 when there is nothing to send back.
//
//go:wasmexport handle
func handle(n int32) int64 {
	out := process(inBuf[:n])
	if len(out) == 0 {
		return 0
	}
	outBuf = out
	return int64(uintptr(unsafe.Pointer(&outBuf[0])))<<32 | int64(len(outBuf))
}

func process(raw []byte) []byte {
	var probe struct {
		Type string `json:"type"`
	}
	if json.Unmarshal(raw, &probe) != nil {
		return nil
	}
	switch probe.Type {
	case "hello":
		var h mech.Hello
		if err := json.Unmarshal(raw, &h); err != nil {
			return []byte(`{"error":"hello: ` + err.Error() + `"}`)
		}
		dir := h.Map.Dir
		if dir == "" {
			dir = "/arena"
		}
		w, err := mech.LoadWorld(dir)
		if err != nil {
			return []byte(`{"error":"map"}`)
		}
		brain = mech.NewBrain(&h, w)
		return []byte(`{"type":"ready"}`)
	case "obs":
		if brain == nil {
			return nil
		}
		var o mech.Obs
		if err := json.Unmarshal(raw, &o); err != nil {
			return []byte(`{"error":"obs"}`)
		}
		act := brain.Act(&o)
		out, err := json.Marshal(struct {
			mech.Action
			Mode string `json:"mode"`
		}{act, brain.Mode})
		if err != nil {
			return nil
		}
		return out
	}
	return nil
}

func main() {}
