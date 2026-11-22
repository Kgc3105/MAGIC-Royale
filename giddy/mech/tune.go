package mech

// Our addition to the imported controller: settings overrides for building variants.
// Read once from the environment (set with ENV lines in the Dockerfile, or by the simulator):
//
//	MECH_BASE   "Name=value,Name=value"  applied to the default profile
//	MECH_SQUAD  "Name=value,Name=value"  applied on top of the practice (squad) profile
//	MECH_SALT   integer: shifts the bot's own random choices without changing its logic
//
// Names are fields of Params. Unknown names and bad values are ignored.

import (
	"os"
	"reflect"
	"strconv"
	"strings"
)

type envPair struct{ k, v string }

var (
	envBase  = parsePairs(os.Getenv("MECH_BASE"))
	envSquad = parsePairs(os.Getenv("MECH_SQUAD"))
	envSalt  = parseSalt(os.Getenv("MECH_SALT"))
)

func parseSalt(s string) int64 {
	n, err := strconv.ParseInt(strings.TrimSpace(s), 10, 64)
	if err != nil {
		return 0
	}
	return n
}

func parsePairs(s string) []envPair {
	var out []envPair
	for _, part := range strings.Split(s, ",") {
		kv := strings.SplitN(strings.TrimSpace(part), "=", 2)
		if len(kv) != 2 || kv[0] == "" {
			continue
		}
		out = append(out, envPair{strings.TrimSpace(kv[0]), strings.TrimSpace(kv[1])})
	}
	return out
}

func applyPairs(p *Params, ps []envPair) {
	if len(ps) == 0 {
		return
	}
	v := reflect.ValueOf(p).Elem()
	for _, q := range ps {
		f := v.FieldByName(q.k)
		if !f.IsValid() || !f.CanSet() {
			continue
		}
		switch f.Kind() {
		case reflect.Float64:
			if x, err := strconv.ParseFloat(q.v, 64); err == nil {
				f.SetFloat(x)
			}
		case reflect.Bool:
			f.SetBool(q.v == "1" || q.v == "true")
		case reflect.Int, reflect.Int64:
			if x, err := strconv.ParseInt(q.v, 10, 64); err == nil {
				f.SetInt(x)
			}
		}
	}
}
