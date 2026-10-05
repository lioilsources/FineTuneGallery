package main

import "math"

// The κ-gate. A second rater is only worth having on a criterion where it
// agrees with the human beyond what chance alone would produce, and "agrees
// 90% of the time" does not say that: pose_adherence runs at ~90% up, so a
// judge that answers "up" to everything agrees 90% of the time and carries no
// information at all. Cohen's κ subtracts exactly that — the agreement two
// raters would reach by guessing with their own base rates — and is 0 for the
// always-up judge.
//
// The gate decides on the lower bound of κ's 95% interval, not the point
// estimate, for the same reason the eval ranks on the Wilson lower bound: 30
// lucky pairs must not open a gate that 300 pairs would keep shut.

// Gate thresholds. Constants on purpose — a gate that an environment variable
// can lower is not a gate.
const (
	// gateMinPairs is the overlap (human ±1 ∧ judge ±1) below which κ is not
	// computed into a decision at all.
	gateMinPairs = 40
	// gateMinHumanDown guards the minority class. With 3 human "down" labels
	// κ is decided by 3 images; the judge has to show it can find failures.
	gateMinHumanDown = 8
	// gateMinKappa is the conventional "substantial agreement" line.
	gateMinKappa = 0.60
	// gateMinKappaLower keeps a thin sample from passing on a high estimate.
	gateMinKappaLower = 0.40
)

const (
	gateUncalibrated = "uncalibrated"
	gateFail         = "fail"
	gatePass         = "pass"
)

// agreement is the 2×2 table between the human and the judge on one
// criterion, plus what follows from it. The first word of each count is the
// human, the second the judge: UpDown = human up, judge down.
type agreement struct {
	N        int `json:"n"`
	UpUp     int `json:"upUp"`
	UpDown   int `json:"upDown"`
	DownUp   int `json:"downUp"`
	DownDown int `json:"downDown"`

	Po    *float64 `json:"po"`    // observed agreement
	Pe    *float64 `json:"pe"`    // agreement expected by chance
	Kappa *float64 `json:"kappa"` // nil when undefined (pe = 1)
	Lower *float64 `json:"lower"` // 95% interval, clamped to [-1, 1]
	Upper *float64 `json:"upper"`
	// DownAgree is the share of human "down" labels the judge also called
	// down — the minority class, and the one a judge that rubber-stamps
	// misses. nil when the human has no downs.
	DownAgree *float64 `json:"downAgree"`

	Gate string `json:"gate"`
}

// add records one pair. Scores are ±1; anything else is the caller's bug.
func (a *agreement) add(human, judge int) {
	switch {
	case human == 1 && judge == 1:
		a.UpUp++
	case human == 1 && judge == -1:
		a.UpDown++
	case human == -1 && judge == 1:
		a.DownUp++
	case human == -1 && judge == -1:
		a.DownDown++
	default:
		return
	}
	a.N++
}

// humanDown is how many minority-class labels the overlap holds.
func (a *agreement) humanDown() int { return a.DownUp + a.DownDown }

// finish computes κ, its interval and the gate state from the counts.
func (a *agreement) finish() {
	a.Gate = gateUncalibrated
	if a.N == 0 {
		return
	}
	n := float64(a.N)
	po := float64(a.UpUp+a.DownDown) / n
	humanUp := float64(a.UpUp+a.UpDown) / n
	judgeUp := float64(a.UpUp+a.DownUp) / n
	pe := humanUp*judgeUp + (1-humanUp)*(1-judgeUp)
	a.Po, a.Pe = &po, &pe
	if hd := a.humanDown(); hd > 0 {
		d := float64(a.DownDown) / float64(hd)
		a.DownAgree = &d
	}
	if pe >= 1 {
		// Both raters used a single class. κ is 0/0 — not zero, undefined —
		// and no decision can rest on it.
		return
	}
	k := (po - pe) / (1 - pe)
	// Asymptotic standard error (Cohen 1960). Adequate at the sample sizes
	// the gate already insists on; below them the gate does not decide.
	se := math.Sqrt(po*(1-po)/n) / (1 - pe)
	const z = 1.959963984540054
	lo, hi := math.Max(-1, k-z*se), math.Min(1, k+z*se)
	a.Kappa, a.Lower, a.Upper = &k, &lo, &hi
	a.Gate = gateState(a)
}

// gateState is the decision, separated so the thresholds read in one place.
func gateState(a *agreement) string {
	if a.N < gateMinPairs || a.humanDown() < gateMinHumanDown || a.Kappa == nil {
		return gateUncalibrated
	}
	if *a.Kappa < gateMinKappa || *a.Lower < gateMinKappaLower {
		return gateFail
	}
	return gatePass
}
