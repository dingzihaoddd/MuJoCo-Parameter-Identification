# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MuJoCo dynamics parameter identification via "simulation-to-simulation": PD position servo generates trajectory data, then L-BFGS-B optimizes damping/frictionloss to minimize 1-step prediction error. The plant is a single-joint pendulum.

- Identified parameters: `damping`, `frictionloss`
- Model: `models/pendulum.xml` (RK4 integrator, joint friction/damping configured, contacts disabled)

## Commands

```bash
python optimize_pd.py   # Generate PD-servo trajectories → L-BFGS-B + Nelder-Mead → 1-step loss
```

Results saved to `results/<timestamp>/` (`.txt`, `.json`, `.pdf` plot).

## Architecture

```
models/pendulum.xml     # MuJoCo model: single hinge joint + motor actuator
src/simulator.py        # MuJoCo wrapper: load, reset, step, run, get/set params
optimize_pd.py          # Full pipeline: config → trajectory generation → optimization → output
```

`optimize_pd.py` is self-contained — all signal generators, PD control logic, loss function, and optimization are inline. It only imports `Simulator` from `src/simulator.py`.

## Key technical details

**1-step prediction loss**: Instead of simulating full trajectories, the loss resets the simulator to the true state at each timestep, applies PD torque, steps once, and measures single-step velocity prediction error. This avoids chaotic divergence and produces a smooth loss landscape.

**damping/frictionloss coupling**: Both resist motion. They form a "coupling valley" in the loss landscape. The multi-trajectory strategy (different frequencies, amplitudes, initial angles) provides diverse velocity profiles to disambiguate them. Low-speed, small-amplitude trajectories (freq=0.1-0.15 Hz, amp=0.03-0.05 rad) are especially helpful for identifying frictionloss in the sign-switching region.

**Multi-round eps schedule + Nelder-Mead refinement**: L-BFGS-B runs with decreasing finite-difference step sizes (`eps` = 0.02 → 0.0002), then Nelder-Mead finishes the narrow coupling valley where L-BFGS-B gradients vanish.

**PD controller**: Position servo with ideal velocity = 0. Torque = `Kp*(q_ref - q) - Kd*qd`. Reference signals are sine waves and logarithmic chirps, with initial phase set so q_ref[0] == q0, eliminating initial tracking transients.
