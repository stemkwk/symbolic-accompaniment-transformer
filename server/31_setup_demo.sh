#!/usr/bin/env bash
#
# 31_setup_demo.sh — install demo dependencies (soundfont, FluidSynth, DSP).
#
# Run once after training completes, before launching app.py.
# Idempotent — safe to re-run.
#
# What it installs:
#   System : fluidsynth          — MIDI→WAV synthesis engine
#            fluid-soundfont-gm  — FluidR3_GM.sf2 (~140 MB, auto-detected by audio.py)
#   Python : jam_transformer[render,audio,demo]
#              render → pyfluidsynth, pedalboard, soundfile
#              audio  → basic-pitch, sounddevice, noisereduce
#              demo   → gradio, librosa
#
# After this script, launch the web demo with:
#   python app.py --checkpoint <path> --share

source "$(dirname "$0")/_common.sh"
require_project_layout

# ── System packages ───────────────────────────────────────────────────────────
log_section "System packages (apt)"

_apt_install() {
    local pkg="$1"
    if dpkg -s "${pkg}" &>/dev/null; then
        log_step "${pkg} : already installed"
    else
        log_step "Installing ${pkg}…"
        apt-get install -y "${pkg}" || log_fail "apt-get install ${pkg} failed."
    fi
}

apt-get update -qq
_apt_install fluidsynth
_apt_install fluid-soundfont-gm

# Verify soundfont landed where audio.py auto-detects it
SF2="/usr/share/sounds/sf2/FluidR3_GM.sf2"
if [[ -f "${SF2}" ]]; then
    log_step "Soundfont : ${SF2}  ($(du -h "${SF2}" | cut -f1))"
else
    log_warn "Soundfont not found at ${SF2}."
    log_warn "Set inference.soundfont in configs/config.yaml manually."
fi

# ── Python packages ───────────────────────────────────────────────────────────
log_section "Python packages"
log_step "pip install -e '.[render,audio,demo]'"
python -m pip install -e ".[render,audio,demo]"

# ── Quick smoke test ──────────────────────────────────────────────────────────
log_section "Smoke test"
python - <<'PY'
errors = []

try:
    import fluidsynth
except ImportError:
    errors.append("pyfluidsynth")
try:
    from pedalboard import Pedalboard, Reverb
except ImportError:
    errors.append("pedalboard")
try:
    import gradio
except ImportError:
    errors.append("gradio")
try:
    from basic_pitch.inference import predict
except ImportError:
    errors.append("basic-pitch")

if errors:
    print(f"MISSING: {', '.join(errors)}")
    raise SystemExit(1)
print("All demo dependencies importable.")
PY

log_section "✔ Demo setup complete"
echo ""
echo "Launch the web demo:"
echo "  python app.py --checkpoint <path/to/best.ckpt> --share"
echo ""
echo "Or CLI:"
echo "  python scripts/inference.py --checkpoint <ckpt> --melody_midi <mid>"
