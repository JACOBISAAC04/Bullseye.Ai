# Bullseye — Phase 6: Offline English Voice Coaching

This optional **Windows-only** stage converts your existing Phase 6 player-feedback
report into English speech. It uses the built-in **Windows System.Speech** voice
engine (SAPI); there is no cloud API, subscription, or pip dependency.

**Current capability:** spoken, measurement-backed review observations, short
review questions, and explanations when a shot lacks sufficient evidence.

**Not yet implemented:** validated form-mistake diagnoses, intended hand/body
placement, ideal ball trajectory overlays, or made/missed-shot classification.
This module explicitly does not narrate any of these as if they were known.

## Install the four separate files

From PowerShell in `D:\Bullseye\backend`:

```powershell
Copy-Item "$HOME\Downloads\english_voice.py" "app\coaching\" -Force
Copy-Item "$HOME\Downloads\english_voice_sapi.ps1" "app\coaching\" -Force
Copy-Item "$HOME\Downloads\test_english_voice.py" "tests\" -Force
Copy-Item "$HOME\Downloads\README_ENGLISH_VOICE.md" "." -Force
```

No existing `app/api/video.py`, YOLO, MediaPipe, ball tracking, or Phase 6
modules need to be modified.

## Automated tests

```powershell
python -m unittest discover -s tests -p "test_english_voice.py" -v
```

Expected: `Ran 35 tests / OK`.

The tests cover narration and mock the Windows speech invocation, so you must
still run the real WAV-generation commands on Windows.

## Option 1: Check narration text without generating WAV

You may use a Phase 6.1/6.2 report even if you have not run Phase 6.3:

```powershell
python -m app.coaching.english_voice `
    --input data/analyzed/phase6_player_feedback_video3.json `
    --output-dir data/analyzed/phase6_voice_preview_video3 `
    --script-only
```

Open `data\analyzed\phase6_voice_preview_video3\voice_manifest.json`.
The script-only command writes no audio. The manifest contains the English
narration for each shot and the saved replay time ranges.

## Option 2: Produce real offline English WAV files

**Recommended input:** the enriched Phase 6.3 report, if it exists:

```powershell
python -m app.coaching.english_voice `
    --input data/analyzed/phase6_evidence_feedback_video3.json `
    --output-dir data/analyzed/phase6_voice_video3
```

If you haven't run Phase 6.3, replace the input with:

`data/analyzed/phase6_player_feedback_video3.json`

Generated files:

- `phase6_voice_video3\shot_001_english.wav` — a voice explanation of insufficient evidence
- `phase6_voice_video3\shot_002_english.wav` — six observed review topics and questions
- `phase6_voice_video3\voice_manifest.json` — text, replay timing, and source URLs

Listen in Windows:

```powershell
Invoke-Item "data\analyzed\phase6_voice_video3\shot_002_english.wav"
```

For Video 8:

```powershell
python -m app.coaching.english_voice `
    --input data/analyzed/phase6_player_feedback_video8.json `
    --output-dir data/analyzed/phase6_voice_video8
```

Video 8 has zero supported review cards; the voice explains the evidence limit,
not a made-up shooting error.

The `--rate` option supports values from -10 (slower) to +10 (faster):

```powershell
python -m app.coaching.english_voice `
    --input data/analyzed/phase6_player_feedback_video3.json `
    --output-dir data/analyzed/phase6_voice_video3_slow `
    --rate -2
```

**Use a fresh output directory for each run.** Existing WAV and
`voice_manifest.json` files are not overwritten.

## Technical implementation

- `english_voice.py`: validates and prepares a safe narration JSON, then
  invokes the bundled `.ps1` helper once per shot.
- `english_voice_sapi.ps1`: selects an **installed English-language** Windows
  System.Speech voice and writes uncompressed WAV.
- No microphone, speech recognition, Windows administrator privileges,
  third-party TTS libraries, or internet connection are required.
- This stage is independent of FastAPI. It does not alter `video.py`.
- Replay times and coaching sources are kept in the JSON manifest; the voice
  does not fabricate any new timestamps.
- Individual shots are emitted separately so they can later be attached to
  the player's replay controls.
- Windows speech synthesis is platform-dependent. The Python tests can check
  the interface on other operating systems but cannot prove a Windows voice is
  installed. Verify actual WAV playback on your Windows PC.

### If the voice engine fails

If you see `No English Windows SAPI voice is installed`, check that an English
text-to-speech voice is installed in Windows language and speech settings.

If `System.Speech` cannot be loaded, use Windows PowerShell (`powershell.exe`),
not Linux/macOS PowerShell, and confirm you're running on Windows.

Send back the unit-test summary, each report's console output, and whether the
WAV files play. Don't commit the module until real Windows playback is checked.
