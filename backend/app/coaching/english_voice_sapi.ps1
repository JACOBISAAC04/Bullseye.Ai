param(
    [Parameter(Mandatory=$true)][string]$InputTextFile,
    [Parameter(Mandatory=$true)][string]$OutputWav,
    [ValidateRange(-10,10)][int]$Rate = -1
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech

if (-not [System.IO.File]::Exists($InputTextFile)) {
    throw "Narration text file was not found."
}
if ([System.IO.File]::Exists($OutputWav)) {
    throw "Refusing to overwrite an existing WAV."
}

$text = [System.IO.File]::ReadAllText(
    $InputTextFile,
    [System.Text.Encoding]::UTF8
)
if ([string]::IsNullOrWhiteSpace($text)) {
    throw "Narration text is empty."
}

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $englishVoice = $synth.GetInstalledVoices() |
        Where-Object {
            $_.Enabled -and
            $_.VoiceInfo.Culture.Name -match '^en(-|$)'
        } |
        Select-Object -First 1

    if ($null -eq $englishVoice) {
        throw "No English Windows SAPI voice is installed. Install an English system voice and retry."
    }

    $synth.SelectVoice($englishVoice.VoiceInfo.Name)
    $synth.Rate = $Rate
    $synth.SetOutputToWaveFile($OutputWav)
    $synth.Speak($text)
}
finally {
    $synth.Dispose()
}
