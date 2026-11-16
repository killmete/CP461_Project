param([string]$Deck, [string]$OutDir)
# Exports every slide of $Deck to PNG with the installed PowerPoint (visual QA).
New-Item -ItemType Directory -Force $OutDir | Out-Null
Get-ChildItem $OutDir -Filter *.png | Remove-Item -Force -Confirm:$false
$app = New-Object -ComObject PowerPoint.Application
$p = $app.Presentations.Open((Resolve-Path $Deck).Path, $true, $false, $false)
$i = 0
foreach ($s in $p.Slides) { $i++; $s.Export((Join-Path (Resolve-Path $OutDir).Path ("slide-{0:D2}.png" -f $i)), "PNG", 1600, 900) }
$p.Close(); $app.Quit()
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
"exported $i slides"
