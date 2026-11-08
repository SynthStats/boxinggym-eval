Output assets/demo.gif

Set Shell "bash"
Set FontFamily "JetBrains Mono"
Set FontSize 20
Set Width 1200
Set Height 675
Set Padding 24
Set Framerate 30
Set WaitTimeout 180s
Set WaitPattern /^\$$/
Set Theme { "name": "SynthStatsDark", "background": "#0D0F17", "foreground": "#E9EBF5", "cursor": "#ABA6F2", "selection": "#242839", "black": "#151925", "red": "#D41950", "green": "#0096D0", "yellow": "#ABA6F2", "blue": "#0072B2", "magenta": "#5C56A5", "cyan": "#0096D0", "white": "#E9EBF5", "brightBlack": "#A0A6BC", "brightRed": "#F03268", "brightGreen": "#0096D0", "brightYellow": "#ABA6F2", "brightBlue": "#0096D0", "brightMagenta": "#ABA6F2", "brightCyan": "#0096D0", "brightWhite": "#FBFBFD" }

Hide
Type "export PS1='$ ' && clear"
Enter
Wait
Type "git lfs pull 2>/dev/null; mkdir -p .boxing-gym-cache && cp scripts/streamlit_app/demo_data/canonical_runs.parquet .boxing-gym-cache/runs.parquet"
Enter
Wait
Type "uv run --quiet box --help >/dev/null 2>&1"
Enter
Wait
Type "uv run --quiet box --help >/dev/null 2>&1 && export UV_NO_SYNC=1"
Enter
Wait
Type "clear"
Enter
Wait
Show
Type "uv run box query --list"
Sleep 400ms
Enter
Wait+Screen /leaderboard/
Sleep 2s

Type "uv run box query leaderboard"
Sleep 400ms
Enter
Wait+Screen /Model|model|rank|Rank/
Sleep 6s
