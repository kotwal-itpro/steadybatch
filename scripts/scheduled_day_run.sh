#!/bin/zsh
# Runs the 1,000-record benchmark on OpenAI and Gemini for day 2 or day 3, then
# publishes the results to the repo. Scheduled by launchd; removes its own schedule after day 3.
set -u
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$HOME/.local/bin"
cd "$HOME/codes/steadybatch" || exit 1
source <(grep -E '^[[:space:]]*export[[:space:]]+(OPENAI_API_KEY|GEMINI_API_KEY)=' "$HOME/.zshrc")

D=$(date +%Y-%m-%d)
case "$D" in
  2026-10-08) R=r2 ;;
  2026-10-09) R=r3 ;;
  *) echo "$(date) not a scheduled day ($D); nothing to do"; exit 0 ;;
esac
LOG="runs/scheduled-$D.log"
mkdir -p runs
exec >>"$LOG" 2>&1
echo "== $(date) starting day run $R"

run() {  # provider model
  local out="runs/$D-$1-$2-1000-$R"
  [ -f "$out/summary.json" ] && { echo "$out already done"; return 0; }
  .venv/bin/steadybatch-bench run --provider "$1" --model "$2" \
    --data examples/support_tickets.jsonl --schema examples/support_ticket.schema.json \
    --instructions examples/support_ticket.instructions.txt \
    --limit 1000 --max-tokens 300 --poll-every 60 --out "$out"
}
run openai gpt-4.1-mini &
run gemini gemini-3.5-flash-lite &
wait

for out in runs/$D-*-1000-$R; do
  [ -f "$out/summary.json" ] || continue
  dest="results/$(basename "$out")"
  mkdir -p "$dest" && cp "$out/summary.json" "$out/results.jsonl" "$dest/"
done
.venv/bin/steadybatch-bench report results/2026-10-* --out results/latest \
  --prices results/prices-2026-10-07.json || true
for p in openai-gpt-4.1-mini gemini-3.5-flash-lite; do
  [ -d "results/$D-$p-1000-$R" ] && .venv/bin/steadybatch-bench compare \
    "results/2026-10-07-$p-1000-r1" "results/$D-$p-1000-$R" > "results/$D-$p-1000-$R/compare-vs-r1.json" || true
done

git pull -q --rebase origin main
git add results && git commit -q -m "Day ${R#r} 1,000-record runs ($D)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin main
echo "== $(date) finished and pushed"

if [ "$R" = r3 ]; then
  launchctl bootout "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.kotwal.steadybatch-dayruns.plist" 2>/dev/null
  rm -f "$HOME/Library/LaunchAgents/com.kotwal.steadybatch-dayruns.plist"
  echo "schedule removed after day 3"
fi
