#!/usr/bin/env bash
# Tidy the project folder. Nothing changes until you type y, and nothing the lab needs to run is touched.
#   - moves files lying next to the project into it: the KU Leuven archive -> data/raw/kuleuven_bev/,
#     the course brief and front-page template -> docs/course_brief/, the deck and speech notes -> docs/presentation/
#   - deletes copies next to the project that are byte-identical to a file already inside it
#   - deletes superseded figures, the extracted NASA cache (re-created by --rebuild) and tool caches
# Safe to run again at any time.
#
#   bash scripts/tidy.sh
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARENT="$(dirname "$ROOT")"
NAME="$(basename "$ROOT")"
cd "$ROOT"
# Started from the app-menu launcher: keep the window open so the result can be read.
if [ "${G14_PAUSE:-0}" = 1 ]; then trap 'read -r -p "Press Enter to close this window." _ || true' EXIT; fi
srcs=(); dsts=(); dels=(); why=()

plan_delete() {  # plan_delete <path> <reason>
  if [ -e "$1" ]; then dels+=("$1"); why+=("$2"); fi
}
plan_move() {    # plan_move <file next to the project> <destination folder inside the project>
  local f="$1" dest="$2" target
  [ -f "$f" ] || return 0
  target="$dest/$(basename "$f")"
  if [ -e "$target" ]; then
    # Already inside: drop the outside copy only if it is identical; otherwise leave both alone.
    if cmp -s "$f" "$target"; then plan_delete "$f" "identical copy is in $NAME/$dest/"; fi
    return 0
  fi
  srcs+=("$f"); dsts+=("$dest")
}

# 1. Files next to the project folder.
plan_move "$PARENT/BEV energy dynamic data_V2.zip" data/raw/kuleuven_bev
plan_move "$PARENT/LLT_Instruction.pdf" docs/course_brief
plan_move "$PARENT/SubTopics.txt" docs/course_brief
plan_move "$PARENT/EVT_LLT-1_Front Page Template.docx" docs/course_brief
plan_move "$PARENT/Group14_Technical_Report.pdf" docs
plan_move "$PARENT/Group14_Technical_Report.docx" docs
for f in "$PARENT"/"Battery Digital Twin Virtual Lab"*.pptx "$PARENT"/"Battery Digital Twin Virtual Lab"*.pdf "$PARENT/Speech_Notes.pdf"; do
  plan_move "$f" docs/presentation
done

# 2. Superseded files and caches inside the project.
plan_delete "$ROOT/outputs/figures/screenshots" "old screenshots, replaced by outputs/figures/demo/"
plan_delete "$ROOT/docs/figures/architecture.png" "old diagram, replaced by docs/figures/system_architecture.png"
plan_delete "$ROOT/data/raw/nasa_pcoe/extracted" "unzipped copy of the NASA archive, re-created by run.sh --rebuild"
if [ -d outputs/reports ] && [ -z "$(ls -A outputs/reports)" ]; then plan_delete "$ROOT/outputs/reports" "empty, unused folder"; fi
while IFS= read -r -d '' d; do plan_delete "$d" "cache, re-created automatically"; done < <(
  find "$ROOT" -path "$ROOT/.venv" -prune -o -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) -print0)

if [ ${#srcs[@]} -eq 0 ] && [ ${#dels[@]} -eq 0 ]; then
  echo "The project folder is already tidy."
  exit 0
fi

echo "Planned changes in $PARENT:"
if [ ${#srcs[@]} -gt 0 ]; then
  for i in "${!srcs[@]}"; do echo "  move    ${srcs[$i]#"$PARENT"/}  ->  $NAME/${dsts[$i]}/"; done
fi
if [ ${#dels[@]} -gt 0 ]; then
  for i in "${!dels[@]}"; do echo "  delete  ${dels[$i]#"$PARENT"/}   (${why[$i]})"; done
fi
read -r -p "Apply these changes? [y/N] " ok || ok=""
case "$ok" in
  [yY]|[yY][eE][sS]) ;;
  *) echo "Nothing was changed."; exit 0 ;;
esac

if [ ${#srcs[@]} -gt 0 ]; then
  for i in "${!srcs[@]}"; do mkdir -p "${dsts[$i]}" && mv -n -- "${srcs[$i]}" "${dsts[$i]}/"; done
fi
if [ ${#dels[@]} -gt 0 ]; then
  for d in "${dels[@]}"; do rm -rf -- "$d"; done
fi
chmod +x run.sh scripts/*.sh 2>/dev/null || true
echo "Done. Start the lab with: bash run.sh"
