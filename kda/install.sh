#!/usr/bin/env bash
# Install KDA into a user repository.
#
#   kda/install.sh [--link] [--force] <user_repo>
#
# Default (copy): kda/kda-skills/* and kda/general-infra-skills/* -> <repo>/.agents/skills/<name>,
# kda/agents/*.md -> <repo>/.agents/agents/, kda/agents/*.toml -> <repo>/.codex/agents/. For Claude
# Code, <repo>/.claude/skills/<name> and <repo>/.claude/agents/<name>.md are relative symlinks into
# .agents/ (always created; Cursor and pi read .agents/skills/ directly), and <repo>/.cursor/agents/
# <name>.md for the Cursor CLI's subagents (it reads .cursor/agents/ and .claude/agents/, not .agents/agents/). The copy is self-contained: the
# user can commit it or ignore it, and KDA can move or change without touching the repo.
#
# --link reproduces every target as a relative symlink into this checkout instead (for developing
# KDA: `git pull` here updates every linked repo).
#
# Requires Linux, Bash 4+, GNU realpath and Python 3.
# <repo>/.kda/manifest.json records {kda_commit, mode, installed}. A re-run removes every
# manifest path and installs again; a destination that exists and is not in the manifest is skipped
# with a message unless --force. .gitignore is never edited.
set -euo pipefail

usage() { echo "usage: $0 [--link] [--force] <user_repo>" >&2; exit 2; }

mode=copy
force=0
repo=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --link) mode=link; shift ;;
        --force) force=1; shift ;;
        -h|--help) usage ;;
        -*) echo "unknown flag $1" >&2; usage ;;
        *) [[ -z "$repo" ]] || usage; repo=$1; shift ;;
    esac
done
[[ -n "$repo" && -d "$repo" ]] || usage
command -v python3 >/dev/null || { echo "Python 3 is required to read and write the KDA manifest" >&2; exit 1; }

repo=$(cd "$repo" && pwd -P)
kda=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
manifest="$repo/.kda/manifest.json"

# Only installed leaves are removable; a symlinked parent must stay inside the repo.
validate_destination() {
    local rel=$1 parent
    if [[ ! $rel =~ ^\.(agents|claude)/skills/[A-Za-z0-9][A-Za-z0-9._-]*$ &&
          ! $rel =~ ^\.(agents|claude|cursor)/agents/[A-Za-z0-9][A-Za-z0-9._-]*\.md$ &&
          ! $rel =~ ^\.codex/agents/[A-Za-z0-9][A-Za-z0-9._-]*\.toml$ ]]; then
        echo "invalid install destination: $rel" >&2; exit 1
    fi
    parent=$(realpath -m -- "$(dirname "$repo/$rel")")
    [[ $parent == "$repo/"* ]] || { echo "install parent escapes repository: $rel" >&2; exit 1; }
}

# Do not read or overwrite a manifest through an external directory or a file symlink.
[[ $(realpath -m -- "$repo/.kda") == "$repo/"* && ! -L "$manifest" ]] || {
    echo "unsafe manifest location: $manifest" >&2; exit 1;
}

# ---------------------------------------------------------------- previous install
# Paths in the manifest are ours to remove and re-create; anything else is the user's.
declare -A owned=()
if [[ -f "$manifest" ]]; then
    previous=$(python3 - "$manifest" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    data = json.load(stream)
paths = data.get("installed") if isinstance(data, dict) else None
if not isinstance(paths, list) or not all(isinstance(p, str) and p and not any(c in p for c in "\r\n\0") for p in paths):
    raise SystemExit("invalid installed paths in KDA manifest")
print("\n".join(paths))
PY
    )
    while IFS= read -r p; do
        [[ -n "$p" ]] || continue
        validate_destination "$p"
        owned["$p"]=1
    done <<< "$previous"
    for p in "${!owned[@]}"; do
        rm -rf "$repo/$p"   # no trailing slash: a symlink is removed, never followed
    done
    echo "refresh: removed ${#owned[@]} previously installed paths"
fi

installed=()

# Claim <rel> under the repo: false (skip with a message) when a foreign file sits there.
claim() {  # claim <rel>
    local rel=$1 dst="$repo/$1"
    validate_destination "$rel"
    if [[ ( -e "$dst" || -L "$dst" ) && -z "${owned[$rel]:-}" && $force -eq 0 ]]; then
        echo "skip  $rel (exists and was not installed by KDA; --force replaces it)" >&2
        return 1
    fi
    rm -rf "$dst"
    mkdir -p "$(dirname "$dst")"
    return 0
}

install_one() {  # install_one <src> <rel_dst>  (copy or link per --link)
    local src=$1 rel=$2 dst="$repo/$2"
    claim "$rel" || return 0
    if [[ $mode == link ]]; then
        # Relative links survive the pair being mounted elsewhere (e.g. a container that sees
        # both KDA and the repo under a different prefix).
        ln -s "$(realpath --relative-to="$(dirname "$dst")" "$src")" "$dst"
    else
        cp -R "$src" "$dst"
    fi
    installed+=("$rel")
    echo "$mode  $rel"
}

alias_link() {  # alias_link <rel_target_in_repo> <rel_dst>  (always a relative symlink)
    local target="$repo/$1" rel=$2 dst="$repo/$2"
    [[ -e "$target" || -L "$target" ]] || return 0   # the target was skipped
    claim "$rel" || return 0
    ln -s "$(realpath --relative-to="$(dirname "$dst")" --no-symlinks "$target")" "$dst"
    installed+=("$rel")
}

# ---------------------------------------------------------------- install
for src in "$kda"/kda-skills/*/ "$kda"/general-infra-skills/*/; do
    [[ -d "$src" ]] || continue
    name=$(basename "${src%/}")
    install_one "${src%/}" ".agents/skills/$name"
    alias_link ".agents/skills/$name" ".claude/skills/$name"
done
for src in "$kda"/agents/*.md; do
    [[ -f "$src" ]] || continue
    name=$(basename "$src")
    [[ $name != README.md ]] || continue
    install_one "$src" ".agents/agents/$name"
    alias_link ".agents/agents/$name" ".claude/agents/$name"
    alias_link ".agents/agents/$name" ".cursor/agents/$name"
done
for src in "$kda"/agents/*.toml; do
    [[ -f "$src" ]] || continue
    install_one "$src" ".codex/agents/$(basename "$src")"
done

# ---------------------------------------------------------------- manifest
commit=$(git -C "$kda" rev-parse --short HEAD 2>/dev/null || echo unknown)
if [[ "$commit" != unknown && -n "$(git -C "$kda" status --porcelain 2>/dev/null)" ]]; then
    commit="$commit-dirty"
fi
mkdir -p "$(dirname "$manifest")"
python3 - "$commit" "$mode" "${installed[@]}" >"$manifest" <<'PY'
import json
import sys

json.dump({"kda_commit": sys.argv[1], "mode": sys.argv[2], "installed": sys.argv[3:]}, sys.stdout, indent=2)
print()
PY

size=$(du -shL "$repo/.agents/skills" 2>/dev/null | cut -f1 || echo "?")
echo
echo "KDA $commit installed into $repo ($mode mode, ${#installed[@]} paths, $size under .agents/skills; manifest: .kda/manifest.json)."
echo "Commit .agents/ .claude/ .codex/ .cursor/ .kda/ or add them to .gitignore, as you prefer; this script never edits .gitignore."
echo "Start with the 'kda-kernels' skill: name the function or module to fuse and the training config."
echo "The generated kernels will live in $repo/kda_kernels/ and do not import KDA at runtime."
