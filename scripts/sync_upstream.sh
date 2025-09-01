#!/usr/bin/env bash
set -euo pipefail

# Create a local branch synced to an upstream release tag.
#
# Required:
#   --release <ver>     Release version (e.g., 0.5.0 or v0.5.0)
#
# Defaults:
#   --upstream upstream
#   --branch (auto: upstream-sync/<resolved-tag>)
#
# Options:
#   --upstream <name>   Upstream remote name (default: upstream)
#   --branch <name>     Local branch name to create/reset
#   --force             Overwrite branch if it already exists
#   --push              Push branch to origin and set upstream
#   --allow-dirty       Allow running with uncommitted changes
#   -h|--help           Show help
#
# Behavior:
#   - Always merges the local 'main' branch into the created/reset sync branch.
#   - Requires a local branch named 'main' to exist (no fallback).

die() { echo "Error: $*" >&2; exit 1; }
info() { echo "[sync] $*"; }

SHOW_HELP=false
RELEASE=""
UPSTREAM="upstream"
BRANCH=""
FORCE=false
PUSH=false
ALLOW_DIRTY=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --release) RELEASE="${2:?}"; shift 2 ;;
    --upstream) UPSTREAM="${2:?}"; shift 2 ;;
    --branch) BRANCH="${2:?}"; shift 2 ;;
    --force) FORCE=true; shift ;;
    --push) PUSH=true; shift ;;
    --allow-dirty) ALLOW_DIRTY=true; shift ;;
    -h|--help) SHOW_HELP=true; shift ;;
    *) die "Unknown argument: $1" ;;
  esac
done

if $SHOW_HELP; then
  sed -n '2,60p' "$0"
  exit 0
fi

# Require --release to be provided
if [[ -z "${RELEASE}" ]]; then
  die "Missing required --release <ver>. Use --help for usage."
fi

# Preconditions
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "Not inside a git repository."
git remote get-url "$UPSTREAM" >/dev/null 2>&1 || die "Remote '$UPSTREAM' not found. Add it with: git remote add $UPSTREAM <url>"

if ! $ALLOW_DIRTY; then
  if ! git diff-index --quiet HEAD --; then
    die "Working tree has uncommitted changes. Commit/stash or use --allow-dirty."
  fi
fi

# Fetch upstream tags
info "Fetching tags from '$UPSTREAM'..."
git fetch --tags --prune "$UPSTREAM"

# Resolve tag (supports '0.5.0' or 'v0.5.0')
base="${RELEASE#v}"
candidates=("v$base" "$base")

FOUND_TAG=""
for t in "${candidates[@]}"; do
  if git rev-parse -q --verify "refs/tags/$t^{commit}" >/dev/null; then
    FOUND_TAG="$t"
    break
  fi
done
[[ -n "$FOUND_TAG" ]] || die "Could not find tag for release '$RELEASE'. Tried: ${candidates[*]}"

info "Resolved release '$RELEASE' to tag '$FOUND_TAG'."

# Determine branch name
if [[ -z "${BRANCH}" ]]; then
  BRANCH="upstream-sync/${FOUND_TAG}"
fi

# Create/reset branch
if git rev-parse -q --verify "$BRANCH" >/dev/null; then
  if $FORCE; then
    info "Branch '$BRANCH' exists; resetting to '$FOUND_TAG'."
    git checkout -B "$BRANCH" "$FOUND_TAG"
  else
    die "Branch '$BRANCH' already exists. Use --force to reset it."
  fi
else
  info "Creating branch '$BRANCH' at '$FOUND_TAG'."
  git checkout -b "$BRANCH" "$FOUND_TAG"
fi

# Merge local 'main' into the sync branch (must exist locally)
if ! git show-ref --verify --quiet refs/heads/main; then
  die "Local branch 'main' not found. Create it locally and try again."
fi

info "Merging local 'main' into '$BRANCH'..."
git merge --no-ff --no-edit main || die "Merge failed. Resolve conflicts and try again."

# Optionally push to origin
if $PUSH; then
  git remote get-url origin >/dev/null 2>&1 || die "Remote 'origin' not found; cannot push. Add it or omit --push."
  info "Pushing '$BRANCH' to 'origin' and setting upstream."
  git push -u origin "$BRANCH"
fi

info "Done. Branch '$BRANCH' now points at tag '$FOUND_TAG'."
