#!/usr/bin/env bash
# Install everything the validator needs: the Python environment, OFIQ (built once, next to this repo)
# and the .env that points at it. Safe to run again: finished steps are skipped.
#
#   ./install.sh                 Python environment + OFIQ
#   ./install.sh --siglip        also the SigLIP second opinions (PyTorch CPU; 3.3 GB model on first use)
#   ./install.sh --without-ofiq  Python environment only (the OFIQ checks then report "skipped")
#
# Environment:
#   OFIQ_PROJECT_DIR   where OFIQ is, or gets cloned to (default: ../OFIQ-Project, next to this repo)
#   OFIQ_VERSION       the OFIQ tag to clone (default: v1.2.1)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFIQ_PROJECT_DIR="${OFIQ_PROJECT_DIR:-$(dirname "$REPO")/OFIQ-Project}"
OFIQ_VERSION="${OFIQ_VERSION:-v1.2.1}"
CONAN_VERSION=2.18.1
WITH_OFIQ=1
WITH_SIGLIP=0

for arg in "$@"; do
  case "$arg" in
    --without-ofiq) WITH_OFIQ=0 ;;
    --siglip) WITH_SIGLIP=1 ;;
    -h|--help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "install.sh: unknown option $arg (try --help)" >&2; exit 2 ;;
  esac
done

say() { printf '\n==> %s\n' "$*"; }
die() { printf 'install.sh: %s\n' "$*" >&2; exit 1; }

# write_env KEY VALUE: set KEY in the repo's .env, replacing the line if it is there
write_env() {
  local file="$REPO/.env" key="$1" value="$2"
  touch "$file"
  if grep -q "^${key}=" "$file"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "$file"
  else
    printf '%s=%s\n' "$key" "$value" >> "$file"
  fi
  echo "$key=$value (in .env)"
}

# --- 1. Prerequisites. Nothing here installs system packages; the apt line is printed instead.
[ "$(uname -s)" = Linux ] || die "only Linux is supported (OFIQ's build script also knows macOS, but this script doesn't)"
missing=()
need() { command -v "$1" >/dev/null 2>&1 || missing+=("$2"); }
need git git
need uv uv
if [ "$WITH_OFIQ" = 1 ]; then
  need python3 python3
  need g++ build-essential
  need make build-essential
  need unzip unzip
  python3 -c 'import ensurepip' 2>/dev/null || missing+=("python3-venv")
fi
if [ "${#missing[@]}" -gt 0 ]; then
  echo "install.sh: missing: ${missing[*]}" >&2
  apt=$(printf '%s\n' "${missing[@]}" | grep -vx uv | sort -u | tr '\n' ' ')
  [ -n "$apt" ] && echo "  sudo apt-get install $apt" >&2
  printf '%s\n' "${missing[@]}" | grep -qx uv && echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

# --- 2. Python environment. --inexact keeps the OFIQ wheel installed below when the script runs again.
say "Python environment (uv sync)"
extras=(--extra test)
[ "$WITH_SIGLIP" = 1 ] && extras+=(--extra siglip)
(cd "$REPO" && uv sync "${extras[@]}" --inexact)

if [ "$WITH_OFIQ" = 1 ]; then
  # --- 3. OFIQ source
  if [ -d "$OFIQ_PROJECT_DIR/.git" ]; then
    say "OFIQ checkout found at $OFIQ_PROJECT_DIR"
  else
    say "Cloning OFIQ $OFIQ_VERSION into $OFIQ_PROJECT_DIR"
    git clone --branch "$OFIQ_VERSION" --depth 1 https://github.com/BSI-OFIQ/OFIQ-Project.git "$OFIQ_PROJECT_DIR"
  fi

  # --- 4. Build OFIQ and its Python wheel, once. About 15 minutes; downloads OFIQ's models (430 MB) from eu-LISA.
  wheel=$(ls "$OFIQ_PROJECT_DIR"/install_*/Release/ofiq-*.whl 2>/dev/null | head -1 || true)
  if [ -n "$wheel" ]; then
    say "OFIQ wheel found: $wheel"
  else
    say "Building OFIQ (about 15 minutes)"
    venv="$OFIQ_PROJECT_DIR/.conan-venv"
    [ -x "$venv/bin/pip" ] || python3 -m venv "$venv"
    # conan and a recent cmake for the build; the venv also provides the "python" the wheel step calls
    "$venv/bin/pip" install --quiet "conan==$CONAN_VERSION" "cmake>=3.26"
    # OFIQ's profile would install missing system packages with sudo apt; make it report them instead
    profile="$OFIQ_PROJECT_DIR/conan/conan_profile_release_linux.txt"
    sed -i 's/^tools.system.package_manager:mode=.*/tools.system.package_manager:mode=check/;
            s/^tools.system.package_manager:sudo=.*/tools.system.package_manager:sudo=False/' "$profile"
    (cd "$OFIQ_PROJECT_DIR/scripts" &&
      PATH="$venv/bin:$PATH" CONAN_HOME="$OFIQ_PROJECT_DIR/.conan2" CMAKE_POLICY_VERSION_MINIMUM=3.5 \
      bash build.sh --build-python)
    wheel=$(ls "$OFIQ_PROJECT_DIR"/install_*/Release/ofiq-*.whl 2>/dev/null | head -1 || true)
    [ -n "$wheel" ] || die "the OFIQ build finished without a wheel under $OFIQ_PROJECT_DIR/install_*/Release/"
  fi

  # --- 5. The wheel goes into this repo's environment (a plain "uv sync" removes it again; run this script instead)
  say "Installing the OFIQ wheel"
  (cd "$REPO" && uv pip install --quiet --reinstall "$wheel")

  # --- 6. Tell the validator where OFIQ's config and models are. .env is gitignored and read from the repo root.
  say "Settings"
  write_env OFIQ_CONFIG_DIR "$OFIQ_PROJECT_DIR/data"
fi
if [ "$WITH_SIGLIP" = 1 ]; then
  [ "$WITH_OFIQ" = 1 ] || say "Settings"
  write_env SIGLIP_ENABLED true
fi

# --- 7. Run the validator once on a sample photo
say "Checking the installation"
(cd "$REPO" && GLOG_minloglevel=2 uv run python - <<'PY'
import sys
import cv2
from passphoto_validation.core import PassportPhotoValidator
from passphoto_validation.core.config import ValidatorConfig

config = ValidatorConfig(autocrop=False)
validator = PassportPhotoValidator(config)
if validator.ofiq is None:
    print(f'OFIQ: not loaded ({validator.ofiq_unavailable})')
    if config.ofiq_config_dir:
        sys.exit(1)
else:
    print(f"OFIQ {'.'.join(map(str, validator.ofiq.version))} from {config.ofiq_config_dir}")
report = validator(cv2.imread('tests/test-images/9.jpg'), crop=False)
skipped = [f'{check.name} ({check.reasons[0]})' for check in report.checks if check.status == 'skipped']
crashed = [check.name for check in report.checks if check.status == 'error']
print(f'{len(report.checks)} checks ran on tests/test-images/9.jpg')
for line in skipped:
    print(f'  skipped: {line}')
if crashed:
    sys.exit(f'  crashed: {", ".join(crashed)}')
print('Done.')
PY
)
