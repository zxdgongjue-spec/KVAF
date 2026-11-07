#!/usr/bin/env bash
# Keep these roots and the explicit-root rules aligned with AicpuPackageSearch.
# Sourcing this file has no side effects; callers decide whether ACL headers
# are needed (build) or only the installed package (check/uninstall).
kvaf_find_cann_root() {
  local candidate selected="" canonical_home="" canonical_package=""
  if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    [[ -d "$ASCEND_HOME_PATH" ]] || { echo "invalid ASCEND_HOME_PATH: $ASCEND_HOME_PATH" >&2; return 1; }
    canonical_home="$(cd "$ASCEND_HOME_PATH" && pwd -P)" || return 1
    selected="$canonical_home"
  fi
  if [[ -n "${ASCEND_CANN_PACKAGE_PATH:-}" ]]; then
    [[ -d "$ASCEND_CANN_PACKAGE_PATH" ]] || { echo "invalid ASCEND_CANN_PACKAGE_PATH: $ASCEND_CANN_PACKAGE_PATH" >&2; return 1; }
    canonical_package="$(cd "$ASCEND_CANN_PACKAGE_PATH" && pwd -P)" || return 1
    if [[ -n "$selected" && "$selected" != "$canonical_package" ]]; then
      echo "CANN roots differ: ASCEND_HOME_PATH=$selected ASCEND_CANN_PACKAGE_PATH=$canonical_package" >&2
      return 1
    fi
    selected="$canonical_package"
  fi
  if [[ -z "$selected" ]]; then
    for candidate in /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/cann/aarch64-linux/asc; do
      if [[ -d "$candidate" && -r "$candidate/conf/ascend_package_load.ini" && -s "$candidate/conf/ascend_package_load.ini" ]]; then
        selected="$(cd "$candidate" && pwd -P)" || return 1
        break
      fi
    done
  fi
  if [[ -z "$selected" || ! -r "$selected/conf/ascend_package_load.ini" || ! -s "$selected/conf/ascend_package_load.ini" ]]; then
    echo "CANN loader config not found under ${selected:-default roots}; source CANN set_env.sh" >&2
    return 1
  fi
  printf '%s\n' "$selected"
}
