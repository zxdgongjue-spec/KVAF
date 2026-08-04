#!/usr/bin/env bash

# File-only inventory for libraries carried by registered CANN packages.
# Archive members are streamed to private temporary files and inspected with
# readelf.  Nothing from an archive is loaded or executed.

set -Eeuo pipefail
umask 077
export LC_ALL=C
unset TAR_OPTIONS

usage()
{
  cat <<'EOF'
Usage: audit_a5_registered_aicpu_packages.sh [options]

Options:
  --ascend-home DIR  CANN installation root (default: ASCEND_HOME_PATH)
  --registry FILE    Registry file (default: DIR/conf/ascend_package_load.ini)
  --candidate-soname SONAME=SHA256
                     Fail if another registered archive already supplies this
                     candidate runtime identity (repeatable)
  --replace-archive FILE
                     Ignore this exact archive path during candidate checks
  -h, --help         Show this help

The audit is read-only apart from private temporary files.  It reports every
matching archive member and exits nonzero for malformed inputs, strict KVAF
target inspection failures, or one runtime SONAME backed by more than one
SHA256.  A missing registry stanza marked optional is reported but is not an
error.  A present CANN package that uses a vendor-private/unsupported container
is reported as unaudited rather than guessed to be a conflict.  Candidate
checks remain fail-closed when such an opaque registered package exists.
EOF
}

audit_usage_error()
{
  echo "ERROR: registered package audit: $*" >&2
  usage >&2
  exit 2
}

audit_error()
{
  echo "ERROR: registered package audit: $*" >&2
}

ascend_home=${ASCEND_HOME_PATH:-}
registry_file=
replace_archive=
declare -a candidate_soname_bindings=()

while (( $# > 0 )); do
  case $1 in
    --ascend-home)
      (( $# >= 2 )) || audit_usage_error "--ascend-home needs a value"
      ascend_home=$2
      shift 2
      ;;
    --ascend-home=*)
      ascend_home=${1#*=}
      shift
      ;;
    --registry)
      (( $# >= 2 )) || audit_usage_error "--registry needs a value"
      registry_file=$2
      shift 2
      ;;
    --registry=*)
      registry_file=${1#*=}
      shift
      ;;
    --candidate-soname)
      (( $# >= 2 )) || audit_usage_error "--candidate-soname needs a value"
      candidate_soname_bindings+=("$2")
      shift 2
      ;;
    --candidate-soname=*)
      candidate_soname_bindings+=("${1#*=}")
      shift
      ;;
    --replace-archive)
      (( $# >= 2 )) || audit_usage_error "--replace-archive needs a value"
      replace_archive=$2
      shift 2
      ;;
    --replace-archive=*)
      replace_archive=${1#*=}
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      (( $# == 0 )) || audit_usage_error "positional arguments are not accepted"
      ;;
    *)
      audit_usage_error "unknown argument: $1"
      ;;
  esac
done

[[ -n ${ascend_home} ]] ||
  audit_usage_error "--ascend-home or ASCEND_HOME_PATH is required"
[[ -d ${ascend_home} && -r ${ascend_home} ]] ||
  audit_usage_error "CANN installation root is not a readable directory: ${ascend_home}"
ascend_home=$(cd "${ascend_home}" && pwd -P) ||
  audit_usage_error "could not resolve the CANN installation root"

if [[ -z ${registry_file} ]]; then
  registry_file="${ascend_home}/conf/ascend_package_load.ini"
fi
[[ ! -L ${registry_file} && -f ${registry_file} && -r ${registry_file} ]] ||
  audit_usage_error "registry is not a readable regular file: ${registry_file}"
if [[ -n ${replace_archive} ]]; then
  [[ ${replace_archive} == /* &&
     ${replace_archive} != *$'\t'* &&
     ${replace_archive} != *$'\n'* ]] ||
    audit_usage_error "--replace-archive must be a safe absolute path"
  replace_archive_basename=${replace_archive##*/}
  replace_archive_dir=${replace_archive%/*}
  replace_archive_dir=$(cd "${replace_archive_dir}" && pwd -P) ||
    audit_usage_error "--replace-archive parent is not a real directory"
  replace_archive="${replace_archive_dir}/${replace_archive_basename}"
fi

for required_command in awk grep mktemp python3 readelf realpath rm sha256sum sort tar; do
  command -v "${required_command}" >/dev/null 2>&1 ||
    audit_usage_error "${required_command} is required"
done

audit_tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/kvaf-a5-package-audit.XXXXXX") ||
  audit_usage_error "could not create a private temporary directory"
audit_cleanup()
{
  local audit_cleanup_rc=$?
  trap - EXIT
  rm -rf -- "${audit_tmp_dir}" || audit_cleanup_rc=1
  exit "${audit_cleanup_rc}"
}
trap audit_cleanup EXIT

audit_snapshot_regular()
{
  local source_path=$1
  local snapshot_path=$2

  python3 -c '
import hashlib, os, stat, sys
source, target = sys.argv[1:3]
read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
source_fd = os.open(source, read_flags)
try:
    before = os.fstat(source_fd)
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or
            before.st_uid != os.geteuid()):
        raise RuntimeError("unsafe snapshot source")
    write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    write_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    target_fd = os.open(target, write_flags, 0o600)
    digest = hashlib.sha256()
    try:
        remaining = before.st_size
        while remaining:
            data = os.read(source_fd, min(remaining, 1024 * 1024))
            if not data:
                raise RuntimeError("short snapshot read")
            digest.update(data)
            view = memoryview(data)
            while view:
                written = os.write(target_fd, view)
                if written <= 0:
                    raise RuntimeError("short snapshot write")
                view = view[written:]
            remaining -= len(data)
        if os.read(source_fd, 1):
            raise RuntimeError("snapshot source grew during read")
    finally:
        os.close(target_fd)
    after = os.fstat(source_fd)
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns)
    if identity(before) != identity(after):
        raise RuntimeError("snapshot source changed during read")
    print(digest.hexdigest())
finally:
    os.close(source_fd)
' "${source_path}" "${snapshot_path}"
}

registry_snapshot="${audit_tmp_dir}/registry.snapshot"
registry_sha256=$(audit_snapshot_regular \
  "${registry_file}" "${registry_snapshot}") ||
  audit_usage_error "could not create a stable registry snapshot"

registered_packages_file="${audit_tmp_dir}/registered-packages.tsv"
runtime_sonames_file="${audit_tmp_dir}/runtime-sonames.tsv"
candidate_sonames_file="${audit_tmp_dir}/candidate-sonames.tsv"
optional_missing_paths_file="${audit_tmp_dir}/optional-missing-paths"
: >"${runtime_sonames_file}"
: >"${candidate_sonames_file}"
: >"${optional_missing_paths_file}"

for candidate_binding in \
  ${candidate_soname_bindings[@]+"${candidate_soname_bindings[@]}"}; do
  candidate_soname=${candidate_binding%%=*}
  candidate_sha256=${candidate_binding#*=}
  [[ ${candidate_binding} == *=* &&
     ${candidate_soname} =~ ^[A-Za-z0-9_.+-]+$ &&
     ${candidate_sha256} =~ ^[0-9a-f]{64}$ ]] ||
    audit_usage_error \
      "--candidate-soname must be a safe SONAME followed by one SHA256"
  printf '%s\t%s\n' "${candidate_soname}" "${candidate_sha256}" \
    >>"${candidate_sonames_file}"
done
if [[ -s ${candidate_sonames_file} ]] &&
   ! awk -F '\t' '!seen[$1]++ { next } { exit 1 }' \
       "${candidate_sonames_file}"; then
  audit_usage_error "duplicate --candidate-soname binding"
fi

# A stanza starts at name:.  Comments may occupy a whole line or follow a
# value; surrounding whitespace around keys, separators, and values is
# ignored.  Only safe relative package_path values paired with tar.gz names
# become effective package registrations.
if ! awk '
  function trim(value) {
    sub(/^[[:space:]]+/, "", value)
    sub(/[[:space:]]+$/, "", value)
    return value
  }
  function uncomment(value, hash, semicolon, cut) {
    hash = index(value, "#")
    semicolon = index(value, ";")
    if (hash && semicolon) cut = hash < semicolon ? hash : semicolon
    else if (hash) cut = hash
    else cut = semicolon
    if (cut) value = substr(value, 1, cut - 1)
    return trim(value)
  }
  function report(message) {
    print "ERROR: registered package audit: registry line " \
      name_line ": " message > "/dev/stderr"
    invalid = 1
  }
  function finish_stanza() {
    if (name == "") return
    if (name ~ /[.]tar[.]gz$/) {
      if (name !~ /^[A-Za-z0-9][A-Za-z0-9._-]*[.]tar[.]gz$/) {
        report("unsafe registered tar.gz name: " name)
      } else if (package_path_count != 1 || package_path == "") {
        report("tar.gz stanza must contain exactly one package_path")
      } else if (package_path !~ \
                   /^[A-Za-z0-9._-]+(\/[A-Za-z0-9._-]+)*$/ ||
                 index(package_path, "..") != 0) {
        report("unsafe package_path for " name ": " package_path)
      } else if (optional_count > 1) {
        report("tar.gz stanza contains duplicate optional values")
      } else if (optional != "" && optional != "true" &&
                 optional != "false") {
        report("invalid optional value for " name ": " optional)
      } else {
        if (optional == "") optional = "false"
        print name "\t" package_path "\t" optional
      }
    }
    name = ""
    package_path = ""
    package_path_count = 0
    optional = ""
    optional_count = 0
    name_line = 0
  }
  {
    line = $0
    sub(/\r$/, "", line)
    line = trim(line)
    if (line == "" || line ~ /^[#;]/) next
    separator = index(line, ":")
    if (!separator) next
    key = tolower(trim(substr(line, 1, separator - 1)))
    value = uncomment(substr(line, separator + 1))
    if (key == "name") {
      finish_stanza()
      name = value
      name_line = NR
    } else if (key == "package_path" && name != "") {
      package_path = value
      package_path_count++
    } else if (key == "optional" && name != "") {
      optional = tolower(value)
      optional_count++
    }
  }
  END {
    finish_stanza()
    exit invalid ? 1 : 0
  }
' "${registry_snapshot}" >"${registered_packages_file}"; then
  exit 1
fi

emit_values()
{
  local record_name=$1
  local archive_path=$2
  local member=$3
  local values_file=$4
  local value

  if [[ ! -s ${values_file} ]]; then
    printf '%s\tarchive_path=%s\tmember=%s\tvalue=<none>\n' \
      "${record_name}" "${archive_path}" "${member}"
    return 0
  fi
  while IFS= read -r value || [[ -n ${value} ]]; do
    printf '%s\tarchive_path=%s\tmember=%s\tvalue=%s\n' \
      "${record_name}" "${archive_path}" "${member}" "${value}"
  done <"${values_file}"
}

join_values()
{
  awk '
    BEGIN { first = 1 }
    {
      if (!first) printf ","
      printf "%s", $0
      first = 0
    }
    END {
      if (first) printf "<none>"
      printf "\n"
    }
  ' "$1"
}

archive_container_hint()
{
  python3 -c '
import sys

path = sys.argv[1]
with open(path, "rb") as stream:
    prefix = stream.read(512)

if prefix.startswith(b"\x1f\x8b"):
    kind = "gzip"
elif prefix.startswith(b"\xfd7zXZ\x00"):
    kind = "xz"
elif prefix.startswith(b"BZh"):
    kind = "bzip2"
elif prefix.startswith(b"\x28\xb5\x2f\xfd"):
    kind = "zstd"
elif prefix.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
    kind = "zip"
elif len(prefix) >= 262 and prefix[257:262] == b"ustar":
    kind = "tar"
else:
    kind = "opaque"
print(kind)
' "$1"
}

is_strict_kvaf_archive()
{
  local package_name=$1
  local archive_path=$2

  [[ ${package_name} == *kvaf* ||
     ( -n ${replace_archive} && ${archive_path} == "${replace_archive}" ) ]]
}

printf 'STATIC_AUDIT\tregistry=%s\tascend_home=%s\n' \
  "${registry_file}" "${ascend_home}"

audit_errors=0
registered_packages=0
library_count=0
optional_missing_count=0
opaque_package_count=0
unaudited_package_count=0
candidate_unknown_count=0

while IFS=$'\t' read -r package_name package_path package_optional; do
  [[ -n ${package_name} && -n ${package_path} ]] || continue
  registered_packages=$((registered_packages + 1))
  normalized_package_path=${package_path}
  while [[ ${normalized_package_path} == ./* ]]; do
    normalized_package_path=${normalized_package_path#./}
  done
  while [[ ${normalized_package_path} == */./* ]]; do
    normalized_package_path=${normalized_package_path/\/\.\//\/}
  done
  [[ ${normalized_package_path} != . ]] || normalized_package_path=
  if [[ -n ${normalized_package_path} ]]; then
    archive_path="${ascend_home}/${normalized_package_path}/${package_name}"
  else
    archive_path="${ascend_home}/${package_name}"
  fi
  strict_kvaf_archive=0
  if is_strict_kvaf_archive "${package_name}" "${archive_path}"; then
    strict_kvaf_archive=1
  fi
  package_library_count_before=${library_count}
  listing_file="${audit_tmp_dir}/archive-${registered_packages}.listing"
  candidates_file="${audit_tmp_dir}/archive-${registered_packages}.candidates"
  unique_candidates_file="${audit_tmp_dir}/archive-${registered_packages}.unique"
  duplicate_candidates_file="${audit_tmp_dir}/archive-${registered_packages}.duplicates"
  aliases_file="${audit_tmp_dir}/archive-${registered_packages}.aliases"
  regular_identities_file="${audit_tmp_dir}/archive-${registered_packages}.regular-identities"
  : >"${aliases_file}"
  : >"${regular_identities_file}"

  if [[ -L ${archive_path} ]]; then
    audit_error "registered archive path is a symlink: ${archive_path}"
    audit_errors=$((audit_errors + 1))
    continue
  fi
  if [[ ! -e ${archive_path} && ${package_optional} == true &&
        ${strict_kvaf_archive} -eq 0 ]]; then
    printf 'PACKAGE_UNAVAILABLE\tarchive_path=%s\tpackage_path=%s\tname=%s\toptional=true\treason=optional-missing\n' \
      "${archive_path}" "${package_path}" "${package_name}"
    printf '%s\n' "${archive_path}" >>"${optional_missing_paths_file}"
    optional_missing_count=$((optional_missing_count + 1))
    continue
  fi
  if [[ ! -f ${archive_path} || ! -r ${archive_path} ]]; then
    audit_error "registered archive is missing or unreadable: ${archive_path}"
    audit_errors=$((audit_errors + 1))
    continue
  fi
  archive_real=$(realpath "${archive_path}") || {
    audit_error "could not resolve registered archive: ${archive_path}"
    audit_errors=$((audit_errors + 1))
    continue
  }
  case ${archive_real} in
    "${ascend_home}"/*) ;;
    *)
      audit_error "registered archive resolves outside CANN root: ${archive_path} -> ${archive_real}"
      audit_errors=$((audit_errors + 1))
      continue
      ;;
  esac
  if [[ ${archive_real} != "${archive_path}" ]]; then
    audit_error "registered archive path contains a symlink or alias: ${archive_path} -> ${archive_real}"
    audit_errors=$((audit_errors + 1))
    continue
  fi
  archive_snapshot="${audit_tmp_dir}/archive-${registered_packages}.tar.gz"
  if ! archive_sha256=$(audit_snapshot_regular \
      "${archive_path}" "${archive_snapshot}"); then
    audit_error "could not snapshot registered archive: ${archive_path}"
    audit_errors=$((audit_errors + 1))
    continue
  fi
  printf 'PACKAGE\tarchive_path=%s\tarchive_sha256=%s\tpackage_path=%s\tname=%s\n' \
    "${archive_path}" "${archive_sha256}" "${package_path}" "${package_name}"

  container_hint=$(archive_container_hint "${archive_snapshot}") || {
    audit_error "could not classify registered archive: ${archive_path}"
    audit_errors=$((audit_errors + 1))
    continue
  }
  listing_error_file="${listing_file}.stderr"
  if ! tar -tf "${archive_snapshot}" >"${listing_file}" \
       2>"${listing_error_file}"; then
    if (( strict_kvaf_archive != 0 )); then
      audit_error "strict KVAF archive is not a readable tar container: ${archive_path} container_hint=${container_hint}"
      audit_errors=$((audit_errors + 1))
      continue
    fi
    printf 'PACKAGE_UNAUDITED\tarchive_path=%s\tarchive_sha256=%s\tname=%s\tcontainer_hint=%s\treason=vendor-private-or-unsupported-container\n' \
      "${archive_path}" "${archive_sha256}" "${package_name}" \
      "${container_hint}"
    unaudited_package_count=$((unaudited_package_count + 1))
    if [[ ${container_hint} == opaque ]]; then
      opaque_package_count=$((opaque_package_count + 1))
    fi
    if [[ -s ${candidate_sonames_file} ]]; then
      printf 'CANDIDATE_UNAUDITED_SCOPE\tarchive_path=%s\tarchive_sha256=%s\tname=%s\tcontainer_hint=%s\n' \
        "${archive_path}" "${archive_sha256}" "${package_name}" \
        "${container_hint}" >&2
      candidate_unknown_count=$((candidate_unknown_count + 1))
    fi
    if [[ -L ${archive_path} || ! -f ${archive_path} ]] ||
       [[ $(sha256sum -- "${archive_path}" | awk '{print $1}') != \
          "${archive_sha256}" ]]; then
      audit_error "registered archive changed during audit: ${archive_path}"
      audit_errors=$((audit_errors + 1))
    fi
    continue
  fi
  printf 'PACKAGE_CONTAINER\tarchive_path=%s\tcontainer_hint=%s\taudit_format=tar-auto\n' \
    "${archive_path}" "${container_hint}"
  awk '
    {
      member = $0
      count = split(member, components, "/")
      base = components[count]
      if (base ~ /^liburma[.]so.*$/ ||
          base ~ /^libkvaf.*[.]so.*$/ ||
          base ~ /^libummu.*[.]so.*$/ ||
          base ~ /^libudma.*[.]so.*$/) {
        print member
      }
    }
  ' "${listing_file}" >"${candidates_file}"
  : >"${duplicate_candidates_file}"
  awk -v duplicates="${duplicate_candidates_file}" '
    {
      values[++value_count] = $0
      counts[$0]++
    }
    END {
      for (item_index = 1; item_index <= value_count; item_index++) {
        value = values[item_index]
        if (counts[value] == 1) {
          print value
        } else if (!reported[value]++) {
          print value > duplicates
        }
      }
    }
  ' "${candidates_file}" >"${unique_candidates_file}"
  while IFS= read -r duplicate_member || [[ -n ${duplicate_member} ]]; do
    [[ -n ${duplicate_member} ]] || continue
    audit_error "duplicate matching member is ambiguous: archive=${archive_path} member=${duplicate_member}"
    audit_errors=$((audit_errors + 1))
  done <"${duplicate_candidates_file}"

  member_index=0
  while IFS= read -r member || [[ -n ${member} ]]; do
    [[ -n ${member} ]] || continue
    member_index=$((member_index + 1))
    normalized_member=${member}
    while [[ ${normalized_member} == ./* ]]; do
      normalized_member=${normalized_member#./}
    done
    while [[ ${normalized_member} == */./* ]]; do
      normalized_member=${normalized_member/\/\.\//\/}
    done
    if [[ ${member} == *$'\t'* ||
          ! ${normalized_member} =~ ^[A-Za-z0-9._+-]+(/[A-Za-z0-9._+-]+)*$ ||
          ${normalized_member} == *..* ]]; then
      audit_error "matching member has an unsafe path: archive=${archive_path} member=${member}"
      audit_errors=$((audit_errors + 1))
      continue
    fi

    # CANN packages commonly carry liburma.so as a symlink to a versioned
    # regular ELF.  tar -xO deliberately emits no bytes for such an alias; do
    # not misclassify that valid packaging pattern as an empty/corrupt ELF.
    # The target must remain a safe in-archive path whose basename is also in
    # the audited library families, so the regular target is inspected below.
    if ! member_metadata=$(LC_ALL=C tar -tvf "${archive_snapshot}" -- "${member}"); then
      audit_error "could not inspect matching member type: archive=${archive_path} member=${member}"
      audit_errors=$((audit_errors + 1))
      continue
    fi
    member_type=${member_metadata:0:1}
    if [[ ${member_type} == l || ${member_type} == h ]]; then
      if [[ ${member_type} == l && ${member_metadata} == *' -> '* ]]; then
        alias_target=${member_metadata##* -> }
        alias_directory=${normalized_member%/*}
        [[ ${alias_directory} != "${normalized_member}" ]] || alias_directory=
        if [[ -n ${alias_directory} ]]; then
          alias_target_path=${alias_directory}/${alias_target}
        else
          alias_target_path=${alias_target}
        fi
      elif [[ ${member_type} == h && ${member_metadata} == *' link to '* ]]; then
        alias_target=${member_metadata##* link to }
        alias_target_path=${alias_target}
      else
        audit_error "could not parse matching library alias: archive=${archive_path} member=${member}"
        audit_errors=$((audit_errors + 1))
        continue
      fi
      alias_target_path=${alias_target_path#./}
      while [[ ${alias_target_path} == ./* ]]; do
        alias_target_path=${alias_target_path#./}
      done
      while [[ ${alias_target_path} == */./* ]]; do
        alias_target_path=${alias_target_path/\/\.\//\/}
      done
      alias_target_base=${alias_target_path##*/}
      if [[ ! ${alias_target_path} =~ ^[A-Za-z0-9._+-]+(/[A-Za-z0-9._+-]+)*$ ||
            ${alias_target_path} == *..* ||
            ! ${alias_target_base} =~ ^(liburma|libkvaf.*|libummu.*|libudma.*)[.]so.*$ ]]; then
        audit_error "matching library alias has an unsafe or unaudited target: archive=${archive_path} member=${member} target=${alias_target_path}"
        audit_errors=$((audit_errors + 1))
        continue
      fi
      printf 'LIBRARY_ALIAS\tarchive_path=%s\tmember=%s\ttype=%s\ttarget=%s\n' \
        "${archive_path}" "${member}" "${member_type}" "${alias_target_path}"
      printf '%s\t%s\n' "${normalized_member}" "${alias_target_path}" \
        >>"${aliases_file}"
      continue
    fi
    if [[ ${member_type} != - ]]; then
      audit_error "matching library member is not a regular file or safe alias: archive=${archive_path} member=${member} type=${member_type}"
      audit_errors=$((audit_errors + 1))
      continue
    fi
    library_count=$((library_count + 1))

    member_file="${audit_tmp_dir}/member-${registered_packages}-${member_index}.elf"
    header_file="${member_file}.header"
    dynamic_file="${member_file}.dynamic"
    notes_file="${member_file}.notes"
    symbols_file="${member_file}.symbols"
    needed_file="${member_file}.needed"
    sonames_file="${member_file}.sonames"
    build_ids_file="${member_file}.build-ids"
    exports_file="${member_file}.exports"

    if ! tar -xOf "${archive_snapshot}" -- "${member}" >"${member_file}"; then
      audit_error "could not stream matching member: archive=${archive_path} member=${member}"
      audit_errors=$((audit_errors + 1))
      continue
    fi
    if ! member_sha256=$(sha256sum -- "${member_file}" | awk '{print $1}'); then
      audit_error "could not hash matching member: archive=${archive_path} member=${member}"
      audit_errors=$((audit_errors + 1))
      continue
    fi
    if ! readelf -h "${member_file}" >"${header_file}" 2>/dev/null; then
      audit_error "matching member is not an inspectable ELF: archive=${archive_path} member=${member} sha256=${member_sha256}"
      audit_errors=$((audit_errors + 1))
      continue
    fi
    if ! readelf -d "${member_file}" >"${dynamic_file}" 2>/dev/null ||
       ! awk '/Dynamic section/ { found = 1 } END { exit found ? 0 : 1 }' \
         "${dynamic_file}"; then
      audit_error "matching ELF has no inspectable dynamic section: archive=${archive_path} member=${member} sha256=${member_sha256}"
      audit_errors=$((audit_errors + 1))
      continue
    fi
    if ! readelf -n "${member_file}" >"${notes_file}" 2>/dev/null; then
      audit_error "could not inspect ELF notes: archive=${archive_path} member=${member}"
      audit_errors=$((audit_errors + 1))
      : >"${notes_file}"
    fi
    if ! readelf --dyn-syms --wide "${member_file}" \
         >"${symbols_file}" 2>/dev/null; then
      audit_error "could not inspect ELF dynamic exports: archive=${archive_path} member=${member}"
      audit_errors=$((audit_errors + 1))
      : >"${symbols_file}"
    fi

    awk '
      /\(NEEDED\)/ {
        value = $0
        sub(/^.*\[/, "", value)
        sub(/\].*$/, "", value)
        if (value != "") print value
      }
    ' "${dynamic_file}" | sort -u >"${needed_file}"
    awk '
      /\(SONAME\)/ {
        value = $0
        sub(/^.*\[/, "", value)
        sub(/\].*$/, "", value)
        if (value != "") print value
      }
    ' "${dynamic_file}" | sort -u >"${sonames_file}"
    awk '
      /Build ID:/ {
        for (field = 1; field < NF; field++) {
          if ($field == "ID:") print tolower($(field + 1))
        }
      }
    ' "${notes_file}" | sort -u >"${build_ids_file}"
    awk '
      $1 ~ /^[0-9]+:$/ && $7 != "UND" &&
      ($5 == "GLOBAL" || $5 == "WEAK") &&
      ($6 == "DEFAULT" || $6 == "PROTECTED") && $8 != "" {
        print $8
      }
    ' "${symbols_file}" | sort -u >"${exports_file}"

    soname_count=$(awk 'END { print NR + 0 }' "${sonames_file}")
    dt_soname=$(join_values "${sonames_file}")
    if (( soname_count == 0 )); then
      runtime_soname=${member##*/}
    elif (( soname_count == 1 )); then
      runtime_soname=${dt_soname}
    else
      runtime_soname='<invalid-multiple-soname>'
      audit_error "matching ELF has multiple DT_SONAME values: archive=${archive_path} member=${member} values=${dt_soname}"
      audit_errors=$((audit_errors + 1))
    fi
    if [[ ${runtime_soname} == *$'\t'* || ${runtime_soname} == *$'\n'* ]]; then
      audit_error "matching ELF has an unsafe runtime SONAME: archive=${archive_path} member=${member}"
      audit_errors=$((audit_errors + 1))
      runtime_soname='<invalid-soname>'
    fi
    build_ids=$(join_values "${build_ids_file}")
    if [[ ${build_ids} != '<none>' ]] &&
       ! awk 'BEGIN { valid = 1 }
         $0 !~ /^[0-9a-f]+$/ { valid = 0 }
         END { exit valid ? 0 : 1 }' "${build_ids_file}"; then
      audit_error "matching ELF has a malformed GNU build-id: archive=${archive_path} member=${member} value=${build_ids}"
      audit_errors=$((audit_errors + 1))
    fi

    printf 'LIBRARY\tarchive_path=%s\tmember=%s\tsha256=%s\tgnu_build_id=%s\tdt_soname=%s\truntime_soname=%s\n' \
      "${archive_path}" "${member}" "${member_sha256}" "${build_ids}" \
      "${dt_soname}" "${runtime_soname}"
    emit_values DT_NEEDED "${archive_path}" "${member}" "${needed_file}"
    emit_values DYNAMIC_EXPORT "${archive_path}" "${member}" "${exports_file}"
    printf '%s\t%s\t%s\n' \
      "${normalized_member}" "${member_sha256}" "${runtime_soname}" \
      >>"${regular_identities_file}"

    if [[ -z ${replace_archive} ||
          ${archive_path} != "${replace_archive}" ]]; then
      # Both names can select or overwrite a library at package extraction
      # time.  DT_SONAME models dependency resolution after load; the archive
      # member basename models the loader/package key before load.  Auditing
      # only DT_SONAME would miss a foreign libkvaf_*.so whose embedded SONAME
      # claims to be some other library.
      member_loader_key=${normalized_member##*/}
      printf '%s\t%s\t%s\t%s\n' \
        "${member_loader_key}" "${member_sha256}" "${archive_path}" \
        "${member}" >>"${runtime_sonames_file}"
      if [[ ${runtime_soname} != '<invalid-multiple-soname>' &&
            ${runtime_soname} != '<invalid-soname>' &&
            ${runtime_soname} != "${member_loader_key}" ]]; then
        printf '%s\t%s\t%s\t%s\n' \
          "${runtime_soname}" "${member_sha256}" "${archive_path}" \
          "${member}" >>"${runtime_sonames_file}"
      fi
    fi
  done <"${unique_candidates_file}"

  while IFS=$'\t' read -r alias_member alias_target; do
    [[ -n ${alias_member} && -n ${alias_target} ]] || continue
    # Resolve a complete in-archive alias chain.  Common packages use
    # liburma.so -> liburma.so.1 -> liburma.so.1.2; requiring the first hop to
    # be a regular ELF incorrectly rejects that valid layout.  Ambiguity,
    # cycles and missing terminal ELFs still fail closed.
    alias_resolution=$(awk -F '\t' -v target="${alias_target}" '
      FNR == NR {
        alias_target[$1] = $2
        alias_count[$1]++
        alias_total++
        next
      }
      {
        regular_sha[$1] = $2
        regular_soname[$1] = $3
        regular_count[$1]++
      }
      END {
        current = target
        for (step = 0; step <= alias_total + 1; step++) {
          if (regular_count[current] == 1) {
            print current "\t" regular_sha[current] "\t" \
              regular_soname[current]
            exit 0
          }
          if (regular_count[current] != 0 || alias_count[current] != 1 ||
              seen[current]++) {
            exit 1
          }
          current = alias_target[current]
        }
        exit 1
      }
    ' "${aliases_file}" "${regular_identities_file}") || {
      audit_error "matching library alias target was not resolved to exactly one regular ELF: archive=${archive_path} member=${alias_member} target=${alias_target}"
      audit_errors=$((audit_errors + 1))
      continue
    }
    IFS=$'\t' read -r alias_resolved_target alias_target_sha \
      alias_target_soname <<<"${alias_resolution}"
    alias_loader_key=${alias_member##*/}
    printf 'LIBRARY_ALIAS_RESOLVED\tarchive_path=%s\tmember=%s\tloader_key=%s\ttarget=%s\ttarget_sha256=%s\ttarget_dt_soname=%s\tresolved_target=%s\n' \
      "${archive_path}" "${alias_member}" "${alias_loader_key}" \
      "${alias_target}" "${alias_target_sha}" "${alias_target_soname}" \
      "${alias_resolved_target}"
    if [[ -z ${replace_archive} || ${archive_path} != "${replace_archive}" ]]; then
      printf '%s\t%s\t%s\t%s\n' \
        "${alias_loader_key}" "${alias_target_sha}" "${archive_path}" \
        "${alias_member}" >>"${runtime_sonames_file}"
    fi
  done <"${aliases_file}"

  if (( strict_kvaf_archive != 0 &&
        library_count == package_library_count_before )); then
    audit_error \
      "strict KVAF archive contains no inspectable runtime library: ${archive_path}"
    audit_errors=$((audit_errors + 1))
  fi

  if [[ -L ${archive_path} || ! -f ${archive_path} ]] ||
     [[ $(sha256sum -- "${archive_path}" | awk '{print $1}') != \
        "${archive_sha256}" ]]; then
    audit_error \
      "registered archive changed during audit: ${archive_path}"
    audit_errors=$((audit_errors + 1))
  fi
done <"${registered_packages_file}"

while IFS= read -r optional_missing_path ||
      [[ -n ${optional_missing_path} ]]; do
  [[ -n ${optional_missing_path} ]] || continue
  if [[ -e ${optional_missing_path} || -L ${optional_missing_path} ]]; then
    audit_error \
      "optional registered archive appeared during audit: ${optional_missing_path}"
    audit_errors=$((audit_errors + 1))
  fi
done <"${optional_missing_paths_file}"

if [[ -L ${registry_file} || ! -f ${registry_file} ]] ||
   [[ $(sha256sum -- "${registry_file}" | awk '{print $1}') != \
      "${registry_sha256}" ]]; then
  audit_error "AICPU package registry changed during audit: ${registry_file}"
  audit_errors=$((audit_errors + 1))
fi

candidate_conflict_count=0
while IFS=$'\t' read -r candidate_soname candidate_sha256; do
  [[ -n ${candidate_soname} ]] || continue
  candidate_has_conflict=0
  while IFS=$'\t' read -r runtime_soname runtime_sha256 \
    runtime_archive runtime_member; do
    [[ ${runtime_soname} == "${candidate_soname}" ]] || continue
    [[ -z ${replace_archive} || ${runtime_archive} != "${replace_archive}" ]] ||
      continue
    if (( candidate_has_conflict == 0 )); then
      printf 'CANDIDATE_CONFLICT\truntime_soname=%s\tcandidate_sha256=%s\n' \
        "${candidate_soname}" "${candidate_sha256}" >&2
      candidate_has_conflict=1
      candidate_conflict_count=$((candidate_conflict_count + 1))
    fi
    printf 'CANDIDATE_CONFLICT_MEMBER\truntime_soname=%s\tsha256=%s\tarchive_path=%s\tmember=%s\n' \
      "${runtime_soname}" "${runtime_sha256}" "${runtime_archive}" \
      "${runtime_member}" >&2
  done <"${runtime_sonames_file}"
done <"${candidate_sonames_file}"

if ! conflict_count=$(
  sort -t $'\t' -k1,1 -k2,2 -k3,3 -k4,4 "${runtime_sonames_file}" |
    awk -F '\t' '
      {
        soname = $1
        sha = $2
        record_count++
        record_soname[record_count] = soname
        record_sha[record_count] = sha
        record_archive[record_count] = $3
        record_member[record_count] = $4
        if (!(soname in soname_seen)) {
          soname_seen[soname] = 1
          soname_order[++soname_count] = soname
        }
        identity = soname SUBSEP sha
        if (!(identity in identity_seen)) {
          identity_seen[identity] = 1
          sha_count[soname]++
        }
      }
      END {
        conflicts = 0
        for (soname_index = 1; soname_index <= soname_count;
             soname_index++) {
          soname = soname_order[soname_index]
          if (sha_count[soname] <= 1) continue
          conflicts++
          printf "CONFLICT\truntime_soname=%s\tdistinct_sha256=%d\n", \
            soname, sha_count[soname] > "/dev/stderr"
          for (record_index = 1; record_index <= record_count;
               record_index++) {
            if (record_soname[record_index] != soname) continue
            printf "CONFLICT_MEMBER\truntime_soname=%s\tsha256=%s\tarchive_path=%s\tmember=%s\n", \
              soname, record_sha[record_index], \
              record_archive[record_index], record_member[record_index] \
              > "/dev/stderr"
          }
        }
        print conflicts
      }
    '
); then
  audit_error "could not compare runtime SONAME identities"
  audit_errors=$((audit_errors + 1))
  conflict_count=0
fi

audit_coverage=complete
if (( unaudited_package_count > 0 || optional_missing_count > 0 )); then
  audit_coverage=partial
fi

printf 'SUMMARY\tregistered_packages=%d\tlibraries=%d\tconflicts=%d\tcandidate_conflicts=%d\terrors=%d\toptional_missing=%d\topaque_packages=%d\tunaudited_packages=%d\tcandidate_unknowns=%d\tcoverage=%s\n' \
  "${registered_packages}" "${library_count}" "${conflict_count}" \
  "${candidate_conflict_count}" "${audit_errors}" \
  "${optional_missing_count}" "${opaque_package_count}" \
  "${unaudited_package_count}" "${candidate_unknown_count}" \
  "${audit_coverage}"

if (( audit_errors > 0 || conflict_count > 0 ||
      candidate_conflict_count > 0 || candidate_unknown_count > 0 )); then
  exit 1
fi
