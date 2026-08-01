#!/usr/bin/env python3
"""Durable, no-symlink filesystem operations for direct-URMA recovery state."""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path
from typing import Sequence


class StateFileError(RuntimeError):
    """A recovery-state operation could not be completed safely."""


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _open_directory(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise StateFileError(f"state parent is not a directory: {path}")
    if metadata.st_uid != os.geteuid():
        os.close(descriptor)
        raise StateFileError(
            f"state parent is not owned by uid {os.geteuid()}: {path}"
        )
    return descriptor


def _regular_metadata(directory_fd: int, name: str) -> os.stat_result:
    metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise StateFileError(f"state entry is not a regular file: {name}")
    if metadata.st_uid != os.geteuid():
        raise StateFileError(
            f"state entry is not owned by uid {os.geteuid()}: {name}"
        )
    if metadata.st_nlink != 1:
        raise StateFileError(
            f"state entry must have exactly one hard link: {name}"
        )
    return metadata


def durable_publish(source: Path, target: Path) -> None:
    """Fsync source, atomically replace target, then fsync its directory."""

    source = _absolute(source)
    target = _absolute(target)
    if source.parent != target.parent:
        raise StateFileError("source and target must share one directory")
    directory_fd = _open_directory(target.parent)
    try:
        source_metadata = _regular_metadata(directory_fd, source.name)
        try:
            _regular_metadata(directory_fd, target.name)
        except FileNotFoundError:
            pass

        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        source_fd = os.open(source.name, flags, dir_fd=directory_fd)
        try:
            opened_metadata = os.fstat(source_fd)
            if (
                opened_metadata.st_dev != source_metadata.st_dev
                or opened_metadata.st_ino != source_metadata.st_ino
            ):
                raise StateFileError("state source changed while opening")
            os.fsync(source_fd)
        finally:
            os.close(source_fd)

        os.replace(
            source.name,
            target.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def durable_remove(target: Path) -> None:
    """Unlink a regular state file and durably record the deletion."""

    target = _absolute(target)
    directory_fd = _open_directory(target.parent)
    try:
        try:
            _regular_metadata(directory_fd, target.name)
        except FileNotFoundError:
            return
        os.unlink(target.name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def durable_sync(paths: Sequence[Path]) -> None:
    """Fsync existing regular files/directories without following symlinks."""

    for original in paths:
        path = _absolute(original)
        metadata = os.lstat(path)
        if metadata.st_uid != os.geteuid():
            raise StateFileError(
                f"sync target is not owned by uid {os.geteuid()}: {path}"
            )
        if stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1:
                raise StateFileError(
                    f"sync file must have exactly one hard link: {path}"
                )
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
        elif stat.S_ISDIR(metadata.st_mode):
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
        else:
            raise StateFileError(
                f"sync target is not a regular file/directory: {path}"
            )
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    publish = subparsers.add_parser("publish")
    publish.add_argument("source", type=Path)
    publish.add_argument("target", type=Path)

    remove = subparsers.add_parser("remove")
    remove.add_argument("target", type=Path)

    sync = subparsers.add_parser("sync")
    sync.add_argument("paths", nargs="+", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.operation == "publish":
            durable_publish(args.source, args.target)
        elif args.operation == "remove":
            durable_remove(args.target)
        else:
            durable_sync(args.paths)
    except (OSError, StateFileError) as error:
        print(f"ERROR: durable state operation failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
