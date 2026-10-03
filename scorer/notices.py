#!/usr/bin/env python3
# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Third-party licence notices for what OpenHydra ships: the licence texts of the code compiled into
a binary, collected from the exact sources it was built from.

  notices.py llamacpp <llama.cpp tree> <out>     the patched llama-server (openhydra-scorer)
  notices.py cargo <manifest> <package> <out> [--target T] [--features F] [--no-default-features]
                   [--own-root DIR]               a Rust binary: every crate it links (cargo tree)
  notices.py merge <out> <part>...               one file from several parts

Standard library only. A llama.cpp vendored component this script doesn't know fails the build, so a
llama.cpp update can't ship code whose licence nobody looked at.
"""
import hashlib
import json
import os
import re
import subprocess
import sys

RULE = "=" * 78

# llama-server links llama.cpp (ggml, llama, common, mtmd, the server) and these vendored libraries:
# (component, licence, source file, how to take the notice from it).
LLAMACPP = [
    ("llama.cpp / ggml", "MIT", "LICENSE", "file"),
    ("cpp-httplib", "MIT", "vendor/cpp-httplib/LICENSE", "file"),
    ("nlohmann/json", "MIT", "licenses/LICENSE-jsonhpp", "file"),
    ("xxHash", "BSD-2-Clause", "vendor/hash/xxhash/LICENSE", "file"),
    ("rotate-bits", "see text", "vendor/hash/rotate-bits/LICENSE.md", "file"),
    ("SHA-256 (Igor Pavlov)", "Public domain", "vendor/hash/sha256/LICENSE", "file"),
    ("SHA-1 (Steve Reid)", "Public domain", "vendor/hash/sha1/sha1.c", ("/*", "*/")),
    ("subprocess.h (sheredom)", "Unlicense", "vendor/sheredom/subprocess.h", ("/*", "*/")),
    ("stb_image", "MIT or public domain", "vendor/stb/stb_image.h", ("This software is available under 2 licenses", "*/")),
    ("miniaudio", "MIT-0 or public domain", "vendor/miniaudio/miniaudio.h", ("This software is available as a choice", "*/")),
]
# Vendored directories the list above covers.
LLAMACPP_VENDOR = {"cpp-httplib", "hash", "miniaudio", "nlohmann", "sheredom", "stb"}

LICENSE_FILE = re.compile(r"^(licen[cs]e|copying|notice|copyright|unlicense)([-_.].*)?$", re.I)

MIT_TEMPLATE = """Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
THE USE OR OTHER DEALINGS IN THE SOFTWARE."""


def die(msg):
    sys.exit(f"notices.py: {msg}")


def read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def section(title, licence, texts, source=None):
    """One component: a header, then its notice texts (already deduplicated by the caller)."""
    out = [RULE, title, f"Licence: {licence}"]
    if source:
        out.append(f"Source: {source}")
    out.append(RULE)
    for t in texts:
        out += ["", t.strip("\n")]
    return "\n".join(out) + "\n\n"


def llamacpp(src, out):
    vendored = {d for d in os.listdir(os.path.join(src, "vendor")) if os.path.isdir(os.path.join(src, "vendor", d))}
    if vendored - LLAMACPP_VENDOR:
        die(f"unknown vendored code in llama.cpp: {sorted(vendored - LLAMACPP_VENDOR)} (add its licence to LLAMACPP)")
    parts = []
    for name, licence, rel, how in LLAMACPP:
        path = os.path.join(src, rel)
        if not os.path.isfile(path):
            die(f"{rel} is missing (llama.cpp changed: review LLAMACPP)")
        text = read(path)
        if how != "file":
            start, end = how
            i = text.find(start)
            j = text.find(end, i + len(start)) if i >= 0 else -1
            if i < 0 or j < 0:
                die(f"no licence block in {rel}")
            text = text[i + (len(start) if start == "/*" else 0) : j]
        parts.append(section(name, licence, [text]))
    write(out, "openhydra-scorer (llama.cpp's llama-server with OpenHydra's prompt-scoring patch)", parts)


def cargo(manifest, package, out, target=None, features=None, no_default=False, own_root=None):
    # `cargo tree` resolves for one target and needs only the crates that build uses (offline-capable),
    # unlike `cargo metadata`, which wants every platform's dependencies.
    cmd = ["cargo", "tree", "--manifest-path", manifest, "-p", package, "-e", "normal", "--prefix", "none",
           "-f", "{p}\t{l}\t{r}"]
    if target:
        cmd += ["--target", target]
    if features:
        cmd += ["--features", features]
    if no_default:
        cmd += ["--no-default-features"]
    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode != 0:
        die(f"cargo tree failed:\n{run.stderr}")
    manifest_dir = os.path.dirname(os.path.realpath(manifest))
    own_root = os.path.realpath(own_root) if own_root else None
    cargo_home = os.environ.get("CARGO_HOME") or os.path.join(os.path.expanduser("~"), ".cargo")
    crates = {}
    for line in run.stdout.splitlines():
        line = re.sub(r" \(\*\)$", "", line)  # cargo tree's "already shown" marker
        spec, licence, repo = (line.split("\t") + ["", ""])[:3]
        m = re.match(r"^(\S+) v(\S+)(.*)$", spec)
        if m:
            # "(proc-macro)" and "(*)" are annotations; any other parenthesised part is the source
            src = [g for g in re.findall(r"\(([^)]*)\)", m.group(3)) if g not in ("proc-macro", "*")]
            crates[(m.group(1), m.group(2))] = (src[0] if src else "", licence, repo)
    parts, by_hash = [], {}
    for (name, version), (src, licence, repo) in sorted(crates.items()):
        if src.startswith("/"):
            pdir, walk_up = os.path.realpath(src), manifest_dir
            if own_root and (pdir == own_root or pdir.startswith(own_root + os.sep)):
                continue  # our own code
        elif src.startswith("http"):
            pdir, walk_up = git_checkout(cargo_home, name, src)
        else:
            pdir, walk_up = registry_dir(cargo_home, name, version), None
        if pdir is None:
            die(f"no local sources for {name} {version} ({src or 'crates.io'}): build it first")
        title = f"{name} {version}"
        source = repo or f"https://crates.io/crates/{name}/{version}"
        texts = []
        for f in licence_files(pdir, walk_up):
            text = read(f)
            h = hashlib.sha256(text.encode()).hexdigest()
            if h in by_hash:
                texts.append(f"[{os.path.basename(f)}: same text as {by_hash[h]} above]")
            else:
                by_hash[h] = f"{title} ({os.path.basename(f)})"
                texts.append(f"--- {os.path.basename(f)} ---\n{text}")
        if not texts:
            # The published crate carries no licence file: the declared licence, and for MIT its
            # standard text with the crate's authors as the copyright holders.
            note = "The crate ships no licence file; it declares the licence above."
            if re.search(r"\bMIT\b", licence):
                note += f"\n\n--- MIT ---\nCopyright (c) {authors(pdir) or 'the ' + name + ' authors'}\n\n{MIT_TEMPLATE}"
            texts.append(note)
        if "MPL" in licence:
            texts.append(f"Source code of this MPL-2.0 component: https://crates.io/crates/{name}/{version}"
                         + (f" and {repo}" if repo else ""))
        parts.append(section(title, licence or "not declared", texts, source))
    write(out, f"{package} (Rust; {len(parts)} components)", parts)


def registry_dir(cargo_home, name, version):
    root = os.path.join(cargo_home, "registry", "src")
    for index in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        d = os.path.join(root, index, f"{name}-{version}")
        if os.path.isfile(os.path.join(d, "Cargo.toml")):
            return d
    return None


def git_checkout(cargo_home, name, url):
    """A git dependency's package directory in cargo's checkouts, and the checkout root to walk up to."""
    sha = url.rsplit("#", 1)[-1][:7]
    root = os.path.join(cargo_home, "git", "checkouts")
    for repo in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        co = os.path.join(root, repo, sha)
        if not os.path.isdir(co):
            continue
        for d, dirs, files in os.walk(co):
            dirs[:] = [x for x in dirs if x not in (".git", "target")]
            if "Cargo.toml" in files and re.search(rf'^name\s*=\s*"{re.escape(name)}"', read(os.path.join(d, "Cargo.toml")), re.M):
                return d, co
    return None, None


def authors(pdir):
    m = re.search(r"^authors\s*=\s*\[([^\]]*)\]", read(os.path.join(pdir, "Cargo.toml")), re.M)
    return ", ".join(a.split("<")[0].strip() for a in re.findall(r'"([^"]+)"', m.group(1))) if m else ""


def licence_files(pdir, walk_up_to):
    """Licence files at the top of a package, or (a workspace member without its own) at the
    nearest ancestor up to `walk_up_to` (the workspace or checkout root)."""
    d = pdir
    while True:
        found = sorted(os.path.join(d, f) for f in os.listdir(d) if LICENSE_FILE.match(f) and os.path.isfile(os.path.join(d, f)))
        if found or not walk_up_to or os.path.realpath(d) == os.path.realpath(walk_up_to) or d == os.path.dirname(d):
            return found
        d = os.path.dirname(d)


def write(out, what, parts):
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"THIRD-PARTY SOFTWARE NOTICES: {what}\n\n")
        f.write("This program includes the third-party software listed below, used under the licences shown.\n\n")
        f.writelines(parts)


def merge(out, inputs):
    with open(out, "w", encoding="utf-8") as f:
        f.write("OpenHydra: third-party software notices\n\n"
                "OpenHydra is licensed under the Apache License, Version 2.0. It ships with the third-party\n"
                "software below, each part under its own licence.\n\n")
        for p in inputs:
            f.write(read(p).rstrip("\n") + "\n\n")


def main(argv):
    if len(argv) >= 3 and argv[0] == "llamacpp":
        return llamacpp(argv[1], argv[2])
    if len(argv) >= 4 and argv[0] == "cargo":
        opts = {"--target": None, "--features": None, "--own-root": None}
        rest, no_default, i = argv[4:], False, 0
        while i < len(rest):
            if rest[i] == "--no-default-features":
                no_default, i = True, i + 1
            elif rest[i] in opts and i + 1 < len(rest):
                opts[rest[i]], i = rest[i + 1], i + 2
            else:
                die(f"unknown option {rest[i]}")
        return cargo(argv[1], argv[2], argv[3], opts["--target"], opts["--features"], no_default, opts["--own-root"])
    if len(argv) >= 3 and argv[0] == "merge":
        return merge(argv[1], argv[2:])
    die(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
