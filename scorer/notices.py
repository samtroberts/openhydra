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
                   [--own-root DIR] [--locked]    a Rust binary: every crate it links (cargo tree)
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
    ("subprocess.h (sheredom)", "Unlicense", "vendor/sheredom/subprocess.h", ("This is free and unencumbered", "*/")),
    ("stb_image", "MIT or public domain", "vendor/stb/stb_image.h", ("This software is available under 2 licenses", "*/")),
    ("miniaudio", "MIT-0 or public domain", "vendor/miniaudio/miniaudio.h", ("This software is available as a choice", "*/")),
    # Third-party code inside llama.cpp's own sources, outside vendor/.
    ("llamafile sgemm (Mozilla Foundation)", "MIT", "ggml/src/ggml-cpu/llamafile/sgemm.cpp", ("// Copyright 2024 Mozilla Foundation", "// SOFTWARE.")),
    ("YaRN rope scaling (Jeffrey Quesnelle and Bowen Peng)", "MIT", "ggml/src/ggml-cpu/ops.cpp", ("mit-line", "Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng")),
    # Code bundled inside nlohmann/json's single header beyond its own licence file.
    ("Grisu2 number formatting (Florian Loitsch, in nlohmann/json)", "MIT", "vendor/nlohmann/json.hpp", ("mit-line", "Copyright (c) 2009 Florian Loitsch")),
    ("UTF-8 decoder (Björn Hoehrmann, in nlohmann/json)", "MIT", "vendor/nlohmann/json.hpp", ("mit-line", "Copyright (c) 2008-2009 Bjoern Hoehrmann")),
    ("Abseil integer_sequence (The Abseil Authors, in nlohmann/json)", "Apache-2.0", "@scorer/licenses/LICENSE-apache-2.0", "file"),
    ("Hedley (Evan Nemerson, in nlohmann/json)", "CC0-1.0", "vendor/nlohmann/json.hpp", ("mit-line", "2016 - 2021 Evan Nemerson")),
    # llamacpp-ollama-compat.patch applies the Qwen3.5 fixes of Ollama's compatibility layer.
    ("Ollama (Qwen3.5 GGUF compatibility, in llamacpp-ollama-compat.patch)", "MIT", "@scorer/licenses/LICENSE-ollama", "file"),
]
# Vendored directories the list above covers.
LLAMACPP_VENDOR = {"cpp-httplib", "hash", "miniaudio", "nlohmann", "sheredom", "stb"}

# Where llama-server's compiled sources live, and the backend directories (and their public headers)
# the scorer build leaves out: the copyright scan covers everything else.
LLAMACPP_SOURCES = ["ggml/include", "ggml/src", "src", "common", "tools/server", "tools/mtmd", "vendor"]
LLAMACPP_NOT_BUILT = {"ggml-sycl", "ggml-openvino", "ggml-cann", "ggml-cuda", "ggml-hip", "ggml-musa", "ggml-vulkan",
                      "ggml-opencl", "ggml-webgpu", "ggml-zdnn", "ggml-zendnn", "ggml-rpc", "ggml-virtgpu", "ggml-hexagon",
                      "ggml-et", "kleidiai", "spacemit"}
SOURCE_EXT = (".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh", ".metal", ".m", ".mm", ".comp", ".inc")
# A line that names a copyright holder (a year after "Copyright", or an SPDX copyright tag).
COPYRIGHT_LINE = re.compile(r"(?i)copyright\s*(?:\(c\)|©)?\s*\d{4}|SPDX-FileCopyrightText")
# Every copyright holder found in llama-server's sources, and the component above that covers it.
# A holder not listed here fails the build: new third-party code whose licence nobody looked at.
LLAMACPP_HOLDERS = [
    ("The ggml authors", "llama.cpp / ggml"),
    ("Georgi Gerganov", "llama.cpp / ggml"),
    ("Mozilla Foundation", "llamafile sgemm"),
    ("Jeffrey Quesnelle", "YaRN rope scaling"),
    ("Niels Lohmann", "nlohmann/json"),
    ("Florian Loitsch", "Grisu2"),
    ("Hoehrmann", "UTF-8 decoder"),
    ("Evan Nemerson", "Hedley"),
    ("Abseil Authors", "Abseil"),
    ("Yann Collet", "xxHash"),
    ("Sean Barrett", "stb_image"),
    ("David Reid", "miniaudio"),
    ("Yuji Hirose", "cpp-httplib"),
]

# Path dependencies under --own-root are our own code, except third-party crates vendored in these.
VENDORED_DIRS = {"patched-deps", "vendor", "third_party", "third-party"}

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


BSD2_TEMPLATE = """Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and
   the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
   and the following disclaimer in the documentation and/or other materials provided with the
   distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""

BSD3_TEMPLATE = BSD2_TEMPLATE.replace(
    "   distribution.\n",
    "   distribution.\n3. Neither the name of the copyright holder nor the names of its contributors may be used\n"
    "   to endorse or promote products derived from this software without specific prior written permission.\n")

ISC_TEMPLATE = """Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee
is hereby granted, provided that the above copyright notice and this permission notice appear in all
copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE
INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE
FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE."""

ZLIB_TEMPLATE = """This software is provided 'as-is', without any express or implied warranty. In no event will the
authors be held liable for any damages arising from the use of this software.

Permission is granted to anyone to use this software for any purpose, including commercial
applications, and to alter it and redistribute it freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the
   original software. If you use this software in a product, an acknowledgment in the product
   documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the
   original software.
3. This notice may not be removed or altered from any source distribution."""

# A crate shipping no licence file: the standard text of the first licence it declares that we have one
# for (copyright: its authors). Anything else needs an override, `licenses/crates/<name>.txt`, or the
# build stops: shipping a component without the text its licence requires isn't an option.
LICENCE_TEMPLATES = [
    (r"\bMIT\b", "MIT", lambda: MIT_TEMPLATE),
    (r"\bApache-2\.0\b", "Apache-2.0", lambda: read(os.path.join(HERE, "licenses", "LICENSE-apache-2.0"))),
    (r"\bBSD-3-Clause\b", "BSD-3-Clause", lambda: BSD3_TEMPLATE),
    (r"\bBSD-2-Clause\b", "BSD-2-Clause", lambda: BSD2_TEMPLATE),
    (r"\bISC\b", "ISC", lambda: ISC_TEMPLATE),
    (r"\bZlib\b", "Zlib", lambda: ZLIB_TEMPLATE),
    # In executable form MPL-2.0 asks that recipients be told the licence and where the source is (the
    # source link follows for every MPL component).
    (r"\bMPL-2\.0\b", "MPL-2.0", lambda: "This component is subject to the Mozilla Public License, v. 2.0. "
                                            "A copy of the licence is at https://mozilla.org/MPL/2.0/."),
]
# Licences that ask for no notice: a component under one of these needs none shipped.
NO_NOTICE = re.compile(r"\b(Unlicense|CC0-1\.0|0BSD|MIT-0)\b")

HERE = os.path.dirname(os.path.abspath(__file__))

# Directories inside a crate whose licence files aren't for code the binary contains.
NOT_SHIPPED_DIRS = {"tests", "test", "examples", "benches", "target", ".git", "docs", "doc", "fuzz", "ci", ".github"}


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


def built_sources(src):
    """llama-server's source files: [`LLAMACPP_SOURCES`], less the backends the build leaves out."""
    for root in LLAMACPP_SOURCES:
        for d, dirs, files in os.walk(os.path.join(src, root)):
            dirs[:] = [x for x in dirs if x not in LLAMACPP_NOT_BUILT]
            for f in files:
                if f.endswith(SOURCE_EXT) and os.path.splitext(f)[0] not in LLAMACPP_NOT_BUILT:
                    yield os.path.join(d, f)


def unknown_holders(src):
    """Copyright lines in llama-server's sources whose holder [`LLAMACPP_HOLDERS`] doesn't map."""
    unknown = {}
    for path in built_sources(src):
        for line in read(path).splitlines():
            if COPYRIGHT_LINE.search(line) and not any(h in line for h, _ in LLAMACPP_HOLDERS):
                unknown.setdefault(line.strip(), os.path.relpath(path, src))
    return unknown


def llamacpp(src, out):
    vendored = {d for d in os.listdir(os.path.join(src, "vendor")) if os.path.isdir(os.path.join(src, "vendor", d))}
    if vendored - LLAMACPP_VENDOR:
        die(f"unknown vendored code in llama.cpp: {sorted(vendored - LLAMACPP_VENDOR)} (add its licence to LLAMACPP)")
    unknown = unknown_holders(src)
    if unknown:
        listing = "\n".join(f"  {path}: {line}" for line, path in sorted(unknown.items()))
        die(f"copyright holders in llama.cpp's sources with no notice here (add them to LLAMACPP):\n{listing}")
    parts = []
    for name, licence, rel, how in LLAMACPP:
        # "@scorer/…" entries are licences kept in this repo (code our patches derive from).
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel[len("@scorer/"):]) if rel.startswith("@scorer/") else os.path.join(src, rel)
        if not os.path.isfile(path):
            die(f"{rel} is missing (llama.cpp changed: review LLAMACPP)")
        text = read(path)
        if how != "file" and how[0] == "mit-line":
            # A one-line notice in the source: that line, and (for MIT) the licence's standard text.
            line = next((l for l in text.splitlines() if how[1] in l), None)
            if line is None:
                die(f"no notice naming {how[1]!r} in {rel}")
            line = line.strip().lstrip("/*# ").strip()
            text = line if licence != "MIT" else f"{line}\n\n{MIT_TEMPLATE}"
        elif how != "file":
            start, end = how
            i = text.find(start)
            j = text.find(end, i + len(start)) if i >= 0 else -1
            if i < 0 or j < 0:
                die(f"no licence block in {rel}")
            # A block comment's markers aren't part of the text; a line-comment block's end line is.
            text = text[i + (len(start) if start == "/*" else 0) : j + (0 if end == "*/" else len(end))]
            if start.startswith("//"):
                text = "\n".join(l.strip()[2:].strip() if l.strip().startswith("//") else l for l in text.splitlines())
        parts.append(section(name, licence, [text]))
    write(out, "openhydra-scorer (llama.cpp's llama-server with OpenHydra's prompt-scoring patch)", parts)


def cargo(manifest, package, out, target=None, features=None, no_default=False, own_root=None, locked=False):
    # `cargo tree` resolves for one target and needs only the crates that build uses (offline-capable),
    # unlike `cargo metadata`, which wants every platform's dependencies.
    # Build dependencies too: a build script can compile C code into the binary (e.g. a vendored
    # library built by a `-sys` crate's build script).
    cmd = ["cargo", "tree", "--manifest-path", manifest, "-p", package, "-e", "normal,build", "--prefix", "none",
           "-f", "{p}\t{l}\t{r}"]
    if target:
        cmd += ["--target", target]
    if features:
        cmd += ["--features", features]
    if no_default:
        cmd += ["--no-default-features"]
    if locked:
        cmd += ["--locked"]  # exactly the lockfile the build used; a stale one fails
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
        if os.path.isabs(src) or re.match(r"^[A-Za-z]:[\\/]", src):  # a path dependency (cargo prints D:\ on Windows)
            pdir, walk_up = os.path.realpath(src), manifest_dir
            if own_root and (pdir == own_root or pdir.startswith(own_root + os.sep)):
                if not VENDORED_DIRS & set(os.path.relpath(pdir, own_root).split(os.sep)):
                    continue  # our own code
                walk_up = None  # a vendored crate: its own licence, never ours
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
            label = os.path.relpath(f, pdir) if f.startswith(pdir + os.sep) else os.path.basename(f)
            h = hashlib.sha256(text.encode()).hexdigest()
            if h in by_hash:
                texts.append(f"[{label}: same text as {by_hash[h]} above]")
            else:
                by_hash[h] = f"{title} ({label})"
                texts.append(f"--- {label} ---\n{text}")
        if not texts:
            texts.append(no_licence_file_notice(name, licence, pdir))
        bundled = bundled_copyrights(pdir)
        if bundled:
            texts.append("Copyright notices in its bundled C/C++/Metal sources (under the licence above):\n"
                         + "\n".join(bundled))
        if "MPL" in licence:
            texts.append(f"Source code of this MPL-2.0 component: https://crates.io/crates/{name}/{version}"
                         + (f" and {repo}" if repo else ""))
        parts.append(section(title, licence or "not declared", texts, source))
    # The standard library is compiled into every Rust binary but never appears in `cargo tree`.
    parts.append(section("Rust standard library (std, core, alloc and their dependencies)", "MIT OR Apache-2.0",
                         [f"Copyright (c) The Rust Project Developers\n\n{MIT_TEMPLATE}"], "https://github.com/rust-lang/rust"))
    write(out, f"{package} (Rust; {len(parts)} components)", parts)


def no_licence_file_notice(name, licence, pdir):
    """The notice for a crate that ships no licence file (see [`LICENCE_TEMPLATES`])."""
    override = os.path.join(HERE, "licenses", "crates", f"{name}.txt")
    if os.path.isfile(override):
        return read(override)
    note = "The crate ships no licence file; it declares the licence above."
    for pattern, spdx, text in LICENCE_TEMPLATES:
        if re.search(pattern, licence):
            return f"{note}\n\n--- {spdx} ---\nCopyright (c) {authors(pdir) or 'the ' + name + ' authors'}\n\n{text()}"
    if licence and NO_NOTICE.search(licence):
        return note
    die(f"{name} ships no licence file and declares {licence or 'no licence'}: add its text as licenses/crates/{name}.txt")


def bundled_copyrights(pdir):
    """Distinct copyright lines in a crate's non-Rust sources (C, C++, Metal, CUDA it compiles in),
    which can name holders its own licence file doesn't."""
    lines = set()
    for d, dirs, files in os.walk(pdir):
        dirs[:] = [x for x in dirs if x not in NOT_SHIPPED_DIRS]
        for f in files:
            if f.endswith(SOURCE_EXT):
                for line in read(os.path.join(d, f)).splitlines():
                    if COPYRIGHT_LINE.search(line):
                        lines.add(line.strip().lstrip("/*#! ").strip())
    return sorted(lines)


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
    nearest ancestor up to `walk_up_to` (the workspace or checkout root); plus any deeper in the
    package, for code it bundles under another licence (a vendored C library, say)."""
    d = pdir
    while True:
        found = sorted(os.path.join(d, f) for f in os.listdir(d) if LICENSE_FILE.match(f) and os.path.isfile(os.path.join(d, f)))
        if found or not walk_up_to or os.path.realpath(d) == os.path.realpath(walk_up_to) or d == os.path.dirname(d):
            break
        d = os.path.dirname(d)
    for sub, dirs, files in os.walk(pdir):
        dirs[:] = sorted(x for x in dirs if x not in NOT_SHIPPED_DIRS)
        if sub != pdir:
            found += sorted(os.path.join(sub, f) for f in files if LICENSE_FILE.match(f))
    return found


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
        rest, no_default, locked, i = argv[4:], False, False, 0
        while i < len(rest):
            if rest[i] == "--no-default-features":
                no_default, i = True, i + 1
            elif rest[i] == "--locked":
                locked, i = True, i + 1
            elif rest[i] in opts and i + 1 < len(rest):
                opts[rest[i]], i = rest[i + 1], i + 2
            else:
                die(f"unknown option {rest[i]}")
        return cargo(argv[1], argv[2], argv[3], opts["--target"], opts["--features"], no_default, opts["--own-root"], locked)
    if len(argv) >= 3 and argv[0] == "merge":
        return merge(argv[1], argv[2:])
    die(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
