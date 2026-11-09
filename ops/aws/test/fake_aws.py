#!/usr/bin/env python3
"""A stand-in for the `aws` CLI, for testing launch-cheapest-spot.sh without AWS.

It answers the calls the script makes from a scenario (JSON in $FAKE_AWS_SCENARIO), appends every
call to $FAKE_AWS_LOG, and never contacts AWS. Unknown calls fail loudly, so a script change that
makes a new call shows up in the tests.
"""
import json, os, sys, time

args = sys.argv[1:]
sc = json.load(open(os.environ["FAKE_AWS_SCENARIO"]))
state_dir = os.environ["FAKE_AWS_STATE"]
with open(os.environ["FAKE_AWS_LOG"], "a") as log:
    log.write(" ".join(args) + "\n")


def opt(name, default=None):
    return args[args.index(name) + 1] if name in args else default


def out(text, code=0):
    if text is not None:
        print(text)
    sys.exit(code)


def err(code, op):
    sys.stderr.write(f"\nAn error occurred ({code}) when calling the {op} operation: test\n")
    sys.exit(254)


def counter(name):
    path = os.path.join(state_dir, name)
    n = int(open(path).read()) if os.path.exists(path) else 0
    open(path, "w").write(str(n + 1))
    return n + 1


words = [a for a in args if not a.startswith("--")]
service, op = words[0], words[1] if len(words) > 1 else ""
joined = " ".join(args)
dry = "--dry-run" in args

if service == "sts":
    out("{}")
if service != "ec2":
    err("UnknownService", service)

if op == "describe-regions":
    out("\t".join(sc["regions"]))
if op == "describe-spot-price-history":
    r = opt("--region")
    out("\n".join("\t".join(p) for p in sc["prices"].get(r, [])))
if op == "describe-instances":
    if "client-token" in joined:
        n = counter("token-lookups")
        found = sc.get("token_found")
        out(found if found and n >= sc.get("token_found_after", 1) else "None")
    if "tag:Name" in joined:
        if opt("--region") in sc.get("existing_fail", []):
            err("RequestLimitExceeded", "DescribeInstances")
        out("\n".join("\t".join(i) for i in sc.get("existing", {}).get(opt("--region"), [])))
    out("203.0.113.9")
if op == "describe-images":
    q = opt("--query", "")
    if "Architecture,State" in q:
        out(f"{sc.get('source_arch', 'arm64')}\tavailable")
    if "openhydra-source-image" in joined:
        out(sc.get("existing_copy") or "None")
    if "al2023" in joined:
        out("ami-standin")
    if "RootDeviceName" in q:
        out(sc.get("root", "/dev/xvda"))
    if "BlockDeviceMappings[?Ebs]" in q:
        # The source image (in its own region) can have its own volumes.
        if opt("--region") == sc.get("source_region", "r0") and "source_bdm" in sc:
            out("\n".join("\t".join(m) for m in sc["source_bdm"]))
        out("\n".join("\t".join(m) for m in sc.get("bdm", [["/dev/xvda", "8", "True"]])))
    if "SnapshotId" in q:
        out("snap-1")
    if "State" in q:
        states = sc.get("copy_states", ["available"])
        n = counter("copy-state")
        out(states[min(n, len(states)) - 1])
    out("ami-found")
if op == "create-security-group":
    if dry:
        err("DryRunOperation", "CreateSecurityGroup")
    out("sg-new")
if op == "describe-vpcs":
    out("vpc-1")
if op == "describe-key-pairs":
    out(None, 0 if sc.get("key_exists", True) else 254)
if op == "import-key-pair":
    out("{}")
if op == "describe-security-groups":
    if "Values=default" in joined:
        out("sg-default")
    if "group-name" in joined:
        out(sc.get("sg") or "None")
    out(sc.get("sg_managed") or "None")
if op == "describe-network-interfaces":
    out(str(sc.get("sg_in_use", 0)))
if op == "describe-security-group-rules":
    out("\n".join("\t".join(r) for r in sc.get("sg_rules", [])))
if op in ("revoke-security-group-ingress", "authorize-security-group-ingress"):
    out("{}")
if op == "describe-instance-types":
    out(sc.get("type_arch", "arm64"))
if op == "copy-image":
    if sc.get("copy_fails"):
        err("InvalidRequest", "CopyImage")
    out("ami-copy")
if op == "describe-subnets":
    out(f"subnet-1\t{sc.get('subnet_v6', 0)}")
if op == "run-instances":
    if dry:
        err(sc.get("dryrun_code", "DryRunOperation"), "RunInstances")
    time.sleep(sc.get("launch_sleep", 0))
    res = sc.get("launch", "i-fake")
    if res.startswith("i-"):
        out(res)
    err(res, "RunInstances")
if op == "wait":
    out(None)
if op == "deregister-image":
    if "--generate-cli-skeleton" in args:
        out('{"ImageId": "", "DeleteAssociatedSnapshots": true, "DryRun": true}' if sc.get("cli_new", True)
            else '{"ImageId": "", "DryRun": true}')
    if "--delete-associated-snapshots" in args:
        # The query lists snapshots that weren't deleted.
        out("snap-1\tskipped" if sc.get("snapshot_skipped") else "")
    out("{}")
if op in ("delete-security-group", "delete-snapshot"):
    out("{}")
err("UnknownOperation", op)
