"""Federated recipe-cloud tools (L5 layer)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from gemma42.cloud import CloudClient
from gemma42.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from gemma42.state import AgentState


class CloudSearchTool(Tool):
    name = "cloud_search"
    description = (
        "Search the federated recipe cloud (or its local cache) for "
        "extractor recipes or codebooks. Args: domain, fingerprint, "
        "keyword, kind ('recipe' or 'codebook')."
    )
    args_schema = {
        "domain":      {"type": "string"},
        "fingerprint": {"type": "string"},
        "keyword":     {"type": "string"},
        "kind":        {"type": "string", "enum": ["recipe", "codebook"]},
        "limit":       {"type": "integer", "default": 10},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        client = CloudClient()
        items = client.search(
            domain=args.get("domain") or None,
            fingerprint=args.get("fingerprint") or None,
            keyword=args.get("keyword") or None,
            kind=args.get("kind") or None,
            limit=int(args.get("limit") or 10),
        )
        if not items:
            return ToolResult(output="(no matches)")
        out = [f"found {len(items)} artifact(s):"]
        for a in items:
            out.append(
                f"  - id={a.id}  kind={a.kind}  domain={a.domain}  name={a.name}"
                f"  fp={a.fingerprint[:8]}"
            )
        return ToolResult(output="\n".join(out),
                           artifact=[a.to_dict() for a in items])


class CloudPushTool(Tool):
    name = "cloud_push"
    description = (
        "Publish a successful recipe or codebook to the federated cloud. "
        "The cloud is opt-in: if no GEMMA42_CLOUD_URL is set, the artifact "
        "is still kept locally in your ~/.gemma42/cloud_cache/. Specs are "
        "anonymous — your `publisher` id is a random 16-hex token created "
        "automatically. Args: kind ('recipe'|'codebook'), domain, name, "
        "fingerprint, spec, stats (dict), license ('CC0-1.0' default)."
    )
    args_schema = {
        "kind":        {"type": "string", "enum": ["recipe", "codebook"]},
        "domain":      {"type": "string"},
        "name":        {"type": "string"},
        "fingerprint": {"type": "string"},
        "spec":        {"type": "object"},
        "stats":       {"type": "object"},
        "license":     {"type": "string", "default": "CC0-1.0"},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        from gemma42.cloud import CloudArtifact, make_artifact_id

        kind = args.get("kind")
        if kind not in ("recipe", "codebook"):
            return ToolResult(output="ERROR: 'kind' must be 'recipe' or 'codebook'",
                               error=True)
        spec = args.get("spec") or {}
        if not isinstance(spec, dict) or not spec:
            return ToolResult(output="ERROR: 'spec' (non-empty object) required",
                               error=True)
        payload = {
            "kind": kind,
            "domain": args.get("domain") or "",
            "name": args.get("name") or kind,
            "fingerprint": args.get("fingerprint") or "",
            "spec": spec,
            "stats": args.get("stats") or {},
            "license": args.get("license") or "CC0-1.0",
        }
        art = CloudArtifact(
            id=make_artifact_id(payload),
            kind=payload["kind"], domain=payload["domain"],
            fingerprint=payload["fingerprint"], name=payload["name"],
            spec=payload["spec"], stats=payload["stats"],
            license=payload["license"],
        )
        client = CloudClient()
        result = client.push(art)
        return ToolResult(output=json.dumps(result, indent=2, ensure_ascii=False))


class CloudPullTool(Tool):
    name = "cloud_pull"
    description = "Fetch a specific artifact by id from the cloud (or local cache)."
    args_schema = {"artifact_id": {"type": "string"}}

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        aid = args.get("artifact_id") or ""
        if not aid:
            return ToolResult(output="ERROR: 'artifact_id' required", error=True)
        client = CloudClient()
        art = client.pull(aid)
        if art is None:
            return ToolResult(output=f"not found: {aid}", error=True)
        return ToolResult(output=json.dumps(art.to_dict(), indent=2,
                                              ensure_ascii=False),
                           artifact=art.to_dict())
