"""SNDlib native ASCII 流量矩阵解析器。

解析单个 SNDlib native 文件文本，返回结构化数据。GÉANT 与 Abilene 同构。
"""
from __future__ import annotations

import re

# NODES 段每行: <id> ( <lon> <lat> )
_NODE_RE = re.compile(r"(\S+)\s*\(\s*(-?[\d.]+)\s+(-?[\d.]+)\s*\)")
# DEMANDS 段每行: <id> ( <src> <dst> ) <ru> <value> UNLIMITED
_DEMAND_RE = re.compile(
    r"\S+\s*\(\s*(\S+)\s+(\S+)\s*\)\s+\d+\s+([\d.eE+-]+)\s+\S+"
)
# META 段 key = value
_META_RE = re.compile(r"(\w+)\s*=\s*(\S+)")


def _extract_block(text: str, name: str) -> str:
    """提取 NAME ( ... ) 段的内容（不含括号行）。

    SNDlib native 的段以行首的 `NAME (` 开始，以独占一行的 `)` 结束。
    段内每条记录可能含 `( ... )`，故不能用第一个 `)` 截断。
    """
    lines = text.splitlines()
    inside = False
    out = []
    for line in lines:
        s = line.strip()
        if not inside:
            if s.startswith(name + " (") or s == name + " (":
                inside = True
            continue
        # 段结束：行首单独的 ')'
        if s == ")":
            break
        out.append(line)
    return "\n".join(out)


def parse_sndlib_native(text: str) -> dict:
    """解析单个 SNDlib native 文件文本。

    返回:
      {
        "network": str, "granularity": str, "timestamp": str, "unit": str,
        "nodes": [(node_id, lon, lat), ...],
        "demands": [(src, dst, value), ...],   # 有向, value 为 float
      }
    """
    # network 名从注释行 # network demandMatrix-<gran>-<ts>-<net> 取
    network = ""
    for line in text.splitlines():
        if line.startswith("# network"):
            # demandMatrix-15min-20050504-1630-geant
            tail = line.split("network", 1)[1].strip()
            parts = tail.split("-")
            if parts:
                network = parts[-1]
            break

    meta_block = _extract_block(text, "META")
    meta = dict(_META_RE.findall(meta_block))

    nodes_block = _extract_block(text, "NODES")
    nodes = []
    for m in _NODE_RE.finditer(nodes_block):
        nid, lon, lat = m.group(1), float(m.group(2)), float(m.group(3))
        nodes.append((nid, lon, lat))

    demands_block = _extract_block(text, "DEMANDS")
    demands = []
    for m in _DEMAND_RE.finditer(demands_block):
        src, dst, val = m.group(1), m.group(2), float(m.group(3))
        demands.append((src, dst, val))

    return {
        "network": network,
        "granularity": meta.get("granularity", ""),
        "timestamp": meta.get("time", ""),
        "unit": meta.get("unit", ""),
        "nodes": nodes,
        "demands": demands,
    }


if __name__ == "__main__":
    import subprocess
    out = subprocess.run(
        ["tar", "-xOzf",
         "directed-geant-uhlig-15min-over-4months-ALL-native.tgz",
         "directed-geant-uhlig-15min-over-4months-ALL-native/"
         "demandMatrix-geant-uhlig-15min-20050504-1630.txt"],
        capture_output=True, text=True,
    ).stdout
    r = parse_sndlib_native(out)
    print("network:", r["network"], "| nodes:", len(r["nodes"]),
          "| demands:", len(r["demands"]), "| unit:", r["unit"],
          "| ts:", r["timestamp"])
    print("first demand:", r["demands"][0], "| last:", r["demands"][-1])
