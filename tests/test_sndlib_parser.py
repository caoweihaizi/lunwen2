"""SNDlib native 解析器单元测试。用小型手写文本，不依赖真实大文件。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.sndlib_parser import parse_sndlib_native

FIXTURE = """?SNDlib native format; type: network; version: 1.0
# network demandMatrix-15min-20050504-1630-testnet

# META SECTION
META (
  granularity  = 15min
  time  = 20050504-1630
  unit  = MBITPERSEC
  origin  = test
)

NODES (
  A ( 10.0 20.0 )
  B ( -5.5 30.25 )
  C ( 100.0 -80.0 )
)

LINKS (
)

DEMANDS (
  A_B ( A B ) 1 12.5 UNLIMITED
  A_C ( A C ) 1 0.0 UNLIMITED
  B_A ( B A ) 1 7.25 UNLIMITED
  B_C ( B C ) 1 99.9 UNLIMITED
  C_A ( C A ) 1 3.0 UNLIMITED
  C_B ( C B ) 1 0.1119 UNLIMITED
)
"""


def test_meta_fields():
    r = parse_sndlib_native(FIXTURE)
    assert r["network"] == "testnet"
    assert r["granularity"] == "15min"
    assert r["timestamp"] == "20050504-1630"
    assert r["unit"] == "MBITPERSEC"


def test_nodes():
    r = parse_sndlib_native(FIXTURE)
    assert r["nodes"] == [
        ("A", 10.0, 20.0),
        ("B", -5.5, 30.25),
        ("C", 100.0, -80.0),
    ]


def test_demands_directed():
    r = parse_sndlib_native(FIXTURE)
    # 含反向 demand（有向性）
    assert len(r["demands"]) == 6
    assert ("A", "B", 12.5) in r["demands"]
    assert ("B", "A", 7.25) in r["demands"]
    # value 为 float，含 0 与小数
    assert ("A", "C", 0.0) in r["demands"]
    assert ("B", "C", 99.9) in r["demands"]
    assert ("C", "B", 0.1119) in r["demands"]
    assert all(isinstance(v, float) for _, _, v in r["demands"])


def test_demands_count_n_squared():
    """3 节点 → 3*2=6 有向需求。"""
    r = parse_sndlib_native(FIXTURE)
    assert len(r["demands"]) == 3 * 2
