"""根据论文数据清单恢复交易所月档、指数日行情及分合约面板。"""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIELDS = {"合约代码": "contract", "今开盘": "open", "今收盘": "close",
          "今结算": "settle", "前结算": "prev_settle", "成交量": "volume", "持仓量": "oi"}


def obtain(record):
    """只让与研究快照校验值一致的原档进入计算。"""
    path = DATA / record["file"]
    if path.exists():
        content = path.read_bytes()
    else:
        with requests.Session() as session:
            retries = Retry(total=3, backoff_factor=0.7, status_forcelist=(429, 500, 502, 503, 504))
            for scheme in ("http://", "https://"):
                session.mount(scheme, HTTPAdapter(max_retries=retries))
            response = session.get(record["url"], timeout=(10, 40))
            response.raise_for_status()
            content = response.content
    if sha256(content).hexdigest() != record["sha256"]:
        raise ValueError(f"{path.name} 与 data_sources.json 记载的论文快照不同，请核对官方数据版本。")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(content)
        temporary.replace(path)
    return path


def read_month(path):
    """按原始合约代码筛选日统计，保留开盘与结算两个价格基准。"""
    frames = []
    with ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith("_1.csv"):
                continue
            with archive.open(name) as source:
                frame = pd.read_csv(source, encoding="gb18030")
            frame.columns = frame.columns.str.strip()
            frame = frame.rename(columns=FIELDS)[list(FIELDS.values())].copy()
            frame["contract"] = frame.contract.str.strip()
            frame = frame.loc[frame.contract.str.fullmatch(r"IF\d{4}", na=False)].copy()
            frame.insert(0, "date", pd.to_datetime(Path(name).name[:8], format="%Y%m%d"))
            for field in list(FIELDS.values())[1:]:
                frame[field] = pd.to_numeric(frame[field], errors="raise")
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def prepare():
    """按来源清单获取数据并按交易日和交割月排序。"""
    records = json.loads((ROOT / "data_sources.json").read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(obtain, records))
    panel = pd.concat([read_month(p) for p in paths if p.suffix == ".zip"], ignore_index=True)
    if (len(panel), panel.date.nunique()) != (7768, 1942) or panel.duplicated(["date", "contract"]).any():
        raise ValueError("IF面板的交易日、行数或合约唯一性与论文样本不符。")
    panel.sort_values(["date", "contract"]).to_csv(DATA / "IF_panel.csv.gz", index=False)
    print(f"Verified {len(records)} source files; {len(panel)} IF contract-day records.", flush=True)


if __name__ == "__main__":
    prepare()
