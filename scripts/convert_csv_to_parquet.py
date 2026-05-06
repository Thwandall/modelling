#!/usr/bin/env python3
"""Convert a large CSV/CSV.GZ to Parquet in chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=25000)
    parser.add_argument("--compression", default="zstd")
    return parser.parse_args()


def downcast_chunk(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    rows = 0
    chunks = 0
    try:
        for chunk in pd.read_csv(args.input, chunksize=args.chunksize, low_memory=False):
            chunk = downcast_chunk(chunk)
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(args.out, table.schema, compression=args.compression)
            else:
                table = table.cast(writer.schema, safe=False)
            writer.write_table(table)
            rows += len(chunk)
            chunks += 1
            if chunks % 5 == 0:
                print(f"converted_chunks={chunks} rows={rows}", flush=True)
    finally:
        if writer is not None:
            writer.close()
    print(json.dumps({"input": str(args.input), "out": str(args.out), "rows": rows, "chunks": chunks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
