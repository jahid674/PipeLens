import os
import sys
import pandas as pd

def csv_to_parquet_chunked(
    csv_path: str,
    parquet_path: str,
    chunksize: int = 200_000,
    compression: str = "snappy",
):
    """
    Convert a large CSV to a single Parquet file without loading everything into memory.
    Requires: pip install pyarrow
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("Missing dependency: pyarrow. Install with: pip install pyarrow")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    writer = None
    total_rows = 0

    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
        # optional: enforce numeric columns if needed
        # chunk = chunk.apply(pd.to_numeric, errors="ignore")

        table = pa.Table.from_pandas(chunk, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(
                parquet_path,
                table.schema,
                compression=compression
            )

        writer.write_table(table)
        total_rows += len(chunk)
        print(f"[chunk {i}] wrote {len(chunk)} rows (total={total_rows})")

    if writer is not None:
        writer.close()

    print(f"\nDONE ✅ Parquet saved at: {parquet_path}")
    return parquet_path


def demo_read(parquet_path: str):
    """Quick demo: read a few columns fast."""
    df = pd.read_parquet(
        parquet_path,
        columns=["sampling", "missing_value", "normalization", "utility_sp"]
    )
    print("\nSample rows:")
    print(df.head(10))
    print("\nDescribe utility_sp:")
    print(df["utility_sp"].describe())


if __name__ == "__main__":
    # ---- edit these paths if needed ----
    csv_path = "synthetic_large_pipeline_all_combinations_1based.csv"
    parquet_path = "synthetic_large_pipeline_all_combinations_1based.parquet"

    # You can also pass args:
    # python csv_to_parquet_chunked.py input.csv output.parquet
    if len(sys.argv) >= 3:
        csv_path = sys.argv[1]
        parquet_path = sys.argv[2]

    out = csv_to_parquet_chunked(csv_path, parquet_path, chunksize=200_000)
    demo_read(out)
