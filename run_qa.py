import duckdb
import pandas as pd
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Run QA Checks on DuckDB Tables")
    parser.add_argument("--roster-table", default="roster_naive_NATIONAL", help="Name of the roster table to audit (default: roster_naive_NATIONAL)")
    args = parser.parse_args()

    roster_table = args.roster_table
    os.makedirs("metrics", exist_ok=True)
    out_file_path = f"metrics/qa_{roster_table}.txt"

    con = duckdb.connect("pharma_commercial.duckdb", read_only=True)

    with open('sql/04_qa_checks.sql', 'r') as f:
        sql_content = f.read()

    # Parameterize roster table
    sql_formatted = sql_content.replace("{roster_table}", roster_table)
    queries = [q.strip() for q in sql_formatted.split(';') if q.strip()]

    output_lines = []

    check_titles = [
        "1. Uniqueness & Duplicate Check",
        "2. Valid ZIP Match Rate",
        "3. ZCTA Match Rate (Primary Targets only)",
        "4. Staging Drug Audit",
        "5. Decile Sanity Check",
        "6. Unassigned Targets",
        "6b. Primary Target Count Integrity Check",
        "6c. State Representation Check",
        "6d. ZIP Integrity Check",
        "6e. Workload Band Integrity Check"
    ]

    for i, query in enumerate(queries):
        title = check_titles[i] if i < len(check_titles) else f"Check {i+1}"
        header = f"--- QA Check {title} ---"
        print(header)
        output_lines.append(header)
        try:
            df = con.execute(query).df()
            df_str = df.to_string()
            print(df_str)
            output_lines.append(df_str)

            # Loud assertions for 6b and 6c
            if "status_6b" in df.columns:
                status = df["status_6b"].iloc[0]
                if status != "PASS":
                    msg = f"CRITICAL QA 6b FAILURE: {status} in table {roster_table}"
                    print(msg, file=sys.stderr)
                    output_lines.append(msg)
                    sys.exit(msg)
            if "status_6c" in df.columns:
                status = df["status_6c"].iloc[0]
                if status != "PASS":
                    msg = f"CRITICAL QA 6c FAILURE: {status} in table {roster_table}"
                    print(msg, file=sys.stderr)
                    output_lines.append(msg)
                    sys.exit(msg)

        except Exception as e:
            err_msg = f"Error executing query: {e}"
            print(err_msg)
            output_lines.append(err_msg)
        print("\n")
        output_lines.append("\n")

    con.close()

    with open(out_file_path, "w") as f:
        f.write("\n".join(output_lines))
    print(f"QA audit results written to {out_file_path}")

if __name__ == "__main__":
    main()
