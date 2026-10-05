from datetime import date, timedelta

from common.clients import bq_client
from common.clients import bq_job_config
from common.config import load_sources


def check_gdelt_access():
    table = load_sources()["gdelt"]["events_table"]
    day = date.today() - timedelta(days=3)
    query = f"""
        SELECT SQLDATE, Actor1Name, Actor2Name, EventCode, GoldsteinScale, AvgTone
        FROM `{table}`
        WHERE _PARTITIONTIME = TIMESTAMP('{day}')
        LIMIT 5
    """

    client = bq_client()
    print("logged in as", client._credentials.service_account_email)

    dry = client.query(query, job_config=bq_job_config(dry_run=True))
    print(f"dry run ok, query would scan {dry.total_bytes_processed / 1e6:.2f} mb")

    rows = client.query(query, job_config=bq_job_config()).to_dataframe()
    print(rows.to_string())


if __name__ == "__main__":
    check_gdelt_access()
