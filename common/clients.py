from dotenv import load_dotenv
from google.cloud import bigquery

from common.config import ROOT, load_sources

load_dotenv(ROOT / ".env")


def bq_client():
    return bigquery.Client(project=load_sources()["gdelt"]["project"])


def bq_job_config(dry_run=False):
    return bigquery.QueryJobConfig(
        dry_run=dry_run,
        use_query_cache=False,
        maximum_bytes_billed=load_sources()["gdelt"]["max_bytes_billed"],
    )
