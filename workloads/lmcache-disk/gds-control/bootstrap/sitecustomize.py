"""Activate the LMCache GDS backend policy when its mode environment is set."""

from lmcache_gds_backend_adapter import bootstrap_from_env


bootstrap_from_env()

from lmcache_gds_async_prefetch_adapter import bootstrap_from_env as bootstrap_async

bootstrap_async()

from lmcache_kv_reclaim_adapter import (
    bootstrap_from_env as bootstrap_kv_reclaim,
)

bootstrap_kv_reclaim()
