import collections
import os
import secrets
from datetime import timedelta
from functools import cache
from typing import Any, List, Optional

from pydantic_settings import BaseSettings

DatabaseSettings = collections.namedtuple(
    "DatabaseSettings", "uri pool_size pool_pre_ping max_overflow"
)


class Settings(BaseSettings):
    tree: Any = None
    allow_anonymous_access: bool = bool(
        int(os.getenv("TILED_ALLOW_ANONYMOUS_ACCESS", False))
    )
    allow_origins: List[str] = [
        item for item in os.getenv("TILED_ALLOW_ORIGINS", "").split() if item
    ]
    authenticator: Any = None
    # These 'single user' settings are only applicable if authenticator is None.
    single_user_api_key: str = os.getenv(
        "TILED_SINGLE_USER_API_KEY", secrets.token_hex(32)
    )
    single_user_api_key_generated: bool = not (
        "TILED_SINGLE_USER_API_KEY" in os.environ
    )
    # The TILED_SERVER_SECRET_KEYS may be a single key or a ;-separated list of
    # keys to support key rotation. The first key will be used for encryption. Each
    # key will be tried in turn for decryption.
    secret_keys: List[str] = os.getenv(
        "TILED_SERVER_SECRET_KEYS", secrets.token_hex(32)
    ).split(";")
    access_token_max_age: timedelta = timedelta(
        seconds=int(os.getenv("TILED_ACCESS_TOKEN_MAX_AGE", 15 * 60))  # 15 minutes
    )
    refresh_token_max_age: timedelta = timedelta(
        seconds=int(
            os.getenv("TILED_REFRESH_TOKEN_MAX_AGE", 7 * 24 * 60 * 60)
        )  # 7 days
    )
    session_max_age: Optional[timedelta] = timedelta(
        seconds=int(os.getenv("TILED_SESSION_MAX_AGE", 365 * 24 * 60 * 60))  # 365 days
    )
    # Put a fairly low limit on the maximum size of one chunk, keeping in mind
    # that data should generally be chunked. When we implement async responses,
    # we can raise this global limit.
    response_bytesize_limit: int = int(
        os.getenv("TILED_RESPONSE_BYTESIZE_LIMIT", 300_000_000)
    )  # 300 MB
    reject_undeclared_specs: bool = bool(
        int(os.getenv("TILED_REJECT_UNDECLARED_SPECS", 0))
    )
    database_uri: Optional[str] = os.getenv("TILED_DATABASE_URI")
    database_init_if_not_exists: bool = int(
        os.getenv("TILED_DATABASE_INIT_IF_NOT_EXISTS", False)
    )
    database_pool_size: Optional[int] = int(os.getenv("TILED_DATABASE_POOL_SIZE", 5))
    database_pool_pre_ping: Optional[bool] = bool(
        int(os.getenv("TILED_DATABASE_POOL_PRE_PING", 1))
    )
    database_max_overflow: Optional[int] = int(
        os.getenv("TILED_DATABASE_MAX_OVERFLOW", 5)
    )
    expose_raw_assets: bool = True

    @property
    def database_settings(self):
        # The point of this alias is to return a hashable cache key for use in
        # the module tiled.authn_database.connection_pool.
        return DatabaseSettings(
            uri=self.database_uri,
            pool_size=self.database_pool_size,
            pool_pre_ping=self.database_pool_pre_ping,
            max_overflow=self.database_max_overflow,
        )


@cache
def get_settings() -> Settings:
    return Settings()
