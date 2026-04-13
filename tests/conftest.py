import os
import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def tmp_db_path(tmp_path):
    return str(tmp_path / "test_kortex.db")


@pytest.fixture
def kortex_db(tmp_db_path):
    from kortex.db import KortexDB

    db = KortexDB(tmp_db_path)
    yield db
    db.close()


@pytest.fixture
def kortex_config(tmp_db_path):
    from kortex.config import KortexConfig

    return KortexConfig(db_path=tmp_db_path)


@pytest.fixture
def ingestor(kortex_db):
    from kortex.ingest import Ingestor

    return Ingestor(kortex_db)


@pytest.fixture
def recall(kortex_db, kortex_config):
    from kortex.recall import Recall

    return Recall(kortex_db, kortex_config)
