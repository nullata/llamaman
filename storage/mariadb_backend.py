# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import hashlib
import json
import logging

from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, Text, func
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker

from storage.base import StorageBackend

logger = logging.getLogger("llamaman")

Base = declarative_base()


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------

class InstanceRow(Base):
    __tablename__ = "instances"
    id = Column(String(64), primary_key=True)
    data = Column(Text, default="{}")


class DownloadRow(Base):
    __tablename__ = "downloads"
    id = Column(String(64), primary_key=True)
    data = Column(Text, default="{}")


class PresetRow(Base):
    __tablename__ = "presets"
    model_path = Column(String(768), primary_key=True)
    data = Column(Text, default="{}")


class UserRow(Base):
    __tablename__ = "users"
    username = Column(String(255), primary_key=True)
    password_hash = Column(String(255), nullable=False)


class SettingsRow(Base):
    __tablename__ = "settings"
    key = Column(String(64), primary_key=True)
    data = Column(Text, default="{}")


class ApiKeyRow(Base):
    __tablename__ = "api_keys"
    id = Column(String(32), primary_key=True)
    name = Column(String(255), default="")
    key_hash = Column(String(64), nullable=False)
    prefix = Column(String(16), default="")
    created_at = Column(Integer, default=0)


# ---------------------------------------------------------------------------
# Backend implementation
# ---------------------------------------------------------------------------

class MariaDBBackend(StorageBackend):
    """Stores data in MariaDB/MySQL via SQLAlchemy."""

    def __init__(self, database_url: str):
        self._engine = create_engine(database_url, pool_pre_ping=True)
        Base.metadata.create_all(self._engine)
        self._session_factory = scoped_session(sessionmaker(bind=self._engine))
        logger.info("MariaDB backend connected: %s", database_url.split("@")[-1])

    def _session(self):
        return self._session_factory()

    # -- State --

    def save_state(self, instances: list[dict], downloads: list[dict]) -> None:
        session = self._session()
        try:
            session.query(InstanceRow).delete()
            session.query(DownloadRow).delete()
            for inst in instances:
                session.add(InstanceRow(id=inst["id"], data=json.dumps(inst)))
            for dl in downloads:
                session.add(DownloadRow(id=dl["id"], data=json.dumps(dl)))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self._session_factory.remove()

    def load_instances(self) -> list[dict]:
        session = self._session()
        try:
            return [json.loads(row.data) for row in session.query(InstanceRow).all()]
        finally:
            self._session_factory.remove()

    def load_downloads(self) -> list[dict]:
        session = self._session()
        try:
            return [json.loads(row.data) for row in session.query(DownloadRow).all()]
        finally:
            self._session_factory.remove()

    # -- Presets --

    def get_all_presets(self) -> dict[str, dict]:
        session = self._session()
        try:
            return {row.model_path: json.loads(row.data)
                    for row in session.query(PresetRow).all()}
        finally:
            self._session_factory.remove()

    def get_preset(self, model_path: str) -> dict | None:
        session = self._session()
        try:
            row = session.get(PresetRow, model_path)
            return json.loads(row.data) if row else None
        finally:
            self._session_factory.remove()

    def save_preset(self, model_path: str, data: dict) -> None:
        session = self._session()
        try:
            row = session.get(PresetRow, model_path)
            if row:
                row.data = json.dumps(data)
            else:
                session.add(PresetRow(model_path=model_path, data=json.dumps(data)))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self._session_factory.remove()

    def delete_preset(self, model_path: str) -> None:
        session = self._session()
        try:
            row = session.get(PresetRow, model_path)
            if row:
                session.delete(row)
                session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self._session_factory.remove()

    # -- Auth --

    def get_user(self, username: str) -> dict | None:
        session = self._session()
        try:
            row = session.get(UserRow, username)
            if not row:
                return None
            return {"username": row.username, "password_hash": row.password_hash}
        finally:
            self._session_factory.remove()

    def save_user(self, username: str, password_hash: str) -> None:
        session = self._session()
        try:
            row = session.get(UserRow, username)
            if row:
                row.password_hash = password_hash
            else:
                session.add(UserRow(username=username, password_hash=password_hash))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self._session_factory.remove()

    def user_count(self) -> int:
        session = self._session()
        try:
            return session.query(func.count(UserRow.username)).scalar()
        finally:
            self._session_factory.remove()

    # -- Settings --

    def get_settings(self) -> dict:
        session = self._session()
        try:
            row = session.get(SettingsRow, "global")
            return json.loads(row.data) if row else {}
        finally:
            self._session_factory.remove()

    def save_settings(self, settings: dict) -> None:
        session = self._session()
        try:
            row = session.get(SettingsRow, "global")
            if row:
                row.data = json.dumps(settings)
            else:
                session.add(SettingsRow(key="global", data=json.dumps(settings)))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self._session_factory.remove()

    # -- API Keys --

    @staticmethod
    def _hash_key(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode()).hexdigest()

    def get_api_keys(self) -> list[dict]:
        session = self._session()
        try:
            return [
                {"id": row.id, "name": row.name, "key_hash": row.key_hash,
                 "prefix": row.prefix, "created_at": row.created_at}
                for row in session.query(ApiKeyRow).all()
            ]
        finally:
            self._session_factory.remove()

    def save_api_key(self, key_entry: dict) -> None:
        session = self._session()
        try:
            row = session.get(ApiKeyRow, key_entry["id"])
            if row:
                row.name = key_entry.get("name", "")
                row.key_hash = key_entry["key_hash"]
                row.prefix = key_entry.get("prefix", "")
                row.created_at = key_entry.get("created_at", 0)
            else:
                session.add(ApiKeyRow(
                    id=key_entry["id"],
                    name=key_entry.get("name", ""),
                    key_hash=key_entry["key_hash"],
                    prefix=key_entry.get("prefix", ""),
                    created_at=key_entry.get("created_at", 0),
                ))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self._session_factory.remove()

    def delete_api_key(self, key_id: str) -> None:
        session = self._session()
        try:
            row = session.get(ApiKeyRow, key_id)
            if row:
                session.delete(row)
                session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self._session_factory.remove()

    def verify_api_key(self, raw_key: str) -> bool:
        hashed = self._hash_key(raw_key)
        session = self._session()
        try:
            return session.query(ApiKeyRow).filter_by(key_hash=hashed).first() is not None
        finally:
            self._session_factory.remove()
