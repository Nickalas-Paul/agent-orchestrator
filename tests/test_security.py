"""Tests for HITL RBAC and secrets management."""

from __future__ import annotations

import pytest

from packages.core.security import (
    EnvironmentSecretsProvider,
    HITLAccessControl,
    HITLPermission,
    HITLRole,
)


def test_reviewer_can_view_jobs() -> None:
    access = HITLAccessControl()
    access.check_permission(HITLRole.REVIEWER, HITLPermission.VIEW_JOBS)
    assert access.has_permission(HITLRole.REVIEWER, HITLPermission.VIEW_JOBS)


def test_reviewer_cannot_submit_review() -> None:
    access = HITLAccessControl()
    assert not access.has_permission(HITLRole.REVIEWER, HITLPermission.SUBMIT_REVIEW)
    with pytest.raises(PermissionError):
        access.check_permission(HITLRole.REVIEWER, HITLPermission.SUBMIT_REVIEW)


def test_approver_can_submit_review() -> None:
    access = HITLAccessControl()
    access.check_permission(HITLRole.APPROVER, HITLPermission.SUBMIT_REVIEW)
    assert access.has_permission(HITLRole.APPROVER, HITLPermission.SUBMIT_REVIEW)


def test_approver_cannot_override_confidence() -> None:
    access = HITLAccessControl()
    assert not access.has_permission(HITLRole.APPROVER, HITLPermission.OVERRIDE_CONFIDENCE)
    with pytest.raises(PermissionError):
        access.check_permission(HITLRole.APPROVER, HITLPermission.OVERRIDE_CONFIDENCE)


def test_admin_has_all_permissions() -> None:
    access = HITLAccessControl()
    for permission in HITLPermission:
        assert access.has_permission(HITLRole.ADMIN, permission)
        access.check_permission(HITLRole.ADMIN, permission)


def test_check_permission_raises_on_denied() -> None:
    access = HITLAccessControl()
    with pytest.raises(PermissionError, match="submit_review"):
        access.check_permission(
            HITLRole.REVIEWER,
            HITLPermission.SUBMIT_REVIEW,
            context={"job_id": "abc"},
        )


def test_has_permission_returns_bool_without_raising() -> None:
    access = HITLAccessControl()
    assert access.has_permission(HITLRole.REVIEWER, HITLPermission.LIST_PENDING) is True
    assert access.has_permission(HITLRole.REVIEWER, HITLPermission.SUBMIT_REVIEW) is False


def test_minimum_role_for_returns_least_privileged() -> None:
    access = HITLAccessControl()
    assert access._minimum_role_for(HITLPermission.VIEW_JOBS) == "reviewer"
    assert access._minimum_role_for(HITLPermission.SUBMIT_REVIEW) == "approver"
    assert access._minimum_role_for(HITLPermission.OVERRIDE_CONFIDENCE) == "admin"


def test_environment_secrets_provider_reads_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BATCH8_TEST_SECRET", "s3cr3t")
    provider = EnvironmentSecretsProvider()
    assert provider.get_secret("BATCH8_TEST_SECRET") == "s3cr3t"


def test_environment_secrets_provider_returns_default_for_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BATCH8_MISSING_SECRET", raising=False)
    provider = EnvironmentSecretsProvider()
    assert provider.get_secret("BATCH8_MISSING_SECRET", default="fallback") == "fallback"


def test_environment_secrets_provider_has_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BATCH8_HAS_SECRET", "1")
    monkeypatch.delenv("BATCH8_NO_SECRET", raising=False)
    provider = EnvironmentSecretsProvider()
    assert provider.has_secret("BATCH8_HAS_SECRET") is True
    assert provider.has_secret("BATCH8_NO_SECRET") is False


def test_environment_secrets_provider_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_DB_PASS", "password")
    provider = EnvironmentSecretsProvider(prefix="APP_")
    assert provider.get_secret("DB_PASS") == "password"
    assert provider.has_secret("DB_PASS") is True
