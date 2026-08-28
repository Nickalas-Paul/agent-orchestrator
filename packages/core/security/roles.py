"""Role-based access control for HITL operations.

Maps to AWS IAM concepts:
- HITLRole ≈ IAM Role (defines what a principal can do)
- HITLPermission ≈ IAM Action (specific operations)
- HITLAccessControl ≈ IAM Policy evaluation (checks role against required permission)

This is application-layer RBAC — no HTTP auth, no session management.
In production, roles would be sourced from IAM / SSO / LDAP.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from packages.core.logging import get_logger

logger = get_logger("hitl_access_control")


class HITLPermission(str, Enum):
    """Permissions for HITL operations."""

    VIEW_JOBS = "view_jobs"
    SUBMIT_REVIEW = "submit_review"
    OVERRIDE_CONFIDENCE = "override_confidence"
    LIST_PENDING = "list_pending"


class HITLRole(str, Enum):
    """Roles for HITL access control.

    Follows least-privilege principle (IAM 5.1.1):
    - REVIEWER: can view jobs and list pending, but cannot approve/reject
    - APPROVER: can view, list, and submit reviews
    - ADMIN: full access including confidence threshold overrides
    """

    REVIEWER = "reviewer"
    APPROVER = "approver"
    ADMIN = "admin"


# Permission mapping — which roles have which permissions
ROLE_PERMISSIONS: dict[HITLRole, set[HITLPermission]] = {
    HITLRole.REVIEWER: {
        HITLPermission.VIEW_JOBS,
        HITLPermission.LIST_PENDING,
    },
    HITLRole.APPROVER: {
        HITLPermission.VIEW_JOBS,
        HITLPermission.LIST_PENDING,
        HITLPermission.SUBMIT_REVIEW,
    },
    HITLRole.ADMIN: {
        HITLPermission.VIEW_JOBS,
        HITLPermission.LIST_PENDING,
        HITLPermission.SUBMIT_REVIEW,
        HITLPermission.OVERRIDE_CONFIDENCE,
    },
}


class HITLAccessControl:
    """Validates HITL operations against role-based permissions.

    Usage:
        access = HITLAccessControl()
        access.check_permission(HITLRole.REVIEWER, HITLPermission.SUBMIT_REVIEW)
        # raises PermissionError

        access.check_permission(HITLRole.APPROVER, HITLPermission.SUBMIT_REVIEW)
        # passes silently
    """

    def check_permission(
        self,
        role: HITLRole,
        permission: HITLPermission,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Check if a role has a specific permission.

        Args:
            role: The caller's HITL role.
            permission: The permission required for the operation.
            context: Optional context (e.g., job_id) for audit logging.

        Raises:
            PermissionError: If the role does not have the required permission.
        """
        allowed = ROLE_PERMISSIONS.get(role, set())
        if permission not in allowed:
            logger.warning(
                "hitl_access_denied",
                role=role.value,
                permission=permission.value,
                context=context or {},
            )
            raise PermissionError(
                f"Role '{role.value}' does not have permission '{permission.value}'. "
                f"Required role: {self._minimum_role_for(permission)}."
            )
        logger.info(
            "hitl_access_granted",
            role=role.value,
            permission=permission.value,
        )

    def has_permission(self, role: HITLRole, permission: HITLPermission) -> bool:
        """Check permission without raising. Returns True/False."""
        allowed = ROLE_PERMISSIONS.get(role, set())
        return permission in allowed

    def _minimum_role_for(self, permission: HITLPermission) -> str:
        """Find the least-privileged role that has a given permission."""
        for role in [HITLRole.REVIEWER, HITLRole.APPROVER, HITLRole.ADMIN]:
            if permission in ROLE_PERMISSIONS.get(role, set()):
                return role.value
        return "unknown"
