"""Dispatch-layer Permission Enforcement for Ocean Motion.

Guarantees that tool permission checks are evaluated and enforced deterministically
at the dispatch layer BEFORE reaching the LLM or execution services.
"""

from contextvars import ContextVar
import functools
import logging
from typing import Any, Callable, Optional, Set

from app.motion.spec import (
    MOTION_ALLOWED_PERMISSIONS,
    MOTION_FORBIDDEN_PERMISSIONS,
    MotionPermission,
    PersonaType,
)

logger = logging.getLogger("notion-assistant.motion.permissions")

# ContextVar tracking the active persona in the current execution context
active_persona_var: ContextVar[PersonaType] = ContextVar("active_persona", default=PersonaType.OCEAN)


class MotionPermissionError(PermissionError):
    """Raised when a forbidden tool or capability is invoked under the Motion persona."""
    def __init__(self, permission: MotionPermission, persona: PersonaType = PersonaType.MOTION, message: Optional[str] = None):
        self.permission = permission
        self.persona = persona
        msg = message or (
            f"Permission Denied: Permission '{permission.value}' is strictly forbidden for persona '{persona.value}'. "
            f"Motion owns long-term strategic reasoning and trajectory evaluation, not execution."
        )
        super().__init__(msg)


class PermissionEngine:
    """Deterministic permission verification engine."""

    @staticmethod
    def get_current_persona() -> PersonaType:
        """Return the active persona from the context variable."""
        return active_persona_var.get()

    @staticmethod
    def set_current_persona(persona: PersonaType) -> None:
        """Set the active persona for the current context."""
        active_persona_var.set(persona)

    def is_permission_allowed(self, permission: MotionPermission, persona: Optional[PersonaType] = None) -> bool:
        """Check whether a permission is allowed for the given persona."""
        active_p = persona or self.get_current_persona()
        if active_p == PersonaType.OCEAN:
            # Ocean is the execution engine and has full system permissions
            return True
        elif active_p == PersonaType.MOTION:
            return permission in MOTION_ALLOWED_PERMISSIONS
        return False

    def assert_permission(self, permission: MotionPermission, persona: Optional[PersonaType] = None) -> None:
        """Assert that a permission is permitted. Raises MotionPermissionError if forbidden."""
        active_p = persona or self.get_current_persona()
        if not self.is_permission_allowed(permission, persona=active_p):
            logger.warning(
                "Dispatch permission check blocked tool execution: persona=%s, permission=%s",
                active_p.value,
                permission.value,
            )
            raise MotionPermissionError(permission=permission, persona=active_p)


permission_engine = PermissionEngine()


def enforce_persona_permission(permission: MotionPermission):
    """Decorator enforcing dispatch-layer permission before executing a function or tool."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            permission_engine.assert_permission(permission)
            return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            permission_engine.assert_permission(permission)
            return await func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
