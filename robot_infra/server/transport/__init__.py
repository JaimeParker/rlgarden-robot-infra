"""Transports move ``codec`` messages between a ``RobotService`` and its clients."""
from robot_infra.server.transport.base import (
    READ_ONLY_METHODS,
    Transport,
    TransportClient,
    TransportError,
    is_read_only,
)
from robot_infra.server.transport.local import LocalClient, LocalTransport

__all__ = [
    "READ_ONLY_METHODS",
    "LocalClient",
    "LocalTransport",
    "Transport",
    "TransportClient",
    "TransportError",
    "is_read_only",
]

_GRPC_EXPORTS = ("GrpcClient", "GrpcTransport")


def __getattr__(name: str):
    # Lazy so this package imports without grpcio installed.
    if name in _GRPC_EXPORTS:
        from robot_infra.server.transport import grpc_transport

        return getattr(grpc_transport, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
