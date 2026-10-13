"""gRPC transport without protoc: generic method handlers carry ``codec`` (msgpack) bytes.

Wire: ``/robot_infra.RobotRpc/Call`` (unary) takes ``{"method", "kwargs"}``;
``/robot_infra.RobotRpc/Subscribe`` (server stream) takes ``{"method", "rate_hz"}``.
Every response is ``{"result": ...}`` or ``{"error": DeviceError.to_dict()}``.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Iterator

import grpc

from robot_infra.devices.errors import DeviceError
from robot_infra.server.codec import pack, unpack
from robot_infra.server.transport.base import Transport, TransportClient, TransportError, is_read_only

if TYPE_CHECKING:
    from robot_infra.server.service import RobotService

CALL_METHOD = "/robot_infra.RobotRpc/Call"
SUBSCRIBE_METHOD = "/robot_infra.RobotRpc/Subscribe"
TOKEN_METADATA_KEY = "x-robot-token"
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
_MESSAGE_SIZE_OPTIONS = [
    ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
    ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
]
# gRPC's default 1 s first reconnect delay would swallow a whole open() timeout on a LAN link.
_CHANNEL_OPTIONS = _MESSAGE_SIZE_OPTIONS + [
    ("grpc.initial_reconnect_backoff_ms", 100),
    ("grpc.min_reconnect_backoff_ms", 100),
    ("grpc.max_reconnect_backoff_ms", 1000),
]

logger = logging.getLogger("robot_infra.transport.grpc")


class _RobotRpcHandler(grpc.GenericRpcHandler):
    def __init__(self, service: "RobotService", token: str | None) -> None:
        self._service = service
        self._token = token

    def service(self, handler_call_details: grpc.HandlerCallDetails):
        if handler_call_details.method == CALL_METHOD:
            return grpc.unary_unary_rpc_method_handler(
                self._call, request_deserializer=unpack, response_serializer=pack
            )
        if handler_call_details.method == SUBSCRIBE_METHOD:
            return grpc.unary_stream_rpc_method_handler(
                self._subscribe, request_deserializer=unpack, response_serializer=pack
            )
        return None

    def _authorize(self, context: grpc.ServicerContext) -> None:
        if self._token is None:
            return
        metadata = dict(context.invocation_metadata())
        if metadata.get(TOKEN_METADATA_KEY) != self._token:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "token mismatch")

    def _dispatch(self, method: str, kwargs: dict, context: grpc.ServicerContext) -> dict:
        try:
            return {"result": self._service.dispatch(method, kwargs)}
        except DeviceError as error:
            return {"error": error.to_dict()}
        except (ValueError, TypeError) as error:  # unknown method, bad arguments
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        except Exception as error:
            logger.exception("'%s' failed in the robot service", method)
            context.abort(grpc.StatusCode.INTERNAL, repr(error))

    def _call(self, request: dict, context: grpc.ServicerContext) -> dict:
        self._authorize(context)
        return self._dispatch(request["method"], request.get("kwargs") or {}, context)

    def _subscribe(self, request: dict, context: grpc.ServicerContext) -> Iterator[dict]:
        self._authorize(context)
        rate_hz = float(request.get("rate_hz", 0.0))
        if rate_hz <= 0:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"rate_hz must be positive, got {rate_hz}")
        period = 1.0 / rate_hz
        while context.is_active():
            response = self._dispatch(request["method"], {}, context)
            yield response
            if "error" in response:
                return
            time.sleep(period)


class GrpcTransport(Transport):
    def __init__(self, *, max_workers: int = 8) -> None:
        self.max_workers = max_workers
        self._server: grpc.Server | None = None
        self._port: int | None = None

    @property
    def port(self) -> int:
        if self._port is None:
            raise RuntimeError("GrpcTransport is not started")
        return self._port

    def start(self, service: "RobotService", bind: str, *, token: str | None = None) -> None:
        if self._server is not None:
            raise RuntimeError("GrpcTransport is already started; call stop() first")
        server = grpc.server(ThreadPoolExecutor(max_workers=self.max_workers), options=_MESSAGE_SIZE_OPTIONS)
        server.add_generic_rpc_handlers([_RobotRpcHandler(service, token)])
        try:
            port = server.add_insecure_port(bind)
        except RuntimeError:
            port = 0
        if port == 0:
            raise TransportError(
                f"cannot bind the robot daemon to {bind}",
                hint="the port is in use or the address is not local; pick another --bind host:port",
                device=bind,
            )
        server.start()
        self._server = server
        self._port = port
        logger.info("robot daemon listening on %s (port %d)", bind, port)

    def serve(self, service: "RobotService", bind: str, *, token: str | None = None) -> None:
        self.start(service, bind, token=token)
        self._server.wait_for_termination()

    def stop(self, grace_s: float = 1.0) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.stop(grace_s).wait()


class GrpcClient(TransportClient):
    def __init__(
        self,
        address: str,
        *,
        token: str | None = None,
        timeout_s: float = 5.0,
        reconnect_attempts: int = 5,
        reconnect_backoff_s: float = 0.5,
        backoff_multiplier: float = 2.0,
        max_backoff_s: float = 5.0,
        log: logging.Logger = logger,
    ) -> None:
        self.address = address
        self.timeout_s = timeout_s
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_backoff_s = reconnect_backoff_s
        self.backoff_multiplier = backoff_multiplier
        self.max_backoff_s = max_backoff_s
        self._log = log
        self._metadata = ((TOKEN_METADATA_KEY, token),) if token is not None else None
        self._channel: grpc.Channel | None = None
        self._call_rpc = None
        self._subscribe_rpc = None
        self._streams: set = set()
        self._lock = threading.RLock()

    # -- lifecycle -----------------------------------------------------------

    def open(self) -> None:
        with self._lock:
            if self._channel is not None:
                return
            channel = grpc.insecure_channel(self.address, options=_CHANNEL_OPTIONS)
            try:
                grpc.channel_ready_future(channel).result(timeout=self.timeout_s)
            except grpc.FutureTimeoutError:
                channel.close()
                raise TransportError(
                    f"robot daemon unreachable at {self.address}",
                    hint="check it is running on the robot PC: python -m robot_infra.server --config robot.yaml "
                    "--bind 0.0.0.0:<port>; check the network/firewall",
                    device=self.address,
                ) from None
            self._channel = channel
            self._call_rpc = channel.unary_unary(
                CALL_METHOD, request_serializer=pack, response_deserializer=unpack
            )
            self._subscribe_rpc = channel.unary_stream(
                SUBSCRIBE_METHOD, request_serializer=pack, response_deserializer=unpack
            )

    def close(self) -> None:
        with self._lock:
            channel, self._channel = self._channel, None
            self._call_rpc = self._subscribe_rpc = None
            streams, self._streams = self._streams, set()
        for stream in streams:
            stream.cancel()
        if channel is not None:
            channel.close()

    @property
    def is_open(self) -> bool:
        return self._channel is not None

    # -- calls ---------------------------------------------------------------

    def call(self, method: str, **kwargs: Any) -> Any:
        try:
            response = self._invoke(method, kwargs)
        except grpc.RpcError as error:
            if error.code() != grpc.StatusCode.UNAVAILABLE:
                raise self._map_rpc_error(error, method) from None
            if not (self._reconnect() and is_read_only(method)):
                raise self._lost_connection(method) from None
            try:
                response = self._invoke(method, kwargs)
            except grpc.RpcError as retry_error:
                raise self._map_rpc_error(retry_error, method) from None
        return self._unwrap(response)

    def subscribe(self, method: str, rate_hz: float) -> Iterator[Any]:
        if rate_hz <= 0:
            raise ValueError(f"rate_hz must be positive, got {rate_hz}")
        self._require_open(method)
        stream = self._subscribe_rpc({"method": method, "rate_hz": rate_hz}, metadata=self._metadata)
        self._streams.add(stream)
        try:
            for response in stream:
                yield self._unwrap(response)
        except grpc.RpcError as error:
            if stream not in self._streams:  # cancelled by close()
                return
            raise self._map_rpc_error(error, method) from None
        finally:
            self._streams.discard(stream)
            stream.cancel()

    # -- internals -----------------------------------------------------------

    def _require_open(self, method: str) -> None:
        if self._channel is None:
            raise TransportError(
                f"gRPC client for {self.address} is closed; cannot call '{method}'",
                hint="call open() on the client (or connect() on the RemoteRobot) first",
                device=self.address,
            )

    def _invoke(self, method: str, kwargs: dict) -> dict:
        self._require_open(method)
        start = time.perf_counter()
        response = self._call_rpc(
            {"method": method, "kwargs": kwargs}, timeout=self.timeout_s, metadata=self._metadata
        )
        self._last_rtt_ms = (time.perf_counter() - start) * 1000.0
        return response

    @staticmethod
    def _unwrap(response: dict) -> Any:
        if "error" in response:
            raise DeviceError.from_dict(response["error"])
        return response.get("result")

    def _reconnect(self) -> bool:
        with self._lock:
            self.close()
            backoff = self.reconnect_backoff_s
            for attempt in range(1, self.reconnect_attempts + 1):
                self._log.warning("reconnecting to %s, attempt %d/%d", self.address, attempt, self.reconnect_attempts)
                try:
                    self.open()
                    return True
                except TransportError:
                    if attempt < self.reconnect_attempts:
                        time.sleep(backoff)
                        backoff = min(backoff * self.backoff_multiplier, self.max_backoff_s)
            return False

    def _lost_connection(self, method: str) -> TransportError:
        return TransportError(
            f"lost connection to robot daemon at {self.address} during '{method}'",
            hint="the daemon restarted or the network dropped; read the daemon log, then reconnect (connect()) "
            "and set the control mode again — commands are never retried automatically",
            device=self.address,
        )

    def _map_rpc_error(self, error: grpc.RpcError, method: str) -> Exception:
        code, details = error.code(), error.details()
        if code == grpc.StatusCode.UNAVAILABLE:
            return self._lost_connection(method)
        if code == grpc.StatusCode.DEADLINE_EXCEEDED:
            rtt = "unknown" if self._last_rtt_ms is None else f"{self._last_rtt_ms:.0f}"
            return TransportError(
                f"'{method}' exceeded {self.timeout_s}s (last RTT {rtt} ms)",
                hint="check network load / Wi-Fi; raise timeout_s only if RTT is consistently high",
                device=self.address,
                recoverable=True,
            )
        if code == grpc.StatusCode.UNAUTHENTICATED:
            return TransportError(
                "token mismatch",
                hint="pass the same --token to the daemon and the remote config",
                device=self.address,
            )
        if code == grpc.StatusCode.INVALID_ARGUMENT:
            return ValueError(details)
        return TransportError(f"rpc failed: {code} {details}", device=self.address)
