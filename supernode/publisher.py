from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .adapter import SupernodeAdapter
from .discovery import SupernodeDiscovery
from .manifest import (
    SupernodeManifest,
    ModelCapability,
    HardwareInfo,
    MANIFEST_REFRESH_S,
    supernode_record_key,
    model_provider_key,
)

logger = logging.getLogger(__name__)


class ManifestPublisher:
    """Periodically builds and publishes a SupernodeManifest from an adapter.

    Lifecycle:
        publisher = ManifestPublisher(adapter, discovery, ...)
        publisher.start()   # background thread
        ...
        publisher.stop()    # graceful shutdown, removes manifest
    """

    def __init__(
        self,
        adapter: SupernodeAdapter,
        discovery: SupernodeDiscovery,
        private_key: Ed25519PrivateKey,
        peer_id: str,
        libp2p_peer_id: str,
        version: str = "0.1.0",
        listen_addrs: list[str] | None = None,
        nat_status: str = "unknown",
        region: str = "",
        hardware: HardwareInfo | None = None,
        refresh_interval: float = MANIFEST_REFRESH_S,
        p2p_node: Any | None = None,
    ):
        self._adapter = adapter
        self._discovery = discovery
        self._private_key = private_key
        self._peer_id = peer_id
        self._libp2p_peer_id = libp2p_peer_id
        self._version = version
        self._listen_addrs = listen_addrs or []
        self._nat_status = nat_status
        self._region = region
        self._hardware = hardware or HardwareInfo()
        self._refresh_interval = refresh_interval
        self._p2p_node = p2p_node

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_model_ids: list[str] = []
        self._publish_count = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="manifest-publisher", daemon=True,
        )
        self._thread.start()
        logger.info("manifest_publisher_started peer=%s", self._peer_id)

    def stop(self) -> None:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=5.0)
            self._thread = None

        self._stop_providing_dht()
        self._discovery.remove_manifest(self._libp2p_peer_id)
        logger.info("manifest_publisher_stopped peer=%s removed_manifest=true", self._peer_id)

    @property
    def publish_count(self) -> int:
        return self._publish_count

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        try:
            asyncio.run(self._publish())
            while not self._stop_event.wait(timeout=self._refresh_interval):
                asyncio.run(self._publish())
        except Exception:
            logger.error("manifest_publisher_crash", exc_info=True)

    async def _publish(self) -> None:
        try:
            models = await self._adapter.list_models()
        except Exception:
            logger.warning("manifest_publish_list_models_failed", exc_info=True)
            return

        current_ids = sorted(m.model_id for m in models)
        if current_ids != self._last_model_ids and self._publish_count > 0:
            logger.info(
                "manifest_models_changed old=%s new=%s",
                self._last_model_ids, current_ids,
            )

        try:
            status = await self._adapter.get_status()
            models_loaded = set(status.models_loaded)
        except Exception:
            models_loaded = set()

        capabilities = [
            ModelCapability(
                model_id=m.model_id,
                model_family=m.model_family,
                parameter_count=m.parameter_count,
                quantization=m.quantization,
                context_length=m.context_length,
                supports_streaming=m.supports_streaming,
                supports_system_prompt=m.supports_system_prompt,
                warm=m.model_id in models_loaded,
                estimated_tps=0.0,
                weights_hash=self._adapter.get_weights_hash(m.model_id) or "",
                weights_size=0,
            )
            for m in models
        ]

        manifest = SupernodeManifest(
            peer_id=self._peer_id,
            libp2p_peer_id=self._libp2p_peer_id,
            backend_type=self._adapter.backend_type(),
            version=self._version,
            integration_level=self._adapter.integration_level(),
            trust_tier=self._adapter.trust_tier(),
            models=capabilities,
            max_concurrent=status.max_concurrent if models_loaded else 4,
            max_context_length=max((m.context_length for m in capabilities), default=4096),
            hardware=self._hardware,
            listen_addrs=self._listen_addrs,
            nat_status=self._nat_status,
            region=self._region,
        )

        manifest.sign(self._private_key)

        if self._discovery.register_manifest(manifest):
            self._publish_count += 1
            self._last_model_ids = current_ids
            logger.debug(
                "manifest_published peer=%s models=%d count=%d",
                self._peer_id, len(capabilities), self._publish_count,
            )
            self._publish_to_dht(manifest)
        else:
            logger.warning("manifest_publish_register_failed peer=%s", self._peer_id)

    def _publish_to_dht(self, manifest: SupernodeManifest) -> None:
        if self._p2p_node is None:
            return
        try:
            manifest_key = supernode_record_key(self._libp2p_peer_id)
            self._p2p_node.put_record_raw(
                key=manifest_key.encode(), value=manifest.to_cbor(),
            )
            for model in manifest.models:
                key = model_provider_key(model.model_id)
                self._p2p_node.start_providing(key=key.encode())
            logger.debug(
                "dht_published peer=%s models=%d",
                self._libp2p_peer_id, len(manifest.models),
            )
        except Exception:
            logger.warning("dht_publish_failed", exc_info=True)

    def _stop_providing_dht(self) -> None:
        if self._p2p_node is None or not self._last_model_ids:
            return
        try:
            for model_id in self._last_model_ids:
                key = model_provider_key(model_id)
                self._p2p_node.stop_providing(key=key.encode())
            logger.info(
                "dht_stop_providing models=%s", self._last_model_ids,
            )
        except Exception:
            logger.warning("dht_stop_providing_failed", exc_info=True)

    def publish_now(self) -> bool:
        """Force an immediate publish (blocking). Returns True on success."""
        asyncio.run(self._publish())
        return True
