"""
Loopback IP allocator for runtime instances.

This module provides a thread-safe and process-safe mechanism to allocate
unique loopback IP addresses in the range 127.0.0.0 to 127.255.255.255
for runtime instances.
"""

import fcntl
import json
import os
import threading
import time
from typing import Optional

from openhands.core.logger import openhands_logger as logger


class LoopbackIPAllocator:
    """Thread-safe and process-safe loopback IP allocator."""

    _instance = None
    _lock = threading.Lock()

    # IP range: 127.0.0.1 to 127.255.255.255 (excluding 127.0.0.0)
    # We'll use 127.0.0.1 to 127.255.255.255 starting from localhost
    MIN_IP = (127, 0, 0, 1)
    MAX_IP = (127, 255, 255, 255)

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self._allocation_file = '/tmp/openhands_loopback_ips.json'
        self._file_lock = threading.Lock()

    def _ip_to_int(self, ip_tuple: tuple[int, int, int, int]) -> int:
        """Convert IP tuple to integer for easier arithmetic."""
        return (ip_tuple[0] << 24) + (ip_tuple[1] << 16) + (ip_tuple[2] << 8) + ip_tuple[3]

    def _int_to_ip(self, ip_int: int) -> tuple[int, int, int, int]:
        """Convert integer to IP tuple."""
        return (
            (ip_int >> 24) & 0xFF,
            (ip_int >> 16) & 0xFF,
            (ip_int >> 8) & 0xFF,
            ip_int & 0xFF
        )

    def _ip_to_str(self, ip_tuple: tuple[int, int, int, int]) -> str:
        """Convert IP tuple to string."""
        return f"{ip_tuple[0]}.{ip_tuple[1]}.{ip_tuple[2]}.{ip_tuple[3]}"

    def _load_allocations(self) -> dict:
        """Load current IP allocations from file with file locking."""
        try:
            if not os.path.exists(self._allocation_file):
                return {}

            with open(self._allocation_file, 'r') as f:
                # Acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    content = f.read()
                    if not content.strip():
                        return {}
                    return json.loads(content)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load IP allocations: {e}")
            return {}

    def _save_allocations(self, allocations: dict) -> None:
        """Save IP allocations to file with file locking."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self._allocation_file), exist_ok=True)

            with open(self._allocation_file, 'w') as f:
                # Acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(allocations, f, indent=2)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except IOError as e:
            logger.error(f"Failed to save IP allocations: {e}")
            raise

    def _cleanup_expired_allocations(self, allocations: dict, current_time: float) -> dict:
        """Remove expired allocations (older than 4 hour)."""
        expired_threshold = current_time - 3600*4  # 4 hour
        cleaned = {}
        for session_id, allocation in allocations.items():
            if allocation.get('timestamp', 0) > expired_threshold:
                cleaned[session_id] = allocation
            else:
                logger.debug(f"Cleaned up expired IP allocation for session {session_id}")
        return cleaned

    def allocate_ip(self, session_id: str) -> str:
        """
        Allocate a unique loopback IP for a session.

        Args:
            session_id: Unique identifier for the session

        Returns:
            Allocated IP address as string

        Raises:
            RuntimeError: If no IP addresses are available
        """
        with self._file_lock:
            current_time = time.time()
            allocations = self._load_allocations()

            # Clean up expired allocations
            allocations = self._cleanup_expired_allocations(allocations, current_time)

            # Check if session already has an IP
            if session_id in allocations:
                existing_ip = allocations[session_id]['ip']
                # Update timestamp
                allocations[session_id]['timestamp'] = current_time
                self._save_allocations(allocations)
                logger.debug(f"Reusing existing IP {existing_ip} for session {session_id}")
                return existing_ip

            # Find next available IP
            allocated_ips = {alloc['ip'] for alloc in allocations.values()}

            min_ip_int = self._ip_to_int(self.MIN_IP)
            max_ip_int = self._ip_to_int(self.MAX_IP)

            for ip_int in range(min_ip_int, max_ip_int + 1):
                ip_tuple = self._int_to_ip(ip_int)
                ip_str = self._ip_to_str(ip_tuple)

                if ip_str not in allocated_ips:
                    # Allocate this IP
                    allocations[session_id] = {
                        'ip': ip_str,
                        'timestamp': current_time
                    }
                    self._save_allocations(allocations)
                    logger.info(f"Allocated IP {ip_str} for session {session_id}")
                    return ip_str

            raise RuntimeError("No available loopback IPs in range 127.0.0.1 to 127.255.255.255")

    def release_ip(self, session_id: str) -> None:
        """
        Release IP allocation for a session.

        Args:
            session_id: Session identifier to release
        """
        with self._file_lock:
            allocations = self._load_allocations()

            if session_id in allocations:
                released_ip = allocations[session_id]['ip']
                del allocations[session_id]
                self._save_allocations(allocations)
                logger.info(f"Released IP {released_ip} for session {session_id}")
            else:
                logger.debug(f"No IP allocation found for session {session_id}")

    def get_ip(self, session_id: str) -> Optional[str]:
        """
        Get currently allocated IP for a session.

        Args:
            session_id: Session identifier

        Returns:
            IP address if allocated, None otherwise
        """
        with self._file_lock:
            allocations = self._load_allocations()
            allocation = allocations.get(session_id)
            return allocation['ip'] if allocation else None


# Global instance
_allocator = LoopbackIPAllocator()


def allocate_loopback_ip(session_id: str) -> str:
    """Allocate a unique loopback IP for a session."""
    return _allocator.allocate_ip(session_id)


def release_loopback_ip(session_id: str) -> None:
    """Release IP allocation for a session."""
    _allocator.release_ip(session_id)


def get_loopback_ip(session_id: str) -> Optional[str]:
    """Get currently allocated IP for a session."""
    return _allocator.get_ip(session_id)
