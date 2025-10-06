"""Tests for the loopback IP allocator."""

import os
import tempfile
import threading
import time
from unittest.mock import patch

import pytest

from openhands.runtime.utils.loopback_ip_allocator import (
    LoopbackIPAllocator,
    allocate_loopback_ip,
    get_loopback_ip,
    release_loopback_ip,
)


@pytest.fixture
def temp_allocator():
    """Create a temporary allocator with a temp file."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_file = f.name

    allocator = LoopbackIPAllocator()
    original_file = allocator._allocation_file
    allocator._allocation_file = temp_file

    yield allocator

    # Cleanup
    allocator._allocation_file = original_file
    if os.path.exists(temp_file):
        os.unlink(temp_file)


def test_ip_allocation_basic(temp_allocator):
    """Test basic IP allocation."""
    session_id = 'test_session_1'

    # Allocate an IP
    ip = temp_allocator.allocate_ip(session_id)
    assert ip is not None
    assert ip.startswith('127.')

    # Verify IP is stored
    stored_ip = temp_allocator.get_ip(session_id)
    assert stored_ip == ip

    # Release the IP
    temp_allocator.release_ip(session_id)

    # Verify IP is released
    stored_ip = temp_allocator.get_ip(session_id)
    assert stored_ip is None


def test_ip_reuse_same_session(temp_allocator):
    """Test that the same session gets the same IP."""
    session_id = 'test_session_2'

    # Allocate an IP
    ip1 = temp_allocator.allocate_ip(session_id)

    # Allocate again for the same session
    ip2 = temp_allocator.allocate_ip(session_id)

    # Should get the same IP
    assert ip1 == ip2


def test_different_sessions_get_different_ips(temp_allocator):
    """Test that different sessions get different IPs."""
    session_id1 = 'test_session_3'
    session_id2 = 'test_session_4'

    # Allocate IPs for different sessions
    ip1 = temp_allocator.allocate_ip(session_id1)
    ip2 = temp_allocator.allocate_ip(session_id2)

    # Should get different IPs
    assert ip1 != ip2

    # Clean up
    temp_allocator.release_ip(session_id1)
    temp_allocator.release_ip(session_id2)


def test_ip_allocation_thread_safety(temp_allocator):
    """Test thread safety of IP allocation."""
    num_threads = 10
    session_ids = [f'thread_session_{i}' for i in range(num_threads)]
    allocated_ips = {}
    errors = []

    def allocate_ip_thread(session_id):
        try:
            ip = temp_allocator.allocate_ip(session_id)
            allocated_ips[session_id] = ip
        except Exception as e:
            errors.append(e)

    # Start threads
    threads = []
    for session_id in session_ids:
        thread = threading.Thread(target=allocate_ip_thread, args=(session_id,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Check results
    assert len(errors) == 0, f'Errors occurred: {errors}'
    assert len(allocated_ips) == num_threads
    assert len(set(allocated_ips.values())) == num_threads  # All IPs should be unique

    # Clean up
    for session_id in session_ids:
        temp_allocator.release_ip(session_id)


def test_ip_range_validation(temp_allocator):
    """Test that allocated IPs are in the correct range."""
    session_id = 'test_session_5'

    ip = temp_allocator.allocate_ip(session_id)
    parts = ip.split('.')

    # Should be in range 127.0.1.0 to 127.255.255.255
    assert parts[0] == '127'
    assert int(parts[1]) >= 0 and int(parts[1]) <= 255
    assert int(parts[2]) >= 0 and int(parts[2]) <= 255
    assert int(parts[3]) >= 0 and int(parts[3]) <= 255

    # First IP should be 127.0.1.0
    if ip == '127.0.1.0':
        assert True
    else:
        # Should be greater than 127.0.1.0
        ip_int = (
            (127 << 24) + (int(parts[1]) << 16) + (int(parts[2]) << 8) + int(parts[3])
        )
        min_ip_int = (127 << 24) + (0 << 16) + (1 << 8) + 0
        assert ip_int >= min_ip_int

    temp_allocator.release_ip(session_id)


def test_cleanup_expired_allocations(temp_allocator):
    """Test cleanup of expired allocations."""
    session_id = 'test_session_6'

    # Mock time to simulate old allocation
    current_time = time.time()
    with patch('time.time') as mock_time:
        # Simulate allocation 5 hours ago (longer than the 4-hour expiry)
        old_time = current_time - 18000  # 5 hours ago
        mock_time.return_value = old_time
        temp_allocator.allocate_ip(session_id)

        # Now simulate current time
        mock_time.return_value = current_time

        # Try to allocate for a new session - should trigger cleanup
        new_session = 'new_session'
        temp_allocator.allocate_ip(new_session)

        # Old session should be cleaned up
        assert temp_allocator.get_ip(session_id) is None
        assert temp_allocator.get_ip(new_session) is not None

        temp_allocator.release_ip(new_session)


def test_global_functions():
    """Test the global convenience functions."""
    session_id = 'global_test_session'

    # Test allocation
    ip = allocate_loopback_ip(session_id)
    assert ip is not None
    assert ip.startswith('127.')

    # Test retrieval
    stored_ip = get_loopback_ip(session_id)
    assert stored_ip == ip

    # Test release
    release_loopback_ip(session_id)
    stored_ip = get_loopback_ip(session_id)
    assert stored_ip is None


def test_singleton_behavior():
    """Test that LoopbackIPAllocator is a singleton."""
    allocator1 = LoopbackIPAllocator()
    allocator2 = LoopbackIPAllocator()

    assert allocator1 is allocator2
