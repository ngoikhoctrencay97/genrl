import os
import pickle
import time
import random
import threading
import signal
import multiprocessing as mp
from typing import Any, Dict, List
from contextlib import contextmanager

import torch.distributed as dist
from hivemind import DHT, get_dht_time

from genrl.communication.communication import Communication
from genrl.serialization.game_tree import from_bytes, to_bytes
from genrl.logging_utils.global_defs import get_logger


class HivemindBackend(Communication):
    def __init__(
        self,
        initial_peers: List[str] | None = None,
        timeout: int = 300,  # Reduced from 600
        startup_timeout: int = 120,  # Reduced 
        disable_caching: bool = True,  # Default to disabled
        beam_size: int = 20,  # Much smaller
        max_retries: int = 3,  # Reduced retries
        retry_delay: float = 2.0,
        **kwargs,
    ):
        self.world_size = int(os.environ.get("HIVEMIND_WORLD_SIZE", 1))
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.beam_size = beam_size
        self.dht = None
        self.step_ = 0
        
        # Force better multiprocessing settings
        self._setup_multiprocessing()
        
        # DHT settings optimized for stability
        dht_kwargs = {
            "cache_locally": False,
            "cache_on_store": False, 
            "max_peers": 10,  # Very limited
            "request_timeout": 30.0,
            "num_workers": 1,  # Single worker to avoid pipe issues
            "daemon": False,  # Don't use daemon processes
            "compression": None,  # Disable compression
        }
        dht_kwargs.update(kwargs)
        
        self.bootstrap = self._is_bootstrap()
        self._init_dht_safe(initial_peers, **dht_kwargs)

    def _setup_multiprocessing(self):
        """Setup multiprocessing environment for stability."""
        # Set multiprocessing method to spawn for better isolation
        try:
            mp.set_start_method('spawn', force=True)
            get_logger().info("✅ Set multiprocessing method to 'spawn'")
        except RuntimeError:
            get_logger().warning("⚠️ Multiprocessing method already set")
        
        # Set environment variables
        os.environ['PYTHONUNBUFFERED'] = '1'
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'

    def _is_bootstrap(self) -> bool:
        """Determine if this is bootstrap node."""
        return os.environ.get("HIVEMIND_BOOTSTRAP", "false").lower() == "true"

    def _init_dht_safe(self, initial_peers: List[str] | None, **kwargs):
        """Initialize DHT with maximum safety and error recovery."""
        
        for attempt in range(self.max_retries):
            try:
                get_logger().info(f"🔄 DHT init attempt {attempt + 1}/{self.max_retries}")
                
                # Clean up any existing DHT
                if self.dht:
                    self._cleanup_dht()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Short delay between attempts
                if attempt > 0:
                    time.sleep(self.retry_delay * attempt)
                
                # Initialize DHT with timeout protection
                self._dht_init_with_timeout(initial_peers, **kwargs)
                
                # Verify DHT is working
                self._verify_dht_health()
                
                get_logger().info("✅ DHT initialized successfully")
                return
                
            except Exception as e:
                get_logger().error(f"❌ DHT init attempt {attempt + 1} failed: {e}")
                
                # Clean up failed attempt
                self._cleanup_dht()
                
                if attempt == self.max_retries - 1:
                    # Final attempt - try minimal configuration
                    get_logger().warning("🔄 Final attempt with minimal config...")
                    try:
                        self._init_minimal_dht(initial_peers)
                        return
                    except Exception as final_e:
                        get_logger().error(f"❌ All DHT init attempts failed: {final_e}")
                        raise RuntimeError(f"Failed to initialize DHT after all attempts: {final_e}")

    def _dht_init_with_timeout(self, initial_peers, **kwargs):
        """Initialize DHT with timeout protection."""
        
        def signal_handler(signum, frame):
            raise TimeoutError("DHT initialization timeout")
        
        # Set timeout signal
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(self.startup_timeout)
        
        try:
            if self.bootstrap:
                get_logger().info("🚀 Starting as bootstrap node")
                self.dht = DHT(
                    start=True,
                    host_maddrs=["/ip4/0.0.0.0/tcp/0"],  # Only TCP
                    initial_peers=initial_peers or [],
                    startup_timeout=self.startup_timeout,
                    **kwargs
                )
            else:
                get_logger().info("🔗 Starting as worker node") 
                self.dht = DHT(
                    start=True,
                    host_maddrs=["/ip4/0.0.0.0/tcp/0"],  # Only TCP
                    initial_peers=initial_peers or [],
                    startup_timeout=self.startup_timeout,
                    **kwargs
                )
                
        finally:
            signal.alarm(0)  # Cancel timeout

    def _init_minimal_dht(self, initial_peers):
        """Last resort - minimal DHT configuration."""
        get_logger().info("🔧 Trying minimal DHT configuration...")
        
        minimal_kwargs = {
            "cache_locally": False,
            "cache_on_store": False,
            "max_peers": 3,
            "request_timeout": 15.0,
            "num_workers": 1,
            "daemon": False,
        }
        
        self.dht = DHT(
            start=True,
            host_maddrs=["/ip4/0.0.0.0/tcp/0"],
            initial_peers=initial_peers[:1] if initial_peers else [],  # Only first peer
            startup_timeout=30,
            **minimal_kwargs
        )

    def _verify_dht_health(self):
        """Verify DHT is healthy and responsive."""
        max_wait = 30
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                # Test basic DHT functionality
                test_key = f"health_check_{time.time()}"
                test_value = b"test"
                
                # Quick store/get test
                self.dht.store(test_key, test_value, expiration_time=get_dht_time() + 60)
                time.sleep(1)
                
                result, _ = self.dht.get(test_key, latest=True)
                if result:
                    get_logger().info("✅ DHT health check passed")
                    return
                    
            except Exception as e:
                get_logger().debug(f"DHT health check failed: {e}")
                time.sleep(2)
        
        raise RuntimeError("DHT health check failed")

    def _cleanup_dht(self):
        """Clean up DHT resources safely."""
        if self.dht:
            try:
                get_logger().debug("🧹 Cleaning up DHT...")
                
                # Try graceful shutdown first
                self.dht.shutdown()
                
                # Wait a bit for cleanup
                time.sleep(1)
                
            except Exception as e:
                get_logger().warning(f"⚠️ DHT cleanup error: {e}")
            finally:
                self.dht = None

    def all_gather_object(self, obj: Any) -> Dict[str | int, Any]:
        """Robust all_gather with fallback mechanisms."""
        
        if not self.dht:
            get_logger().warning("⚠️ DHT not available, returning single object")
            return {self.get_id(): obj}
        
        key = f"gather_{self.step_}"
        
        # Try main gathering approach
        for attempt in range(self.max_retries):
            try:
                result = self._attempt_gather(obj, key, attempt)
                self.step_ += 1
                return result
                
            except Exception as e:
                get_logger().warning(f"⚠️ Gather attempt {attempt + 1} failed: {e}")
                
                if attempt == self.max_retries - 1:
                    get_logger().error("❌ All gather attempts failed, using fallback")
                    return {self.get_id(): obj}
                
                time.sleep(self.retry_delay * (attempt + 1))
        
        # Final fallback
        return {self.get_id(): obj}

    def _attempt_gather(self, obj: Any, key: str, attempt: int) -> Dict[str | int, Any]:
        """Single gather attempt with timeout."""
        
        # Serialize object
        try:
            obj_bytes = to_bytes(obj)
        except Exception as e:
            raise RuntimeError(f"Serialization failed: {e}")
        
        # Store with reduced beam size for stability
        effective_beam_size = max(1, self.beam_size // (attempt + 1))
        
        try:
            self.dht.store(
                key,
                subkey=str(self.dht.peer_id),
                value=obj_bytes,
                expiration_time=get_dht_time() + self.timeout,
                beam_size=effective_beam_size,
            )
        except Exception as e:
            raise RuntimeError(f"DHT store failed: {e}")
        
        # Wait for propagation (adaptive)
        wait_time = min(3.0, 0.5 * self.world_size)
        time.sleep(wait_time)
        
        # Collect results with timeout
        collected = self._collect_results(key, effective_beam_size)
        
        if len(collected) == 0:
            raise RuntimeError("No results collected")
        
        return collected

    def _collect_results(self, key: str, beam_size: int) -> Dict[str, Any]:
        """Collect results with progressive timeout."""
        
        start_time = time.time()
        max_wait = min(self.timeout, 60)  # Cap at 60 seconds
        
        best_result = {}
        
        while time.time() - start_time < max_wait:
            try:
                output, _ = self.dht.get(key, beam_size=beam_size, latest=True)
                
                if output:
                    # Deserialize results
                    current_result = {}
                    for subkey, value in output.items():
                        try:
                            current_result[subkey] = from_bytes(value.value)
                        except Exception as e:
                            get_logger().warning(f"⚠️ Failed to deserialize {subkey}: {e}")
                    
                    if len(current_result) > len(best_result):
                        best_result = current_result
                    
                    # Stop if we have enough results
                    if len(current_result) >= min(self.world_size, 3):
                        break
                
                time.sleep(1.0)
                
            except Exception as e:
                get_logger().debug(f"Collection attempt failed: {e}")
                time.sleep(2.0)
        
        get_logger().info(f"📊 Collected {len(best_result)} objects")
        return best_result

    def get_id(self):
        """Get node ID safely."""
        if self.dht and hasattr(self.dht, 'peer_id'):
            return str(self.dht.peer_id)
        return f"node_{os.getpid()}"

    def shutdown(self):
        """Safe shutdown."""
        self._cleanup_dht()

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.shutdown()
        except:
            pass
