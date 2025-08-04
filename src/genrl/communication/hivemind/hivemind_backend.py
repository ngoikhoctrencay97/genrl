import os
import pickle
import time
import gc
import threading
import sys
from collections import deque
from typing import Any, Dict, List
from weakref import WeakValueDictionary

import torch.distributed as dist
from hivemind import DHT, get_dht_time

from genrl.communication.communication import Communication
from genrl.serialization.game_tree import from_bytes, to_bytes

# ✅ SIMPLIFIED: Minimal logging imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class HivemindRendezvouz:
    _STORE = None
    _IS_MASTER = False
    _IS_LAMBDA = False

    @classmethod
    def init(cls, is_master: bool = False):
        """✅ STREAMLINED: Initialize with timeout"""
        cls._IS_MASTER = is_master
        cls._IS_LAMBDA = os.environ.get("LAMBDA", False)
        
        if cls._STORE is None and cls._IS_LAMBDA:
            try:
                world_size = int(os.environ.get("HIVEMIND_WORLD_SIZE", 1))
                cls._STORE = dist.TCPStore(
                    host_name=os.environ["MASTER_ADDR"],
                    port=int(os.environ["MASTER_PORT"]),
                    is_master=is_master,
                    world_size=world_size,
                    wait_for_workers=True,
                    timeout=300,  # 5 minute timeout
                )
            except Exception as e:
                print(f"Rendezvous init failed: {e}")
                raise

    @classmethod
    def is_bootstrap(cls) -> bool:
        return cls._IS_MASTER

    @classmethod
    def set_initial_peers(cls, initial_peers):
        """✅ STREAMLINED: Set peers with size limit"""
        try:
            if cls._STORE is None and cls._IS_LAMBDA:
                cls.init()
                
            if cls._IS_LAMBDA and cls._STORE is not None:
                # Limit peer data size
                if isinstance(initial_peers, list) and len(initial_peers) > 50:
                    initial_peers = initial_peers[:50]  # Keep only 50 peers max
                
                cls._STORE.set("initial_peers", pickle.dumps(initial_peers))
                
        except Exception as e:
            print(f"Set initial peers failed: {e}")
            # Don't raise - allow fallback

    @classmethod
    def get_initial_peers(cls):
        """✅ STREAMLINED: Get peers with timeout"""
        try:
            if cls._STORE is None and cls._IS_LAMBDA:
                cls.init()
                
            if not cls._IS_LAMBDA or cls._STORE is None:
                return []
            
            cls._STORE.wait(["initial_peers"], timeout=60)
            peer_bytes = cls._STORE.get("initial_peers")
            return pickle.loads(peer_bytes)
            
        except Exception as e:
            print(f"Get initial peers failed: {e}")
            return []

    @classmethod
    def cleanup_store(cls):
        """✅ STREAMLINED: Simple cleanup"""
        try:
            if cls._STORE is not None:
                cls._STORE = None
                gc.collect()
        except Exception:
            pass


class HivemindBackend(Communication):
    """✅ STREAMLINED: Optimized HivemindBackend with memory leak fixes"""
    
    def __init__(
        self,
        initial_peers: List[str] | None = None,
        timeout: int = 600,
        disable_caching: bool = False,
        beam_size: int = 1000,
        **kwargs,
    ):
        self.world_size = int(os.environ.get("HIVEMIND_WORLD_SIZE", 1))
        self.timeout = min(timeout, 300)  # ✅ CAP: Max 5 minutes timeout
        self.bootstrap = HivemindRendezvouz.is_bootstrap()
        
        # ✅ CRITICAL: Cap beam_size to prevent P2PD explosion  
        self.beam_size = min(beam_size, 10)  # ✅ AGGRESSIVE CAP: Max 10 instead of 2000
        self.dht = None

        # ✅ Memory limits configuration
        self.max_memory_mb = kwargs.get('max_memory_mb', 2048)  # 2GB default
        self.max_object_size = kwargs.get('max_object_size', 50 * 1024 * 1024)  # 50MB
        self.emergency_cleanup_threshold = 0.9  # 90% of max memory

        # ✅ STREAMLINED: Essential memory management only
        self._init_memory_management()

        # ✅ FORCE: Always disable caching
        kwargs["cache_locally"] = False
        kwargs["cache_on_store"] = False

        # ✅ STREAMLINED: DHT initialization
        try:
            if self.bootstrap:
                self._init_bootstrap_dht(initial_peers, **kwargs)
            else:
                self._init_worker_dht(initial_peers, **kwargs)
        except Exception as e:
            print(f"DHT init failed: {e}")
            raise
            
        self.step_ = 0
        
        # ✅ START: Background cleanup only
        self._cleanup_thread = None
        self._running = True
        self._start_cleanup_thread()

    def _init_memory_management(self):
        """✅ ENHANCED: Memory management with size tracking"""
        # Message buffers with size tracking
        self.message_buffer = deque()
        self.message_buffer_size = 0
        self.max_buffer_size = 100 * 1024 * 1024  # 100MB
        
        # Operation tracking
        self.operation_counter = 0
        self.last_cleanup = time.time()
        self.cleanup_interval = 30  # 30 seconds cleanup
        
        # Health tracking
        self.consecutive_failures = 0
        self.max_failures = 3  # Aggressive failure limit
        
        # Cache with LRU-like behavior and size tracking
        self.cache = {}
        self.cache_access_times = {}
        self.cache_size = 0
        self.max_cache_size = 50  # Max entries
        self.max_cache_bytes = 200 * 1024 * 1024  # 200MB

        # Track DHT keys for cleanup
        self.active_dht_keys = deque(maxlen=1000)

    def _get_object_size(self, obj):
        """Calculate object size in bytes"""
        if isinstance(obj, bytes):
            return len(obj)
        return sys.getsizeof(obj)

    def _init_bootstrap_dht(self, initial_peers, **kwargs):
        """✅ STREAMLINED: Bootstrap DHT"""
        # Filter to supported parameters only
        supported = ['cache_locally', 'cache_on_store', 'listen', 'announce_maddrs']
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported}
        
        self.dht = DHT(
            start=True,
            host_maddrs=[f"/ip4/0.0.0.0/tcp/0"],  # ✅ SIMPLIFIED: TCP only
            initial_peers=initial_peers,
            **filtered_kwargs,
        )
        
        try:
            dht_maddrs = self.dht.get_visible_maddrs(latest=True)
            HivemindRendezvouz.set_initial_peers(dht_maddrs)
        except Exception as e:
            print(f"Bootstrap setup failed: {e}")

    def _init_worker_dht(self, initial_peers, **kwargs):
        """✅ STREAMLINED: Worker DHT"""
        if initial_peers is None:
            try:
                initial_peers = HivemindRendezvouz.get_initial_peers()
            except Exception:
                initial_peers = []
        
        # Filter to supported parameters only
        supported = ['cache_locally', 'cache_on_store', 'listen', 'announce_maddrs']
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported}
        
        self.dht = DHT(
            start=True,
            host_maddrs=[f"/ip4/0.0.0.0/tcp/0"],  # ✅ SIMPLIFIED: TCP only
            initial_peers=initial_peers,
            **filtered_kwargs,
        )

    def _start_cleanup_thread(self):
        """✅ ENHANCED: Background cleanup with memory monitoring"""
        def cleanup_worker():
            while self._running:
                try:
                    time.sleep(self.cleanup_interval)
                    self._periodic_cleanup()
                    self._check_memory_usage()
                except Exception as e:
                    print(f"Cleanup error: {e}")
                    time.sleep(60)  # Back off on error
                    
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()

    def _check_memory_usage(self):
        """Monitor and enforce memory limits"""
        if not PSUTIL_AVAILABLE:
            return
            
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.max_memory_mb * self.emergency_cleanup_threshold:
                print(f"WARNING: Memory usage {memory_mb:.1f}MB approaching limit")
                self._emergency_cleanup()
                
            elif memory_mb > self.max_memory_mb:
                raise MemoryError(f"Memory usage {memory_mb:.1f}MB exceeds limit {self.max_memory_mb}MB")
                
        except psutil.Error:
            pass

    def _periodic_cleanup(self):
        """✅ ENHANCED: Comprehensive cleanup"""
        try:
            current_time = time.time()
            
            # Skip if cleaned recently
            if current_time - self.last_cleanup < self.cleanup_interval:
                return
            
            # 1. Clean expired messages from buffer
            cutoff_time = current_time - 300  # 5 minutes
            while self.message_buffer and self.message_buffer[0][1] < cutoff_time:
                key, timestamp = self.message_buffer.popleft()
                # Try to delete from DHT if possible
                try:
                    if self.dht and hasattr(self.dht, 'delete'):
                        self.dht.delete(key)
                except:
                    pass
            
            # 2. Clean cache based on LRU and size
            if len(self.cache) > self.max_cache_size or self.cache_size > self.max_cache_bytes:
                # Sort by access time (LRU)
                sorted_keys = sorted(self.cache_access_times.items(), key=lambda x: x[1])
                
                # Remove oldest entries until within limits
                while (len(self.cache) > self.max_cache_size * 0.7 or 
                       self.cache_size > self.max_cache_bytes * 0.7) and sorted_keys:
                    key, _ = sorted_keys.pop(0)
                    if key in self.cache:
                        obj_size = self._get_object_size(self.cache[key])
                        del self.cache[key]
                        del self.cache_access_times[key]
                        self.cache_size -= obj_size
            
            # 3. Recalculate buffer size
            self.message_buffer_size = sum(self._get_object_size(item) for item in self.message_buffer)
            
            # 4. Force garbage collection periodically
            if self.operation_counter % 100 == 0:
                gc.collect()
            
            self.last_cleanup = current_time
            
        except Exception as e:
            print(f"Periodic cleanup error: {e}")

    def _emergency_cleanup(self):
        """Emergency cleanup when memory is critical"""
        print("EMERGENCY CLEANUP: Clearing all non-essential data")
        
        try:
            # Clear all caches
            self.cache.clear()
            self.cache_access_times.clear()
            self.cache_size = 0
            
            # Clear message buffer
            self.message_buffer.clear()
            self.message_buffer_size = 0
            
            # Clear DHT keys tracking
            self.active_dht_keys.clear()
            
            # Reset counters
            self.consecutive_failures = 0
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            print(f"Emergency cleanup error: {e}")

    def all_gather_object(self, obj: Any) -> Dict[str | int, Any]:
        """✅ ENHANCED: Memory-safe all_gather"""
        key = str(self.step_)
        
        try:
            self.operation_counter += 1
            
            # ✅ PERIODIC CLEANUP
            if self.operation_counter % 20 == 0:
                self._periodic_cleanup()
            
            # ✅ SERIALIZE WITH SIZE CHECK
            try:
                obj_bytes = to_bytes(obj)
                obj_size = len(obj_bytes)
                
                # Check object size limit
                if obj_size > self.max_object_size:
                    raise ValueError(f"Object too large: {obj_size/1024/1024:.1f}MB exceeds limit {self.max_object_size/1024/1024:.1f}MB")
                
                # Warn about large objects
                if obj_size > 10 * 1024 * 1024:  # 10MB
                    print(f"Warning: Large object {obj_size/1024/1024:.1f}MB")
                
            except Exception as e:
                print(f"Serialization failed: {e}")
                return {str(self.dht.peer_id): obj}
            
            # ✅ DHT STORE with tracking
            try:
                self.dht.store(
                    key,
                    subkey=str(self.dht.peer_id),
                    value=obj_bytes,
                    expiration_time=get_dht_time() + self.timeout,
                    beam_size=self.beam_size,
                )
                
                # Track in buffer with size limit
                if self.message_buffer_size + obj_size > self.max_buffer_size:
                    # Remove oldest entries
                    while self.message_buffer and self.message_buffer_size + obj_size > self.max_buffer_size:
                        old_key, _ = self.message_buffer.popleft()
                        # Approximate size reduction
                        self.message_buffer_size = max(0, self.message_buffer_size - self.max_object_size // 10)
                
                self.message_buffer.append((key, time.time()))
                self.message_buffer_size += obj_size
                self.active_dht_keys.append(key)
                
            except Exception as e:
                print(f"DHT store failed: {e}")
                self.consecutive_failures += 1
                return {str(self.dht.peer_id): obj}
            
            # ✅ RETRIEVE WITH TIMEOUT
            time.sleep(0.5)  # Brief wait
            
            start_time = time.monotonic()
            max_wait = min(self.timeout, 120)  # Cap at 2 minutes
            
            results = {}
            while True:
                try:
                    output_, _ = self.dht.get(key, beam_size=self.beam_size, latest=True)
                    
                    elapsed = time.monotonic() - start_time
                    
                    # Process results as they come in
                    for peer_key, value in output_.items():
                        if peer_key not in results:
                            try:
                                deserialized = from_bytes(value.value)
                                results[peer_key] = deserialized
                                
                                # Cache with size tracking
                                if len(self.cache) < self.max_cache_size:
                                    cache_key = f"{key}_{peer_key}"
                                    self.cache[cache_key] = deserialized
                                    self.cache_access_times[cache_key] = time.time()
                                    self.cache_size += self._get_object_size(deserialized)
                                    
                            except Exception:
                                continue  # Skip failed deserialization
                    
                    if len(results) >= self.world_size or elapsed > max_wait:
                        break
                        
                    time.sleep(0.5)  # Short wait between retries
                    
                except Exception as e:
                    print(f"DHT get failed: {e}")
                    self.consecutive_failures += 1
                    break
            
            # Ensure self is included
            if str(self.dht.peer_id) not in results:
                results[str(self.dht.peer_id)] = obj
            
            # Clean up this operation's data from DHT if possible
            try:
                if hasattr(self.dht, 'delete'):
                    self.dht.delete(key, subkey=str(self.dht.peer_id))
            except:
                pass
            
            self.step_ += 1
            
            # Reset failure counter on success
            if len(results) > 1:
                self.consecutive_failures = 0
            
            return dict(sorted(results.items()))
            
        except (BlockingIOError, EOFError) as io_error:
            # ✅ HANDLE: The specific I/O errors
            print(f"I/O error in all_gather: {io_error}")
            self.consecutive_failures += 1
            
            # Emergency cleanup after multiple failures
            if self.consecutive_failures >= self.max_failures:
                print("Too many failures, emergency cleanup")
                self._emergency_cleanup()
                self.consecutive_failures = 0
            
            return {str(self.dht.peer_id): obj}
            
        except Exception as e:
            print(f"Unexpected error: {e}")
            self.consecutive_failures += 1
            
            # Check memory on any error
            self._check_memory_usage()
            
            return {str(self.dht.peer_id): obj}

    def get_id(self):
        """✅ SAFE: Get peer ID"""
        try:
            return str(self.dht.peer_id) if self.dht else "unknown"
        except Exception:
            return "error"

    def cleanup(self):
        """✅ ENHANCED: Comprehensive cleanup"""
        try:
            self._running = False
            
            # Clear all data structures
            self.cache.clear()
            self.cache_access_times.clear()
            self.cache_size = 0
            
            self.message_buffer.clear()
            self.message_buffer_size = 0
            
            self.active_dht_keys.clear()
            
            # Shutdown DHT
            if self.dht:
                try:
                    self.dht.shutdown()
                except:
                    pass
                self.dht = None
            
            # Clean up rendezvous
            HivemindRendezvouz.cleanup_store()
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            print(f"Cleanup error: {e}")

    def __del__(self):
        """✅ DESTRUCTOR: Cleanup on destruction"""
        try:
            self.cleanup()
        except Exception:
            pass

    def get_stats(self):
        """Get detailed stats for monitoring"""
        try:
            stats = {
                'operations': self.operation_counter,
                'failures': self.consecutive_failures,
                'cache_entries': len(self.cache),
                'cache_size_mb': self.cache_size / 1024 / 1024,
                'buffer_entries': len(self.message_buffer),
                'buffer_size_mb': self.message_buffer_size / 1024 / 1024,
                'active_dht_keys': len(self.active_dht_keys),
                'beam_size': self.beam_size,
            }
            
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                memory_info = process.memory_info()
                stats.update({
                    'memory_rss_mb': memory_info.rss / 1024 / 1024,
                    'memory_vms_mb': memory_info.vms / 1024 / 1024,
                    'memory_percent': process.memory_percent(),
                })
            
            return stats
            
        except Exception:
            return {}
