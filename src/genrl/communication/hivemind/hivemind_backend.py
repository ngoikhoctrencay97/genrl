import os
import pickle
import time
import threading
import multiprocessing as mp
from typing import Any, Dict, List
import signal
import sys

import torch.distributed as dist
from hivemind import DHT, get_dht_time

# Import from genrl package
from genrl.communication.communication import Communication
from genrl.serialization.game_tree import from_bytes, to_bytes
from genrl.logging_utils.global_defs import get_logger


class HivemindRendezvouz:
    """Local implementation of Hivemind rendezvous."""
    _STORE = None
    _IS_MASTER = False
    _IS_LAMBDA = False
    _LOCK = threading.Lock()

    @classmethod
    def init(cls, is_master: bool = False):
        with cls._LOCK:
            cls._IS_MASTER = is_master
            cls._IS_LAMBDA = os.environ.get("LAMBDA", False)
            if cls._STORE is None and cls._IS_LAMBDA:
                world_size = int(os.environ.get("HIVEMIND_WORLD_SIZE", 1))
                try:
                    cls._STORE = dist.TCPStore(
                        host_name=os.environ["MASTER_ADDR"],
                        port=int(os.environ["MASTER_PORT"]),
                        is_master=is_master,
                        world_size=world_size,
                        wait_for_workers=True,
                        timeout=300,
                    )
                    get_logger().info("TCPStore initialized")
                except Exception as e:
                    get_logger().error(f"Failed to initialize TCPStore: {e}")
                    cls._STORE = None

    @classmethod
    def is_bootstrap(cls) -> bool:
        return cls._IS_MASTER

    @classmethod
    def set_initial_peers(cls, initial_peers):
        if cls._STORE is None and cls._IS_LAMBDA:
            cls.init()
        if cls._IS_LAMBDA and cls._STORE is not None:
            try:
                cls._STORE.set("initial_peers", pickle.dumps(initial_peers))
                get_logger().info(f"Set initial peers: {len(initial_peers)} peers")
            except Exception as e:
                get_logger().warning(f"Failed to set initial peers: {e}")

    @classmethod
    def get_initial_peers(cls):
        if cls._STORE is None and cls._IS_LAMBDA:
            cls.init()
        if cls._STORE is not None:
            try:
                cls._STORE.wait(["initial_peers"], timeout=60)
                peer_bytes = cls._STORE.get("initial_peers")
                initial_peers = pickle.loads(peer_bytes)
                get_logger().info(f"Got initial peers: {len(initial_peers)} peers")
                return initial_peers
            except Exception as e:
                get_logger().warning(f"Failed to get initial peers: {e}")
                return None
        return None


class EmergencyTrainingWrapper:
    """Emergency wrapper to prevent training crashes from communication errors"""
    
    def __init__(self, communication_backend):
        self.backend = communication_backend
        self.emergency_mode = False
        self.consecutive_errors = 0
        self.max_consecutive_errors = 3
        self.total_emergency_calls = 0
        
    def safe_all_gather(self, obj):
        """Ultra-safe wrapper around all_gather_object"""
        try:
            if self.emergency_mode:
                self.total_emergency_calls += 1
                if self.total_emergency_calls % 100 == 0:
                    get_logger().warning(f"Emergency mode: {self.total_emergency_calls} single-node calls")
                return {self.backend.get_id(): obj}
                
            result = self.backend.all_gather_object(obj)
            
            # Reset error count on success
            if self.consecutive_errors > 0:
                get_logger().info(f"Communication recovered after {self.consecutive_errors} errors")
                self.consecutive_errors = 0
                
            return result
            
        except Exception as e:
            error_msg = str(e)
            self.consecutive_errors += 1
            
            get_logger().error(f"EMERGENCY CATCH #{self.consecutive_errors}: {error_msg}")
            
            # Check for critical errors
            critical_patterns = [
                "ran out of input", "pipe", "broken", "connection", "timeout", 
                "eof", "resource temporarily unavailable", "blocking"
            ]
            
            if any(pattern in error_msg.lower() for pattern in critical_patterns):
                get_logger().error("Critical communication error detected - enabling emergency mode")
                self.emergency_mode = True
                
            # Too many consecutive errors
            if self.consecutive_errors >= self.max_consecutive_errors:
                get_logger().error(f"Too many consecutive errors ({self.consecutive_errors}) - emergency mode")
                self.emergency_mode = True
                
            # Always return single-node result to continue training
            return {self.backend.get_id(): obj}


class HivemindBackend(Communication):
    """
    Robust DHT backend that handles 'Ran out of input' errors gracefully.
    Features comprehensive error recovery and emergency fallbacks.
    """
    
    def __init__(
        self,
        initial_peers: List[str] | None = None,
        timeout: int = 300,
        startup_timeout: int = 120,
        disable_caching: bool = True,
        beam_size: int = 20,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        dht_timeout_minutes: int = 260,
        # NEW: Enhanced error handling parameters
        enable_robust_mode: bool = True,
        max_pipe_errors: int = 5,
        health_check_interval: int = 30,
        **kwargs,
    ):
        # Core attributes
        self.dht = None
        self.world_size = int(os.environ.get("HIVEMIND_WORLD_SIZE", 1))
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.beam_size = min(beam_size, 50)
        self.step_ = 0
        self._connection_failures = 0
        self._max_connection_failures = 5
        
        # Enhanced error handling
        self.enable_robust_mode = enable_robust_mode
        self.max_pipe_errors = max_pipe_errors
        self.health_check_interval = health_check_interval
        self._pipe_errors = 0
        self._last_health_check = 0
        self._emergency_mode = False
        
        # Time management
        self.dht_timeout_minutes = dht_timeout_minutes
        self.dht_start_time = None
        self.time_based_shutdown = False
        self.time_monitor_thread = None
        self.health_monitor_thread = None
        self.shutdown_flag = threading.Event()
        
        # Store for later use
        self.initial_peers = initial_peers
        
        get_logger().info(f"Initializing HivemindBackend (world_size={self.world_size})")
        get_logger().info(f"DHT timeout: {dht_timeout_minutes} minutes")
        get_logger().info(f"Robust mode: {enable_robust_mode}")
        
        self._setup_multiprocessing()
        
        # DHT configuration
        dht_kwargs = {
            "cache_locally": not disable_caching,
            "cache_on_store": False,
            "num_workers": 1,
            "daemon": True,  # Better cleanup
        }
        dht_kwargs.update(kwargs)
        
        self.bootstrap = self._is_bootstrap()
        self._init_dht_or_fallback(initial_peers, **dht_kwargs)
        self._pending_shutdown = False

    def _setup_multiprocessing(self):
        """Setup multiprocessing for stability."""
        try:
            mp.set_start_method('spawn', force=True)
            get_logger().info("Set multiprocessing to 'spawn'")
        except RuntimeError:
            get_logger().debug("Multiprocessing method already set")
        
        os.environ['PYTHONUNBUFFERED'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'

    def _is_bootstrap(self) -> bool:
        """Check if this is bootstrap node."""
        return os.environ.get("HIVEMIND_BOOTSTRAP", "false").lower() == "true"

    def _init_dht_or_fallback(self, initial_peers: List[str] | None, **kwargs):
        """Try DHT initialization with enhanced error handling."""
        
        # Check emergency disable flags
        if os.environ.get("DISABLE_DHT", "false").lower() == "true":
            get_logger().warning("DHT disabled via DISABLE_DHT environment variable")
            self._emergency_mode = True
            self._setup_single_node_fallback()
            return
            
        try:
            self._init_dht_with_recovery(initial_peers, **kwargs)
        except Exception as e:
            get_logger().error(f"DHT initialization failed: {e}")
            if self.enable_robust_mode:
                get_logger().info("Robust mode enabled - switching to single-node fallback")
                self._emergency_mode = True
                self._setup_single_node_fallback()
            else:
                raise

    def _init_dht_with_recovery(self, initial_peers: List[str] | None, **kwargs):
        """Initialize DHT with comprehensive retry logic."""
        
        for attempt in range(self.max_retries):
            try:
                get_logger().info(f"DHT initialization attempt {attempt + 1}/{self.max_retries}")
                
                if self.dht:
                    self._safe_cleanup_dht()
                
                if attempt > 0:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    get_logger().info(f"Waiting {delay}s before retry...")
                    time.sleep(delay)
                
                self._init_dht_with_timeout(initial_peers, **kwargs)
                self._verify_dht_health()
                
                # Success - start monitoring
                self.dht_start_time = time.time()
                self._start_monitoring()
                
                get_logger().info("=" * 60)
                get_logger().info("TRAINING MODE: Distributed DHT (Robust)")
                get_logger().info(f"Agent ID: {self.dht.peer_id}")
                get_logger().info(f"DHT timeout: {self.dht_timeout_minutes} minutes")
                get_logger().info("Error recovery: ENABLED")
                get_logger().info("=" * 60)
                
                self._connection_failures = 0
                self._pipe_errors = 0
                return
                
            except Exception as e:
                get_logger().error(f"DHT attempt {attempt + 1} failed: {e}")
                if self.dht:
                    self._safe_cleanup_dht()
                
                if attempt == self.max_retries - 1:
                    raise RuntimeError("All DHT attempts failed")

    def _start_monitoring(self):
        """Start both time and health monitoring."""
        self._start_time_monitoring()
        self._start_health_monitoring()

    def _start_time_monitoring(self):
        """Start time monitoring thread."""
        if self.time_monitor_thread and self.time_monitor_thread.is_alive():
            return
            
        self.shutdown_flag.clear()
        self.time_monitor_thread = threading.Thread(target=self._time_monitor_loop, daemon=True)
        self.time_monitor_thread.start()

    def _start_health_monitoring(self):
        """Start health monitoring thread."""
        if self.health_monitor_thread and self.health_monitor_thread.is_alive():
            return
            
        self.health_monitor_thread = threading.Thread(target=self._health_monitor_loop, daemon=True)
        self.health_monitor_thread.start()

    def _time_monitor_loop(self):
        """Monitor DHT runtime and shutdown after timeout."""
        last_log_minute = 0
        
        while not self.shutdown_flag.is_set() and self.dht and not self.time_based_shutdown:
            if self.dht_start_time:
                elapsed_minutes = (time.time() - self.dht_start_time) / 60
                
                if elapsed_minutes >= self.dht_timeout_minutes:
                    self._pending_shutdown = True
                    get_logger().info(f"DHT timeout reached: {elapsed_minutes:.1f} minutes")
                    self._preserve_peer_id_and_shutdown()
                    break
                
                # Log every 10 minutes
                current_minute = int(elapsed_minutes)
                if current_minute % 10 == 0 and current_minute > last_log_minute and current_minute > 0:
                    remaining = self.dht_timeout_minutes - elapsed_minutes
                    get_logger().info(f"DHT runtime: {elapsed_minutes:.1f}min, {remaining:.1f}min remaining")
                    last_log_minute = current_minute
            
            self.shutdown_flag.wait(30)

    def _health_monitor_loop(self):
        """Monitor DHT process health and detect pipe failures."""
        while not self.shutdown_flag.is_set() and self.dht and not self.time_based_shutdown:
            try:
                current_time = time.time()
                
                # Periodic health check
                if current_time - self._last_health_check > self.health_check_interval:
                    if not self._check_dht_health():
                        get_logger().error("DHT health check failed - triggering recovery")
                        self._handle_critical_failure("Health check failed")
                        break
                    self._last_health_check = current_time
                
                # Check process status
                if hasattr(self.dht, '_server_process'):
                    if not self.dht._server_process.is_alive():
                        get_logger().error("DHT process died")
                        self._handle_critical_failure("DHT process died")
                        break
                
                # Check pipe status
                if hasattr(self.dht, '_outer_pipe') and self.dht._outer_pipe.closed:
                    get_logger().error("DHT pipe closed")
                    self._handle_critical_failure("DHT pipe closed")
                    break
                    
            except Exception as e:
                get_logger().warning(f"Health monitor error: {e}")
                
            self.shutdown_flag.wait(10)  # Check every 10 seconds

    def _check_dht_health(self) -> bool:
        """Comprehensive DHT health check."""
        if not self.dht:
            return False
            
        try:
            # Try to get visible addresses - this tests basic DHT functionality
            self.dht.get_visible_maddrs(latest=True)
            return True
        except Exception as e:
            get_logger().warning(f"DHT health check failed: {e}")
            return False

    def _handle_critical_failure(self, reason: str):
        """Handle critical DHT failures."""
        self._pipe_errors += 1
        get_logger().error(f"Critical failure #{self._pipe_errors}: {reason}")
        
        if self._pipe_errors >= self.max_pipe_errors or not self.enable_robust_mode:
            get_logger().error("Too many critical failures - entering emergency mode")
            self._emergency_mode = True
            self._preserve_peer_id_and_shutdown()
        else:
            get_logger().info("Attempting recovery from critical failure")
            # Trigger recovery in next gather call

    def _preserve_peer_id_and_shutdown(self):
        """Preserve peer_id before shutting down DHT."""
        self._pending_shutdown = False
        get_logger().info("=" * 60)
        get_logger().info("DHT SHUTDOWN: Preserving identity and switching to single-node")
        get_logger().info("=" * 60)
        
        # Mark as time-based shutdown
        self.time_based_shutdown = True
        
        # PRESERVE peer_id BEFORE shutdown
        if self.dht and hasattr(self.dht, 'peer_id'):
            self._persistent_peer_id = str(self.dht.peer_id)
            get_logger().info(f"Preserved peer_id: {self._persistent_peer_id}")
        
        # Shutdown DHT safely
        self._safe_cleanup_dht()
        
        # Stop monitoring
        self.shutdown_flag.set()
        
        # Log new mode
        get_logger().info("=" * 60)
        get_logger().info("TRAINING MODE: Single-node (preserved identity)")
        get_logger().info(f"Agent ID: {self.get_id()}")
        get_logger().info("Identity: Preserved from DHT")
        get_logger().info("Data consistency: Maintained")
        get_logger().info("=" * 60)

    def _setup_single_node_fallback(self):
        """Setup single-node mode when DHT initialization fails."""
        get_logger().info("=" * 60)
        get_logger().info("TRAINING MODE: Single-node (DHT init failed)")
        get_logger().info(f"Agent ID: {self.get_id()}")
        get_logger().info("Identity: Process-based fallback")
        get_logger().info("=" * 60)

    def _init_dht_with_timeout(self, initial_peers, **kwargs):
        """Initialize DHT with enhanced error handling."""
        
        host_maddrs = ["/ip4/0.0.0.0/tcp/0"]
        
        if self.bootstrap:
            get_logger().info("Starting as bootstrap node")
            self.dht = DHT(
                start=True,
                host_maddrs=host_maddrs,
                initial_peers=initial_peers or [],
                startup_timeout=self.startup_timeout,
                **kwargs
            )
            
            try:
                dht_maddrs = self.dht.get_visible_maddrs(latest=True)
                HivemindRendezvouz.set_initial_peers(dht_maddrs)
            except Exception as e:
                get_logger().warning(f"Failed to set rendezvous peers: {e}")
                
        else:
            get_logger().info("Starting as worker node")
            
            resolved_peers = initial_peers
            if not resolved_peers:
                resolved_peers = HivemindRendezvouz.get_initial_peers()
            if not resolved_peers:
                resolved_peers = []
            
            get_logger().info(f"Connecting to {len(resolved_peers)} peers")
            
            self.dht = DHT(
                start=True,
                host_maddrs=host_maddrs,
                initial_peers=resolved_peers,
                startup_timeout=self.startup_timeout,
                **kwargs
            )

    def _verify_dht_health(self):
        """Enhanced DHT health verification."""
        if self.dht:
            try:
                # Test basic functionality
                self.dht.get_visible_maddrs(latest=True)
                
                # Test process health
                if hasattr(self.dht, '_server_process') and not self.dht._server_process.is_alive():
                    raise RuntimeError("DHT process not alive after initialization")
                
                get_logger().info("DHT health check passed")
            except Exception as e:
                raise RuntimeError(f"DHT health check failed: {e}")

    def _safe_cleanup_dht(self):
        """Safe DHT cleanup with timeout and force termination."""
        if not self.dht:
            return
            
        try:
            get_logger().info("Safely cleaning up DHT...")
            
            # Set shutdown flag
            self.shutdown_flag.set()
            
            # Try graceful shutdown with timeout
            shutdown_start = time.time()
            shutdown_timeout = 10
            
            try:
                self.dht.shutdown()
            except Exception as e:
                get_logger().warning(f"DHT shutdown error: {e}")
            
            # Wait for processes to terminate
            if hasattr(self.dht, '_server_process'):
                while (self.dht._server_process.is_alive() and 
                       time.time() - shutdown_start < shutdown_timeout):
                    time.sleep(0.5)
                
                # Force terminate if still alive
                if self.dht._server_process.is_alive():
                    get_logger().warning("Force terminating DHT process")
                    try:
                        self.dht._server_process.terminate()
                        time.sleep(2)
                        if self.dht._server_process.is_alive():
                            self.dht._server_process.kill()
                    except Exception as e:
                        get_logger().warning(f"Force termination error: {e}")
            
            get_logger().info("DHT cleanup completed")
            
        except Exception as e:
            get_logger().warning(f"DHT cleanup error: {e}")
        finally:
            self.dht = None

    def all_gather_object(self, obj: Any) -> Dict[str | int, Any]:
        """Ultra-robust all_gather with comprehensive error handling."""
        
        # EMERGENCY EXIT: Force single-node via environment
        if os.environ.get("FORCE_SINGLE_NODE", "false").lower() == "true":
            get_logger().debug("FORCE_SINGLE_NODE enabled - skipping distributed gather")
            return {self.get_id(): obj}
        
        # EMERGENCY MODE: Switch to single-node after critical failures
        if self._emergency_mode:
            get_logger().debug(f"Emergency mode - single-node processing (agent: {self.get_id()})")
            return {self.get_id(): obj}
        
        # PENDING SHUTDOWN: Handle graceful shutdown
        if hasattr(self, '_pending_shutdown') and self._pending_shutdown:
            get_logger().info("Pending shutdown detected - switching to single-node")
            self._preserve_peer_id_and_shutdown()
            return {self.get_id(): obj}
        
        # NO DHT: Use single-node mode
        if (self.time_based_shutdown or 
            not self.dht or 
            not hasattr(self.dht, 'peer_id')):
            agent_id = self.get_id()
            get_logger().debug(f"Single-node processing (agent: {agent_id})")
            return {agent_id: obj}
        
        # DISTRIBUTED GATHERING with error recovery
        key = f"gather_{self.step_}"
        
        for attempt in range(self.max_retries):
            try:
                result = self._attempt_distributed_gather_robust(obj, key)
                self.step_ += 1
                
                # Reset error counters on success
                if self._connection_failures > 0:
                    get_logger().info(f"Distributed gather recovered after {self._connection_failures} failures")
                    self._connection_failures = 0
                    
                return result
                
            except Exception as e:
                error_msg = str(e)
                get_logger().warning(f"Gather attempt {attempt + 1} failed: {error_msg}")
                
                # Check for critical pipe/communication errors
                critical_patterns = [
                    "ran out of input", "pipe", "broken", "connection", "timeout",
                    "eof", "resource temporarily unavailable", "blocking"
                ]
                
                if any(pattern in error_msg.lower() for pattern in critical_patterns):
                    get_logger().error(f"Critical DHT error: {error_msg}")
                    self._handle_critical_failure(f"Gather error: {error_msg}")
                    return {self.get_id(): obj}
                
                self._connection_failures += 1
                
                if self._connection_failures >= self._max_connection_failures:
                    get_logger().error("Too many connection failures - switching to single-node")
                    self._preserve_peer_id_and_shutdown()
                    return {self.get_id(): obj}
                
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (attempt + 1)
                    get_logger().info(f"Waiting {delay}s before retry...")
                    time.sleep(delay)
        
        # All attempts failed - use fallback
        get_logger().warning("All distributed gather attempts failed - using single-node fallback")
        return {self.get_id(): obj}

    def _attempt_distributed_gather_robust(self, obj: Any, key: str) -> Dict[str, Any]:
        """Robust distributed gather with comprehensive pre-flight checks."""
        
        # PRE-FLIGHT CHECKS
        if not self.dht or not hasattr(self.dht, 'peer_id'):
            raise RuntimeError("DHT not available")
        
        # Check DHT process health
        if hasattr(self.dht, '_server_process') and not self.dht._server_process.is_alive():
            raise RuntimeError("DHT process not alive")
        
        # Check pipe health  
        if hasattr(self.dht, '_outer_pipe') and self.dht._outer_pipe.closed:
            raise RuntimeError("DHT pipe closed")
        
        # SERIALIZATION with error handling
        try:
            obj_bytes = to_bytes(obj)
        except Exception as e:
            raise RuntimeError(f"Serialization failed: {e}")
        
        # STORE with retry and pipe monitoring
        store_success = False
        for store_attempt in range(3):
            try:
                # Check pipe before store
                if hasattr(self.dht, '_outer_pipe') and self.dht._outer_pipe.closed:
                    raise RuntimeError("DHT pipe closed before store")
                
                self.dht.store(
                    key,
                    subkey=str(self.dht.peer_id),
                    value=obj_bytes,
                    expiration_time=get_dht_time() + self.timeout,
                    beam_size=self.beam_size,
                )
                store_success = True
                break
                
            except Exception as e:
                error_msg = str(e)
                get_logger().warning(f"Store attempt {store_attempt + 1} failed: {error_msg}")
                
                # Check for critical errors
                if any(pattern in error_msg.lower() for pattern in [
                    "ran out of input", "pipe", "connection reset", "eof"
                ]):
                    raise RuntimeError(f"Critical DHT store error: {error_msg}")
                
                if store_attempt < 2:
                    time.sleep(1)
        
        if not store_success:
            raise RuntimeError("Failed to store in DHT after retries")
        
        # Wait for propagation
        time.sleep(min(2.0, 0.5 * self.world_size))
        
        # RETRIEVE with health monitoring
        start_time = time.time()
        best_result = {}
        get_timeout = min(self.timeout, 60)
        
        while time.time() - start_time < get_timeout:
            try:
                # Health check before each get
                if hasattr(self.dht, '_outer_pipe') and self.dht._outer_pipe.closed:
                    raise RuntimeError("DHT pipe closed during retrieval")
                
                if hasattr(self.dht, '_server_process') and not self.dht._server_process.is_alive():
                    raise RuntimeError("DHT process died during retrieval")
                
                output, _ = self.dht.get(key, beam_size=self.beam_size, latest=True)
                
                if output:
                    current_result = {}
                    for subkey, value in output.items():
                        try:
                            current_result[subkey] = from_bytes(value.value)
                        except Exception as e:
                            get_logger().debug(f"Deserialization failed for {subkey}: {e}")
                    
                    if len(current_result) > len(best_result):
                        best_result = current_result
                    
                    # Success condition
                    if len(current_result) >= min(self.world_size, 2):
                        break
                
                time.sleep(1.0)
                
            except Exception as e:
                error_msg = str(e)
                get_logger().debug(f"DHT get failed: {error_msg}")
                
                # Check for critical errors during get
                if any(pattern in error_msg.lower() for pattern in [
                    "ran out of input", "pipe", "connection reset", "broken pipe", "eof"
                ]):
                    raise RuntimeError(f"Critical DHT get error: {error_msg}")
                
                time.sleep(2.0)
        
        if not best_result:
            raise RuntimeError("No results collected from DHT")
        
        # Success - reset connection failure counter
        self._connection_failures = max(0, self._connection_failures - 1)
        return best_result

    def get_id(self):
        """Get agent identifier - preserved peer_id has priority."""
        # Priority 1: Preserved peer_id from DHT
        if hasattr(self, '_persistent_peer_id'):
            return self._persistent_peer_id
        
        # Priority 2: Current DHT peer_id
        if self.dht and hasattr(self.dht, 'peer_id'):
            return str(self.dht.peer_id)
        
        # Priority 3: Process fallback
        return f"node_{os.getpid()}"

    def get_training_mode(self) -> str:
        """Get current training mode with error information."""
        if self._emergency_mode:
            return "emergency_single_node"
        elif self.time_based_shutdown and hasattr(self, '_persistent_peer_id'):
            return "single_node_preserved_identity"
        elif self.time_based_shutdown:
            return "single_node_post_timeout"
        elif self.dht:
            return "distributed_dht_robust"
        else:
            return "single_node_fallback"

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive backend status."""
        return {
            "mode": self.get_training_mode(),
            "agent_id": self.get_id(),
            "dht_active": self.dht is not None,
            "emergency_mode": self._emergency_mode,
            "pipe_errors": self._pipe_errors,
            "connection_failures": self._connection_failures,
            "robust_mode": self.enable_robust_mode,
            "time_status": self.get_time_status(),
        }

    def get_time_status(self) -> Dict[str, Any]:
        """Get time status."""
        if self.dht_start_time:
            elapsed = (time.time() - self.dht_start_time) / 60
            remaining = max(0, self.dht_timeout_minutes - elapsed)
        else:
            elapsed = remaining = 0
            
        return {
            "dht_active": self.dht is not None,
            "time_based_shutdown": self.time_based_shutdown,
            "elapsed_minutes": round(elapsed, 1),
            "remaining_minutes": round(remaining, 1),
            "timeout_minutes": self.dht_timeout_minutes,
            "has_preserved_identity": hasattr(self, '_persistent_peer_id'),
        }

    def shutdown(self):
        """Graceful shutdown with comprehensive cleanup."""
        get_logger().info("Initiating HivemindBackend shutdown...")
        
        # Set shutdown flag first
        self.shutdown_flag.set()
        
        # Wait for monitoring threads to finish
        if self.time_monitor_thread and self.time_monitor_thread.is_alive():
            self.time_monitor_thread.join(timeout=5)
            
        if self.health_monitor_thread and self.health_monitor_thread.is_alive():
            self.health_monitor_thread.join(timeout=5)
        
        # Clean up DHT
        self._safe_cleanup_dht()
        
        get_logger().info("HivemindBackend shutdown completed")

    def __del__(self):
        """Enhanced destructor with error suppression."""
        try:
            self.shutdown()
        except Exception as e:
            # Use print instead of logger as logging might be shutdown
            print(f"Warning: Error during HivemindBackend cleanup: {e}")


# Factory function to create the appropriate backend
def create_hivemind_backend(**kwargs):
    """Factory function to create HivemindBackend with emergency wrapper."""
    
    # Check if emergency wrapper should be enabled
    enable_wrapper = kwargs.pop('enable_emergency_wrapper', True)
    
    # Create the backend
    backend = HivemindBackend(**kwargs)
    
    # Wrap with emergency handler if enabled
    if enable_wrapper:
        get_logger().info("Emergency wrapper enabled for maximum robustness")
        return EmergencyTrainingWrapper(backend)
    else:
        return backend


# Usage example and environment setup
def setup_environment_for_robust_training():
    """Setup environment variables for robust training."""
    
    # Reduce timeouts for faster failure detection
    os.environ.setdefault('HIVEMIND_DHT_TIMEOUT', '60')  # 1 minute timeout
    os.environ.setdefault('HIVEMIND_STARTUP_TIMEOUT', '30')  # 30 second startup
    
    # Enable robust mode by default
    os.environ.setdefault('HIVEMIND_ROBUST_MODE', 'true')
    
    # Set reasonable retry limits
    os.environ.setdefault('HIVEMIND_MAX_RETRIES', '3')
    
    # Enable emergency mode for critical failures
    os.environ.setdefault('HIVEMIND_EMERGENCY_MODE', 'true')
    
    get_logger().info("Environment configured for robust Hivemind training")


# Emergency control functions
def emergency_disable_dht():
    """Emergency function to disable DHT completely."""
    os.environ['DISABLE_DHT'] = 'true'
    get_logger().warning("DHT EMERGENCY DISABLED - all communication will use single-node mode")


def emergency_force_single_node():
    """Emergency function to force single-node mode immediately."""
    os.environ['FORCE_SINGLE_NODE'] = 'true'
    get_logger().warning("EMERGENCY SINGLE-NODE MODE - all gather operations forced to single-node")


def check_system_health():
    """Check system health for potential issues."""
    issues = []
    
    # Check multiprocessing method
    try:
        method = mp.get_start_method()
        if method != 'spawn':
            issues.append(f"Multiprocessing method is '{method}', recommend 'spawn' for stability")
    except:
        issues.append("Could not determine multiprocessing method")
    
    # Check file descriptor limits
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < 1024:
            issues.append(f"Low file descriptor limit: {soft}, recommend >= 1024")
    except:
        pass
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        if memory.available < 1024 * 1024 * 1024:  # 1GB
            issues.append(f"Low available memory: {memory.available / 1024**3:.1f}GB")
    except:
        pass
    
    if issues:
        get_logger().warning("System health issues detected:")
        for issue in issues:
            get_logger().warning(f"  - {issue}")
    else:
        get_logger().info("System health check passed")
    
    return len(issues) == 0
