import os
import pickle
import time
import threading
import multiprocessing as mp
from typing import Any, Dict, List

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


class HivemindBackend(Communication):
    """
    Simple time-based DHT backend with peer_id preservation.
    DHT runs for specified time, then switches to single-node with same identity.
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
        dht_timeout_minutes: int = 5,  # 5 minutes for testing
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
        
        # Time management
        self.dht_timeout_minutes = dht_timeout_minutes
        self.dht_start_time = None
        self.time_based_shutdown = False
        self.time_monitor_thread = None
        self.shutdown_flag = threading.Event()
        
        # Store for later use
        self.initial_peers = initial_peers
        
        get_logger().info(f"Initializing HivemindBackend (world_size={self.world_size})")
        get_logger().info(f"DHT timeout: {dht_timeout_minutes} minutes")
        
        self._setup_multiprocessing()
        
        # DHT configuration
        dht_kwargs = {
            "cache_locally": not disable_caching,
            "cache_on_store": False,
            "num_workers": 1,
            "daemon": False,
        }
        dht_kwargs.update(kwargs)
        
        self.bootstrap = self._is_bootstrap()
        self._init_dht_or_fallback(initial_peers, **dht_kwargs)

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
        """Try DHT initialization, fallback to single-node if failed."""
        
        try:
            self._init_dht_with_recovery(initial_peers, **kwargs)
        except Exception as e:
            get_logger().error(f"DHT initialization failed: {e}")
            self._setup_single_node_fallback()

    def _init_dht_with_recovery(self, initial_peers: List[str] | None, **kwargs):
        """Initialize DHT with retry logic."""
        
        for attempt in range(self.max_retries):
            try:
                get_logger().info(f"DHT initialization attempt {attempt + 1}/{self.max_retries}")
                
                if self.dht:
                    self._cleanup_dht()
                
                if attempt > 0:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    get_logger().info(f"Waiting {delay}s before retry...")
                    time.sleep(delay)
                
                self._init_dht_with_timeout(initial_peers, **kwargs)
                self._verify_dht_health()
                
                # Success - start time monitoring
                self.dht_start_time = time.time()
                self._start_time_monitoring()
                
                get_logger().info("=" * 60)
                get_logger().info("TRAINING MODE: Distributed DHT")
                get_logger().info(f"Agent ID: {self.dht.peer_id}")
                get_logger().info(f"DHT timeout: {self.dht_timeout_minutes} minutes")
                get_logger().info("Will preserve peer_id when switching to single-node")
                get_logger().info("=" * 60)
                
                self._connection_failures = 0
                return
                
            except Exception as e:
                get_logger().error(f"DHT attempt {attempt + 1} failed: {e}")
                if self.dht:
                    self._cleanup_dht()
                
                if attempt == self.max_retries - 1:
                    raise RuntimeError("All DHT attempts failed")

    def _start_time_monitoring(self):
        """Start time monitoring thread."""
        if self.time_monitor_thread and self.time_monitor_thread.is_alive():
            return
            
        self.shutdown_flag.clear()
        self.time_monitor_thread = threading.Thread(target=self._time_monitor_loop, daemon=True)
        self.time_monitor_thread.start()

    def _time_monitor_loop(self):
        """Monitor DHT runtime and shutdown after timeout."""
        last_log_minute = 0
        
        while not self.shutdown_flag.is_set() and self.dht and not self.time_based_shutdown:
            if self.dht_start_time:
                elapsed_minutes = (time.time() - self.dht_start_time) / 60
                
                if elapsed_minutes >= self.dht_timeout_minutes:
                    get_logger().info(f"DHT timeout reached: {elapsed_minutes:.1f} minutes")
                    self._preserve_peer_id_and_shutdown()
                    break
                
                # Log every minute for testing (was 30 min)
                current_minute = int(elapsed_minutes)
                if current_minute % 10 == 0 and current_minute > last_log_minute and current_minute > 0:
                    remaining = self.dht_timeout_minutes - elapsed_minutes
                    get_logger().info(f"DHT runtime: {elapsed_minutes:.1f}min, {remaining:.1f}min remaining")
                    last_log_minute = current_minute
            
            self.shutdown_flag.wait(30)  # Check every 30 seconds for testing

    def _preserve_peer_id_and_shutdown(self):
        """Preserve peer_id before shutting down DHT."""
        get_logger().info("=" * 60)
        get_logger().info("DHT TIMEOUT: Preserving identity and switching to single-node")
        get_logger().info("=" * 60)
        
        # PRESERVE peer_id BEFORE shutdown
        if self.dht and hasattr(self.dht, 'peer_id'):
            self._persistent_peer_id = str(self.dht.peer_id)
            get_logger().info(f"Preserved peer_id: {self._persistent_peer_id}")
        
        # Mark as time-based shutdown
        self.time_based_shutdown = True
        
        # Shutdown DHT
        if self.dht:
            try:
                self.dht.shutdown()
                time.sleep(2)
                get_logger().info("DHT shutdown completed")
            except Exception as e:
                get_logger().error(f"Error during shutdown: {e}")
            finally:
                self.dht = None
        
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
        """Initialize DHT - removed signal handling."""
        
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
        """Quick DHT health check."""
        if self.dht:
            try:
                self.dht.get_visible_maddrs(latest=True)
                get_logger().info("DHT health check passed")
            except Exception as e:
                get_logger().warning(f"DHT health check failed: {e}")

    def _cleanup_dht(self):
        """Clean up DHT resources."""
        self.shutdown_flag.set()
        
        if self.dht:
            try:
                self.dht.shutdown()
                time.sleep(1)
            except Exception as e:
                get_logger().warning(f"DHT cleanup warning: {e}")
            finally:
                self.dht = None

    def all_gather_object(self, obj: Any) -> Dict[str | int, Any]:
        """Gather objects with time-aware fallback."""
        
        if self.time_based_shutdown or not self.dht:
            agent_id = self.get_id()
            get_logger().debug(f"Single-node processing (agent: {agent_id})")
            return {agent_id: obj}
        
        key = f"gather_{self.step_}"
        
        for attempt in range(self.max_retries):
            try:
                result = self._attempt_distributed_gather(obj, key)
                self.step_ += 1
                return result
                
            except Exception as e:
                get_logger().warning(f"Gather attempt {attempt + 1} failed: {e}")
                self._connection_failures += 1
                
                if self._connection_failures >= self._max_connection_failures:
                    get_logger().error("Too many failures, switching to single-node")
                    self._preserve_peer_id_and_shutdown()
                    return {self.get_id(): obj}
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        
        return {self.get_id(): obj}

    def _attempt_distributed_gather(self, obj: Any, key: str) -> Dict[str, Any]:
        """Single distributed gather attempt."""
        
        try:
            obj_bytes = to_bytes(obj)
        except Exception as e:
            raise RuntimeError(f"Serialization failed: {e}")
        
        try:
            self.dht.store(
                key,
                subkey=str(self.dht.peer_id),
                value=obj_bytes,
                expiration_time=get_dht_time() + self.timeout,
                beam_size=self.beam_size,
            )
        except Exception as e:
            raise RuntimeError(f"DHT store failed: {e}")
        
        time.sleep(min(2.0, 0.5 * self.world_size))
        
        start_time = time.time()
        best_result = {}
        
        while time.time() - start_time < min(self.timeout, 60):
            try:
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
                    
                    if len(current_result) >= min(self.world_size, 2):
                        break
                
                time.sleep(1.0)
                
            except Exception as e:
                get_logger().debug(f"DHT get failed: {e}")
                time.sleep(2.0)
        
        if not best_result:
            raise RuntimeError("No results collected")
        
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
        """Get current training mode."""
        if self.time_based_shutdown and hasattr(self, '_persistent_peer_id'):
            return "single_node_preserved_identity"
        elif self.time_based_shutdown:
            return "single_node_post_timeout"
        elif self.dht:
            return "distributed_dht"
        else:
            return "single_node_fallback"

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
        """Graceful shutdown."""
        self.shutdown_flag.set()
        self._cleanup_dht()
        
        if self.time_monitor_thread and self.time_monitor_thread.is_alive():
            self.time_monitor_thread.join(timeout=5)

    def __del__(self):
        try:
            self.shutdown()
        except:
            pass
