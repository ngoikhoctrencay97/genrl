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
    Robust Hivemind backend with persistent identity extraction and single-node optimization.
    Extracts agent ID from identity file then operates in stable single-node mode.
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
        **kwargs,
    ):
        # Initialize core attributes
        self.dht = None
        self.world_size = int(os.environ.get("HIVEMIND_WORLD_SIZE", 1))
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.beam_size = min(beam_size, 50)  # Cap beam size
        self.step_ = 0
        self._connection_failures = 0
        self._max_connection_failures = 5
        
        get_logger().info(f"Initializing HivemindBackend (world_size={self.world_size})")
        
        # Setup multiprocessing environment
        self._setup_multiprocessing()
        
        # DHT configuration - only safe parameters
        dht_kwargs = {
            "cache_locally": not disable_caching,
            "cache_on_store": False,
            "num_workers": 1,
            "daemon": False,
        }
        dht_kwargs.update(kwargs)
        
        self.bootstrap = self._is_bootstrap()
        self._init_dht_with_recovery(initial_peers, **dht_kwargs)

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

    def _get_persistent_identity(self, **kwargs):
        """Extract peer_id from identity file then shutdown DHT immediately."""
        try:
            get_logger().info("Extracting persistent identity from file...")
            
            # Setup timeout protection
            def timeout_handler(signum, frame):
                raise TimeoutError("Identity extraction timeout")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30 second timeout
            
            # Filter to only safe parameters for identity extraction
            safe_kwargs = {
                'identity_path': kwargs.get('identity_path'),
                'cache_locally': False,
                'cache_on_store': False
            }
            safe_kwargs = {k: v for k, v in safe_kwargs.items() if v is not None}
            
            get_logger().debug(f"Identity extraction kwargs: {safe_kwargs}")
            
            # Create temporary DHT just to extract peer_id
            temp_dht = DHT(
                start=True,
                host_maddrs=["/ip4/0.0.0.0/tcp/0"],
                initial_peers=[],  # Empty peers - no network needed
                startup_timeout=20,  # Short timeout
                **safe_kwargs
            )
            
            # Extract peer_id
            peer_id = str(temp_dht.peer_id)
            get_logger().info(f"Successfully extracted peer_id: {peer_id}")
            
            # Shutdown immediately to free resources
            temp_dht.shutdown()
            get_logger().debug("Temporary DHT shutdown completed")
            
            # Clear timeout
            signal.alarm(0)
            
            return peer_id
            
        except Exception as e:
            # Clear timeout on any error
            signal.alarm(0)
            get_logger().warning(f"Failed to extract identity: {e}")
            return None
    
    def _init_dht_with_recovery(self, initial_peers: List[str] | None, **kwargs):
        """Initialize DHT with persistent identity extraction and comprehensive error recovery."""
        
        # Try to extract persistent identity first
        if 'identity_path' in kwargs and kwargs['identity_path']:
            identity_path = kwargs['identity_path']
            get_logger().info(f"Identity file configured: {identity_path}")
            
            if os.path.exists(identity_path):
                get_logger().info("Identity file exists, extracting persistent peer ID...")
                persistent_id = self._get_persistent_identity(**kwargs)
                
                if persistent_id:
                    self._persistent_peer_id = persistent_id
                    
                    # Log successful single-node setup
                    get_logger().info("=" * 60)
                    get_logger().info("TRAINING MODE: Single-node with persistent identity")
                    get_logger().info(f"Agent ID: {persistent_id}")
                    get_logger().info(f"Identity file: {identity_path}")
                    get_logger().info("Memory optimization: DHT networking disabled")
                    get_logger().info("Data consistency: Maintained with existing runs")
                    get_logger().info("Performance: Optimized for single-node stability")
                    get_logger().info("=" * 60)
                    return
                else:
                    get_logger().warning("Failed to extract identity, falling back to DHT mode")
            else:
                get_logger().warning(f"Identity file not found: {identity_path}")
        else:
            get_logger().info("No identity_path configured, attempting DHT mode")
        
        # Fallback to distributed DHT mode
        get_logger().info("Attempting distributed DHT initialization...")
        
        for attempt in range(self.max_retries):
            try:
                get_logger().info(f"DHT initialization attempt {attempt + 1}/{self.max_retries}")
                
                # Cleanup previous attempt
                if self.dht:
                    self._cleanup_dht()
                
                # Progressive delay
                if attempt > 0:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    get_logger().info(f"Waiting {delay}s before retry...")
                    time.sleep(delay)
                
                # Try initialization with timeout
                self._init_dht_with_timeout(initial_peers, **kwargs)
                
                # Health check
                self._verify_dht_health()
                
                get_logger().info("=" * 60)
                get_logger().info("TRAINING MODE: Distributed DHT")
                get_logger().info(f"Agent ID: {self.dht.peer_id}")
                get_logger().info(f"Connected peers: {len(initial_peers) if initial_peers else 0}")
                get_logger().info("Memory usage: Higher due to DHT networking")
                get_logger().info("Performance: Distributed synchronization active")
                get_logger().info("=" * 60)
                
                self._connection_failures = 0
                return
                
            except Exception as e:
                get_logger().error(f"DHT init attempt {attempt + 1} failed: {e}")
                self._cleanup_dht()
                
                if attempt == self.max_retries - 1:
                    get_logger().error("All DHT initialization attempts failed")
                    get_logger().info("=" * 60)
                    get_logger().info("TRAINING MODE: Single-node fallback (no persistent identity)")
                    get_logger().info(f"Agent ID: Will use node_{os.getpid()}")
                    get_logger().info("Data consistency: WARNING - New agent ID will be generated")
                    get_logger().info("Memory optimization: DHT disabled")
                    get_logger().info("Performance: Single-node mode")
                    get_logger().info("=" * 60)
                    return

    def _init_dht_with_timeout(self, initial_peers, **kwargs):
        """Initialize DHT with timeout protection."""
        
        def timeout_handler(signum, frame):
            raise TimeoutError("DHT initialization timeout")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.startup_timeout)
        
        try:
            host_maddrs = ["/ip4/0.0.0.0/tcp/0"]  # Simplified - only TCP
            
            if self.bootstrap:
                get_logger().info("Starting as bootstrap node")
                self.dht = DHT(
                    start=True,
                    host_maddrs=host_maddrs,
                    initial_peers=initial_peers or [],
                    startup_timeout=self.startup_timeout,
                    **kwargs
                )
                
                # Set rendezvous peers
                try:
                    dht_maddrs = self.dht.get_visible_maddrs(latest=True)
                    HivemindRendezvouz.set_initial_peers(dht_maddrs)
                except Exception as e:
                    get_logger().warning(f"Failed to set rendezvous peers: {e}")
                    
            else:
                get_logger().info("Starting as worker node")
                
                # Get peers from rendezvous or use provided
                resolved_peers = initial_peers
                if not resolved_peers:
                    resolved_peers = HivemindRendezvouz.get_initial_peers()
                    
                if not resolved_peers:
                    get_logger().warning("No peers found - using empty peer list")
                    resolved_peers = []
                
                get_logger().info(f"Connecting to {len(resolved_peers)} peers")
                
                self.dht = DHT(
                    start=True,
                    host_maddrs=host_maddrs,
                    initial_peers=resolved_peers,
                    startup_timeout=self.startup_timeout,
                    **kwargs
                )
                
        finally:
            signal.alarm(0)

    def _verify_dht_health(self):
        """Quick health check for DHT."""
        if not self.dht:
            return
            
        try:
            # Simple connectivity test
            self.dht.get_visible_maddrs(latest=True)
            get_logger().info("DHT health check passed")
        except Exception as e:
            get_logger().warning(f"DHT health check failed: {e}")
            # Don't raise - DHT might still work

    def _cleanup_dht(self):
        """Safely cleanup DHT resources."""
        if self.dht:
            try:
                get_logger().debug("Cleaning up DHT...")
                self.dht.shutdown()
                time.sleep(1)  # Allow cleanup
            except Exception as e:
                get_logger().warning(f"DHT cleanup warning: {e}")
            finally:
                self.dht = None

    def all_gather_object(self, obj: Any) -> Dict[str | int, Any]:
        """
        Robust all_gather with fallback to single-node operation.
        """
        
        # Single-node mode
        if not self.dht:
            agent_id = self.get_id()
            get_logger().debug(f"Single-node processing (agent: {agent_id})")
            return {agent_id: obj}
        
        # Distributed mode
        get_logger().debug(f"Distributed mode: Gathering from {self.world_size} nodes")
        key = f"gather_{self.step_}"
        
        # Try distributed gather
        for attempt in range(self.max_retries):
            try:
                result = self._attempt_distributed_gather(obj, key)
                self.step_ += 1
                return result
                
            except Exception as e:
                get_logger().warning(f"Gather attempt {attempt + 1} failed: {e}")
                self._connection_failures += 1
                
                # If too many failures, disable DHT
                if self._connection_failures >= self._max_connection_failures:
                    get_logger().error("Too many DHT failures, switching to single-node mode")
                    self._cleanup_dht()
                    return {self.get_id(): obj}
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        
        # All attempts failed - fallback to single node
        get_logger().warning("All gather attempts failed, using single-node fallback")
        return {self.get_id(): obj}

    def _attempt_distributed_gather(self, obj: Any, key: str) -> Dict[str, Any]:
        """Single attempt at distributed gathering."""
        
        # Serialize
        try:
            obj_bytes = to_bytes(obj)
        except Exception as e:
            raise RuntimeError(f"Serialization failed: {e}")
        
        # Store
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
        
        # Wait for propagation
        time.sleep(min(2.0, 0.5 * self.world_size))
        
        # Collect results
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
                    
                    # Success if we have enough results
                    if len(current_result) >= min(self.world_size, 2):
                        break
                
                time.sleep(1.0)
                
            except Exception as e:
                get_logger().debug(f"DHT get failed: {e}")
                time.sleep(2.0)
        
        if not best_result:
            raise RuntimeError("No results collected")
        
        get_logger().debug(f"Collected {len(best_result)}/{self.world_size} objects")
        self._connection_failures = max(0, self._connection_failures - 1)  # Reduce failure count on success
        
        return best_result

    def get_id(self):
        """Get agent identifier with persistent identity priority."""
        # Priority 1: Use extracted persistent ID if available
        if hasattr(self, '_persistent_peer_id'):
            return self._persistent_peer_id
        
        # Priority 2: Use DHT peer_id if running
        if self.dht and hasattr(self.dht, 'peer_id'):
            return str(self.dht.peer_id)
        
        # Priority 3: Final fallback
        return f"node_{os.getpid()}"

    def get_training_mode(self) -> str:
        """Get current training mode for monitoring."""
        if hasattr(self, '_persistent_peer_id'):
            return "single_node_persistent"
        elif self.dht:
            return "distributed_dht"
        else:
            return "single_node_fallback"

    def is_persistent_identity(self) -> bool:
        """Check if using persistent identity."""
        return hasattr(self, '_persistent_peer_id')

    def shutdown(self):
        """Graceful shutdown."""
        get_logger().info("Shutting down HivemindBackend...")
        self._cleanup_dht()

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.shutdown()
        except:
            pass
