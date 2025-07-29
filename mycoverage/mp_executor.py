import json
import logging
import multiprocessing
import multiprocessing as mp
import os
import sys
import time

from mycoverage import tracer
from util.util import ExecutionStatus, wrap_code_with_device

# By default execute in cpu mode
RUN_CPU_MODE = True
test_executor = None
cov_executor = None


def init_test_executor(args, cov=False):
    global test_executor
    global cov_executor
    kwargs = {
        "close_fd_mask": getattr(args, "close_fd_mask", 0),
        "debug": getattr(args, "debug", False)
    }

    if args.library == "torch":
        if cov and (cov_executor is None):
            cov_executor = PyTorchCoverageExecutor(**kwargs)
        if test_executor is None:
            test_executor = PyTorchExecutor(**kwargs)

    elif args.library == "tf":
        # This setup is fine as it's correctly guarded by the library check.
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        from util.util import set_memory_growth
        set_memory_growth()

        kwargs["cpu"] = RUN_CPU_MODE
        if cov and (cov_executor is None):
            cov_executor = TensorFlowCoverageExecutor(**kwargs)
        if test_executor is None:
            test_executor = TensorFlowExecutor(**kwargs)


def kill_executors():
    if test_executor is not None:
        test_executor.kill()
    if cov_executor is not None:
        cov_executor.kill()


# --- Library-specific execution functions ---

def exec_func_generic(filename, execGlobals=None):
    """A generic exec function that assumes the file handles its own imports."""
    with open(filename, "r") as f:
        code = f.read()
    # Execute in a clean scope to avoid conflicts.
    # __name__ being __main__ is important for some scripts.
    scope = execGlobals if execGlobals is not None else {}
    scope['__name__'] = '__main__'
    exec(code, scope)


def exec_func_tf_cpu(filename, execGlobals=None):
    """TF-specific exec function that sets the device context."""
    import tensorflow as tf
    with open(filename, "r") as f:
        code = f.read()
    scope = execGlobals if execGlobals is not None else {}
    scope.update({'__name__': '__main__', 'tf': tf})
    with tf.device("cpu"):
        exec(code, scope)


# --- Library-specific worker functions ---

def worker_torch(target, child_conn, close_fd_mask):
    """Worker specifically for PyTorch tests."""
    logging.captureWarnings(True)
    logging.getLogger().setLevel(logging.CRITICAL)
    # Don't redirect if running in an interactive terminal, for easier debugging
    if not (sys.stdout.isatty() and sys.stderr.isatty()):
        f = open(os.devnull, "w")
        if close_fd_mask & 1: sys.stdout = f
        if close_fd_mask & 2: sys.stderr = f

    while True:
        try:
            buf = child_conn.recv_bytes().decode("utf-8")
            target(buf) # target is exec_func_generic
            child_conn.send_bytes("ok".encode("utf-8"))
        except EOFError:
            break # Parent closed the pipe
        except Exception as e:
            message = {"exception": type(e).__name__, "msg": str(e)}
            child_conn.send_bytes(json.dumps(message).encode("utf-8"))


def worker_tf(target, child_conn, close_fd_mask):
    """Worker specifically for TensorFlow tests."""
    import tensorflow as tf
    logging.captureWarnings(True)
    logging.getLogger().setLevel(logging.CRITICAL)
    if not (sys.stdout.isatty() and sys.stderr.isatty()):
        f = open(os.devnull, "w")
        if close_fd_mask & 1: sys.stdout = f
        if close_fd_mask & 2: sys.stderr = f

    while True:
        try:
            buf = child_conn.recv_bytes().decode("utf-8")
            target(buf) # target is exec_func_generic or exec_func_tf_cpu
            child_conn.send_bytes("ok".encode("utf-8"))
        except EOFError:
            break
        except Exception as e:
            message = {"exception": type(e).__name__, "msg": str(e)}
            child_conn.send_bytes(json.dumps(message).encode("utf-8"))


def cov_worker_torch(target, child_conn, close_fd_mask):
    logging.captureWarnings(True)
    logging.getLogger().setLevel(logging.CRITICAL)
    if not (sys.stdout.isatty() and sys.stderr.isatty()):
        f = open(os.devnull, "w")
        if close_fd_mask & 1: sys.stdout = f
        if close_fd_mask & 2: sys.stderr = f

    sys.settrace(tracer.trace_torch)
    while True:
        try:
            buf = child_conn.recv_bytes().decode("utf-8")
            target(buf)
            child_conn.send_bytes(b"%d" % tracer.get_coverage())
        except EOFError:
            break
        except Exception as e:
            message = {"exception": type(e).__name__, "msg": str(e)}
            child_conn.send_bytes(json.dumps(message).encode("utf-8"))


def cov_worker_tf(target, child_conn, close_fd_mask):
    logging.captureWarnings(True)
    logging.getLogger().setLevel(logging.CRITICAL)
    if not (sys.stdout.isatty() and sys.stderr.isatty()):
        f = open(os.devnull, "w")
        if close_fd_mask & 1: sys.stdout = f
        if close_fd_mask & 2: sys.stderr = f

    sys.settrace(tracer.trace_tf)
    while True:
        try:
            buf = child_conn.recv_bytes().decode("utf-8")
            target(buf)
            child_conn.send_bytes(b"%d" % tracer.get_coverage())
        except EOFError:
            break
        except Exception as e:
            message = {"exception": type(e).__name__, "msg": str(e)}
            child_conn.send_bytes(json.dumps(message).encode("utf-8"))


class Executor:
    def __init__(self, worker, exec_func, single_test_timeout=10, close_fd_mask=0, debug=False):
        self.worker = worker
        self._exec_func = exec_func
        self._close_fd_mask = close_fd_mask
        self._timeout = single_test_timeout
        self.debug = debug
        self._p = None # Initialize process attribute
        self._test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_programs")
        self.restart() # Start the process during initialization

    def run_test(self, filename) -> (str, bool):
        if not self._p.is_alive():
            if self.debug: print("Worker process was dead. Restarting.")
            self.restart()

        self.parent_conn.send_bytes(filename.encode("utf-8"))

        if not self.parent_conn.poll(self._timeout):
            print("=================================================================")
            print(f"Timeout reached ({self._timeout}s). Restarting worker.")
            self.restart()
            return "timeout", False

        try:
            return_bytes = self.parent_conn.recv_bytes()
            message = return_bytes.decode("utf-8")
            valid = message == "ok"
            return message, valid
        except EOFError:
            return "crash", False # Pipe closed unexpectedly
        except Exception as e:
            return (f"Error: {type(e).__name__} - {e}", False)

    def terminate(self):
        if self._p and self._p.is_alive():
            self._p.terminate()

    def kill(self):
        if self._p and self._p.is_alive():
            self._p.kill()

    def restart(self):
        if hasattr(self, "_p") and self._p and self._p.is_alive():
            self._p.kill()
        ctx = multiprocessing.get_context("spawn")
        self.parent_conn, self.child_conn = ctx.Pipe()
        self._p = ctx.Process(
            target=self.worker,
            args=(self._exec_func, self.child_conn, self._close_fd_mask),
        )
        self._p.start()

    def check_if_internal_state_break(self):
        status, valid = self.run_test(self.check_filename)
        if status != "ok":
            self.restart()
        return status, valid


class CoverageExecutor(Executor):
    def __init__(self, worker, exec_func, single_test_timeout=10, **kwargs):
        super().__init__(worker, exec_func, single_test_timeout, **kwargs)
        self.prev_coverage = 0
        if self.debug: print("Init cov executor")

    def run_test(self, filename) -> (str, bool):
        if not self._p.is_alive():
            if self.debug: print("Coverage worker process was dead. Restarting.")
            self.restart()

        self.parent_conn.send_bytes(filename.encode("utf-8"))

        if not self.parent_conn.poll(self._timeout):
            print("=================================================================")
            print(f"Timeout reached during coverage collection ({self._timeout}s). Restarting worker.")
            self.restart()
            return "timeout", False
        try:
            return_bytes = self.parent_conn.recv_bytes()
            total_coverage = int(return_bytes)
            new_coverage = False
            if self.debug: print(f"Cov: {self.prev_coverage} -> {total_coverage}")
            if total_coverage > self.prev_coverage:
                new_coverage = True
            self.prev_coverage = total_coverage
            return "ok", new_coverage
        except ValueError:
            msg = return_bytes.decode("utf-8")
            return msg, False
        except EOFError:
            return "crash", False


class TensorFlowExecutor(Executor):
    def __init__(self, worker=worker_tf, single_test_timeout=10, cpu=False, **kwargs):
        exec_func_to_use = exec_func_tf_cpu if cpu else exec_func_generic
        super().__init__(worker, exec_func_to_use, single_test_timeout, **kwargs)
        if self.debug: print("--- init tf executor: set_memory_growth ---")
        self.check_filename = os.path.join(self._test_dir, "check_tf_state.py")
        self.run_test(self.check_filename) # Initial check

class TensorFlowCoverageExecutor(CoverageExecutor):
    def __init__(self, worker=cov_worker_tf, single_test_timeout=10, cpu=False, **kwargs):
        exec_func_to_use = exec_func_tf_cpu if cpu else exec_func_generic
        super().__init__(worker, exec_func_to_use, single_test_timeout, **kwargs)
        assert tracer.trace_library is None or tracer.trace_library == "tf"
        tracer.trace_library = "tf"
        self.check_filename = os.path.join(self._test_dir, "check_tf_state.py")
        self.run_test(self.check_filename)


class PyTorchExecutor(Executor):
    def __init__(self, single_test_timeout=10, **kwargs):
        super().__init__(
            worker=worker_torch,
            exec_func=exec_func_generic,
            single_test_timeout=single_test_timeout,
            **kwargs
        )
        self.check_filename = os.path.join(self._test_dir, "check_torch_state.py")
        self.run_test(self.check_filename) # Initial check

class PyTorchCoverageExecutor(CoverageExecutor):
    def __init__(self, worker=cov_worker_torch, single_test_timeout=10, **kwargs):
        super().__init__(
            worker=worker,
            exec_func=exec_func_generic,
            single_test_timeout=single_test_timeout,
            **kwargs
        )
        assert tracer.trace_library is None or tracer.trace_library == "torch"
        tracer.trace_library = "torch"
        if self.debug: print("Initialized torch cov")
        self.check_filename = os.path.join(self._test_dir, "check_torch_state.py")
        self.run_test(self.check_filename)


def coverate_run_status_mp(
    g_code, library, cov_executor, device="cpu"
) -> (ExecutionStatus, bool):
    """
    Returns status and whether has new coverage
    """
    CURRENT_TIME = time.time()
    tmp_filename = f"/tmp/tmp{CURRENT_TIME}.py"
    write_code = wrap_code_with_device(g_code, library, device)
    with open(tmp_filename, "w") as f:
        f.write(write_code)

    status, new_coverage = cov_executor.run_test(tmp_filename)

    if status == "ok":
        return ExecutionStatus.SUCCESS, new_coverage
    else:
        if "timeout" in status:
            return ExecutionStatus.TIMEOUT, new_coverage
        elif "exception" in status:
            return ExecutionStatus.EXCEPTION, new_coverage
        elif "crash" in status or "Error" in status:
            return ExecutionStatus.CRASH, new_coverage
        else: # Handle other unexpected messages
            return ExecutionStatus.EXCEPTION, new_coverage