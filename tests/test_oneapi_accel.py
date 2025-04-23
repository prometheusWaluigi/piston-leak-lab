"""
Test oneAPI SYCL device detection and GPU acceleration (Intel Arc GPU).

- Lists available SYCL devices
- Runs a simple vector addition on the Intel Arc GPU using numba-dpex
- Verifies correctness and prints device info/results
"""

import numpy as np

try:
    import dpctl
    import dpctl.tensor as dpt
    from numba import njit
    from numba_dpex import kernel, DEFAULT_LOCAL_SIZE
    oneapi_available = True
except ImportError:
    print("[FAIL] oneAPI Python packages (dpctl, numba-dpex) not installed.")
    oneapi_available = False


def list_sycl_devices():
    print("Available SYCL devices:")
    for dev in dpctl.get_devices():
        print(f"  - {dev}")

def arc_gpu_device():
    # Try to find an Intel Arc GPU device
    for dev in dpctl.get_devices():
        if "gpu" in dev.device_type.lower() and "intel" in dev.vendor.lower() and "arc" in dev.name.lower():
            return dev
    return None

if oneapi_available:
    list_sycl_devices()
    arc_dev = arc_gpu_device()
    if arc_dev:
        print(f"[OK] Intel Arc GPU found: {arc_dev}")
        # Test vector addition kernel
        N = 1024 * 1024
        a = np.arange(N, dtype=np.float32)
        b = np.arange(N, dtype=np.float32)
        c = np.zeros_like(a)

        @kernel
        def vadd(a, b, c):
            i = numba_dpex.get_global_id(0)
            if i < a.shape[0]:
                c[i] = a[i] + b[i]

        # Transfer to device
        with dpctl.device_context(arc_dev):
            d_a = dpt.asarray(a)
            d_b = dpt.asarray(b)
            d_c = dpt.zeros_like(d_a)
            vadd[d_a.shape[0]//DEFAULT_LOCAL_SIZE + 1, DEFAULT_LOCAL_SIZE](d_a, d_b, d_c)
            c_result = d_c.asnumpy()

        # Check result
        if np.allclose(c_result, a + b):
            print("[OK] oneAPI vector addition succeeded on Intel Arc GPU!")
        else:
            print("[FAIL] oneAPI vector addition produced incorrect result.")
    else:
        print("[FAIL] No Intel Arc GPU found with oneAPI SYCL.")
else:
    print("[FAIL] oneAPI stack not available in this environment.")
