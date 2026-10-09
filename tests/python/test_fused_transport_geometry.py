"""Cold queue-pool expansion must not alter timed native/Put geometry."""
from pathlib import Path
import shutil
import subprocess
import tempfile


def test_fused_transport_geometry():
    root = Path(__file__).resolve().parents[2]
    compiler = shutil.which("c++")
    if not compiler:
        raise RuntimeError("C++ compiler required for transport geometry contract")
    with tempfile.TemporaryDirectory() as task:
        source = Path(task) / "geometry.cc"
        source.write_text('''
#include "kvaf/access_operator/access_operator.h"
#include <cassert>
int main() {
  kvaf::AccessOperatorConfig legacy{"get", "put", 12, 1};
  assert(legacy.GetTransportCoreCount() == 12);
  kvaf::AccessOperatorConfig expanded{"get", "put", 12, 1, 24};
  assert(expanded.GetTransportCoreCount() == 24);
  assert(expanded.npu_core_count == 12 && expanded.jettys_per_core == 1);
  assert(expanded.batch_put_kernel_object_path == "put");
}
''')
        binary = Path(task) / "geometry"
        subprocess.run([compiler, "-std=c++17", "-I", str(root / "include"),
                        "-I", str(root / "access_operator/include"), str(source), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)
    text = (root / "access_operator/src/access_operator.cc").read_text()
    begin = text.index("Status DefaultAccessOperator::Initialize(")
    end = text.index("Status DefaultAccessOperator::", begin + 10)
    initialization = text[begin:end]
    assert "impl_->config.GetTransportCoreCount(), impl_->config.jettys_per_core, &transport" in initialization
    assert "static_cast<std::size_t>(impl_->config.GetTransportCoreCount())" in initialization
    assert "KvafBatchPutTransportCreate(" in initialization
    # Timed Get argument construction and Put launch use the native count,
    # not the larger cold pool. Extra queues never add timed native workers.
    assert "exec, impl_->config.npu_core_count, impl_->config.jettys_per_core," in text
    assert "arguments.aiv_core_count = impl_->config.npu_core_count;" in text
    print("KVAF_FUSED_TRANSPORT_GEOMETRY pass=1 native=12 fused_pool=24 put=12")


if __name__ == "__main__":
    test_fused_transport_geometry()
