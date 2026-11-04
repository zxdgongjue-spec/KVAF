## 新建容器并编译KVAF流程

下面统一使用：

| 项目 | 设置 |
|---|---|
| 宿主机用户 | `w50065599` |
| 新容器 | `dsa-offload-wjj-priority` |
| 新代码目录 | `/home/w50065599/kvaf-tests_priority/KVAF` |
| CPU 服务端口 | `29000` |
| 打流 NPU | `DEVICE_ID=0`，与上次保持一致 |

分支名还没有提供，下面的 `KVAF_BRANCH` 需要替换为你要测试的实际分支。

**1. 在宿主机，以 `w50065599` 登录并配置代理。**

```bash
whoami
# 应输出 w50065599

export http_proxy='http://<实际代理地址>:<端口>'
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$http_proxy"
export ALL_PROXY="$http_proxy"

export NO_PROXY='127.0.0.1,localhost,local,.local,10.*,100.*,141.1.64.249'
export no_proxy="$NO_PROXY"
```

列出远端分支：

```bash
git ls-remote --heads https://gitcode.com/CacheOS/KVAF.git
```

输出中的 `refs/heads/xxx` 表示分支名是 `xxx`。

**2. 将目标分支拉到新目录。**

```bash
KVAF_BRANCH='替换成实际分支名'

mkdir -p /home/w50065599/kvaf-tests_priority

git clone --branch "$KVAF_BRANCH" --single-branch \
    https://gitcode.com/CacheOS/KVAF.git \
    /home/w50065599/kvaf-tests_priority/KVAF

cd /home/w50065599/kvaf-tests_priority/KVAF

git branch --show-current
git log -1 --oneline
```

`--branch` 会检出指定分支，`--single-branch` 限定本次克隆的分支范围。[Git 官方说明](https://git-scm.com/docs/git-clone)

保存版本信息，并检查测试入口：

```bash
mkdir -p logs
git show -s --format=fuller HEAD | tee logs/source_version.txt

ls tests/run_remote_cpu_batch_get_miss_count_perf.sh
```

新代码放在独立的父目录中，默认构建所用的 `../third_party/umdk` 也会与旧目录分开。

**3. 在宿主机创建新容器。**

复制管理员的脚本：

```bash
cp /home/w50065599/start_container.sh \
   /home/w50065599/start_container_priority_test.sh

vi /home/w50065599/start_container_priority_test.sh
```

修改副本：

- 将容器名改成 `--name dsa-offload-wjj-priority`。
- `/usr/local/sbin/npu-smi` 的重复挂载行只保留一条。
- 镜像、设备挂载和 `/home/` 挂载保持管理员给出的配置。

然后运行：

```bash
bash /home/w50065599/start_container_priority_test.sh

docker ps --filter name=dsa-offload-wjj-priority
```

因为挂载了 `/home/`，新容器内会直接看到刚拉取的代码。

**4. 进入新容器，传入代理并重新编译。**

在刚才配置过代理的宿主机终端执行：

```bash
docker exec -it \
    -e http_proxy="$http_proxy" \
    -e https_proxy="$https_proxy" \
    -e HTTP_PROXY="$HTTP_PROXY" \
    -e HTTPS_PROXY="$HTTPS_PROXY" \
    -e ALL_PROXY="$ALL_PROXY" \
    -e NO_PROXY="$NO_PROXY" \
    -e no_proxy="$no_proxy" \
    -w /home/w50065599/kvaf-tests_priority/KVAF \
    dsa-offload-wjj-priority bash
```

以下命令在**容器内**执行：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest

npu-smi info

set -o pipefail

./build.sh cpu 2>&1 | tee logs/build_cpu.log
```

确认 CPU 构建成功，再执行：

```bash
./build.sh npu 2>&1 | tee logs/build_npu.log
```

检查产物：

```bash
ls -l build-cpu/runtime/kvaf_cpu_service
ls -l build-npu/bin/kvaf_execute_batch_get_kernels.o
ls -l build-npu/bin/kvaf_execute_batch_put_kernels.o
ls -ld build-npu/python/kvaf
```

这里使用的是全新克隆目录生成的构建产物。

**5. 在终端 A 启动新版本的 CPU 服务。**

仍在新容器、新代码目录中执行：

```bash
./build-cpu/runtime/kvaf_cpu_service \
    --listen 0.0.0.0 \
    --port 29000 \
    --ub-device udmac0d1e2 \
    --eid-index 0 \
    2>&1 | tee logs/cpu_service.log
```

看到 `LISTEN ... port=29000` 后保持这个终端运行。

使用 `29000` 是为了明确连接这次新启动的服务。`--net=host` 会共享宿主机端口，所以新容器名本身不会隔离旧服务使用的 `28999`。[Docker 官方说明](https://docs.docker.com/engine/network/drivers/host/)

**6. 在终端 B 进入同一个新容器并打流。**

另开一个宿主机终端：

```bash
docker exec -it \
    -w /home/w50065599/kvaf-tests_priority/KVAF \
    dsa-offload-wjj-priority bash
```

容器内执行：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

unset KVAF_PEER_DEVICE KVAF_NPU_EIDS
set -o pipefail

KVAF_CPU_SERVICE_ADDR=127.0.0.1:29000 \
DEVICE_ID=7 \
tests/run_remote_cpu_batch_get_miss_count_perf.sh "$PWD/build-npu" \
    2>&1 | tee logs/npu_read_cpu.log

echo "exit_code=$?"
```


如果新分支跑通，可以说明**这套新代码与新构建环境能够工作**。