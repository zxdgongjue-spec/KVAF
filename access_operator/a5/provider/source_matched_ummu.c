#define _GNU_SOURCE

#include "kvaf_source_matched_ummu.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef EUCLEAN
#define EUCLEAN 117
#endif

/* Keep the source-built object portable; this is stronger than the AArch64
 * device barriers used by upstream and avoids a second architecture variant. */
#define __UMMU_BARRIER__ 1
static inline void ummu_to_device_wmb(void)
{
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
}
static inline void ummu_from_device_rmb(void)
{
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

#include "ummu_api.h"
#include "ummu_common.h"
#include "ummu_log.h"
#include "ummu_map.h"
#include "ummu_mapt.h"
#include "ummu_queue.h"
#include "ummu_resource.h"
#include "ummu_seg_mng.h"

/*
 * Do not infer this ABI from the older drivers/ubus header.  The selected
 * 2025 MAPT source and the running A5 driver require the complete UAPI with
 * hw_cap, PLBI ioctl commands and the three advertised capability bits.
 */
#if !defined(KVAF_UMMU_FULL_UAPI_VERIFIED) || \
    KVAF_UMMU_FULL_UAPI_VERIFIED != 1
#error "KVAF requires the full A5 UMMU UAPI header matching the running driver"
#endif

static inline uint32_t kvaf_ummu_uapi_hw_cap(
    const struct ummu_tid_info *info)
{
  return info->hw_cap &
      (HW_CAP_EXPAN | HW_CAP_DOUBLE_PLBI | HW_CAP_KCMD_PLBI);
}

#define KVAF_UMMU_QUEUE_WAIT_COUNT UINT16_C(300)
#define KVAF_UMMU_QUEUE_WAIT_US UINT16_C(25)
#define KVAF_UMMU_CPL_WAIT_COUNT UINT16_C(4)
#define KVAF_UMMU_CPL_WAIT_US UINT16_C(2)

enum kvaf_ummu_cleanup_phase {
  KVAF_UMMU_CLEANUP_ACTIVE = 0,
  KVAF_UMMU_CLEANUP_PLBI_ALL_DONE,
  KVAF_UMMU_CLEANUP_KERNEL_TID_FREED,
  KVAF_UMMU_CLEANUP_LOCAL_RELEASED
};

struct kvaf_ummu_retained_mapping {
  void *address;
  size_t length;
  struct kvaf_ummu_retained_mapping *next;
};

struct kvaf_ummu_tid_state {
  uint32_t tid;
  enum ummu_mapt_mode mode;
  struct ummu_mapt_info *mapt;
  enum kvaf_source_ummu_pending_operation pending;
  enum kvaf_ummu_cleanup_phase cleanup_phase;
  uint64_t pending_address;
  size_t pending_length;
  uint32_t pending_token;
  uint32_t plbi_next_ummu;
  bool pending_by_token;
  bool pending_local_update_done;
  bool pending_state_unknown;
  bool allocation_orphan;
  bool map_inserted;
  struct kvaf_ummu_retained_mapping *retained;
  struct kvaf_ummu_tid_state *next;
};

static pthread_mutex_t g_kvaf_ummu_init_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_kvaf_ummu_status_lock = PTHREAD_MUTEX_INITIALIZER;
static struct ummu_ctx_info *g_kvaf_ummu_ctx;
static struct kvaf_ummu_tid_state *g_kvaf_ummu_states;
static pid_t g_kvaf_ummu_owner_pid;
static uint32_t g_kvaf_ummu_live_tid_count;
static uint32_t g_kvaf_ummu_pending_count;
static uint32_t g_kvaf_ummu_retained_count;
static struct kvaf_source_ummu_status g_kvaf_ummu_last_status;
static _Thread_local struct kvaf_ummu_tid_state *g_kvaf_ummu_current_state;
static _Thread_local enum kvaf_source_ummu_failure_stage g_kvaf_queue_failure_stage;
static _Thread_local bool g_kvaf_queue_state_unknown;

ummu_vlog_level_t g_loglevel = UMMU_VLOG_LEVEL_WARNING;

struct ummu_ctx_info *get_ummu_ctx(void)
{
  return g_kvaf_ummu_ctx;
}

static int kvaf_negative_errno(int status)
{
  if (status < 0) {
    return status;
  }
  if (status > 0) {
    return -status;
  }
  return errno == 0 ? -EIO : -errno;
}

static void kvaf_record_failure(
    const struct kvaf_ummu_tid_state *state,
    enum kvaf_source_ummu_failure_stage stage, int status,
    enum kvaf_source_ummu_retryability retryability,
    bool state_unknown,
    enum kvaf_source_ummu_pending_operation operation,
    uint64_t address, size_t length)
{
  if (pthread_mutex_lock(&g_kvaf_ummu_status_lock) != 0) {
    return;
  }
  g_kvaf_ummu_last_status.abi_version =
      KVAF_SOURCE_UMMU_STATUS_ABI_VERSION;
  g_kvaf_ummu_last_status.structure_size =
      sizeof(g_kvaf_ummu_last_status);
  g_kvaf_ummu_last_status.capability_magic =
      KVAF_SOURCE_UMMU_CAPABILITY_MAGIC;
  g_kvaf_ummu_last_status.capability_bits =
      KVAF_SOURCE_UMMU_CHECKED_RETRYABLE_V1 |
      KVAF_SOURCE_UMMU_PINNED_FULL_UAPI;
  g_kvaf_ummu_last_status.last_failure_stage = (uint32_t)stage;
  g_kvaf_ummu_last_status.last_error = kvaf_negative_errno(status);
  g_kvaf_ummu_last_status.last_retryability = (uint32_t)retryability;
  g_kvaf_ummu_last_status.last_state_unknown = state_unknown ? 1U : 0U;
  g_kvaf_ummu_last_status.last_tid = state == NULL ? 0U : state->tid;
  g_kvaf_ummu_last_status.last_pending_operation = (uint32_t)operation;
  g_kvaf_ummu_last_status.last_address = address;
  g_kvaf_ummu_last_status.last_length = length;
  (void)pthread_mutex_unlock(&g_kvaf_ummu_status_lock);
}

static void kvaf_set_pending(
    struct kvaf_ummu_tid_state *state,
    enum kvaf_source_ummu_pending_operation pending)
{
  if (state->pending == KVAF_SOURCE_UMMU_PENDING_NONE &&
      pending != KVAF_SOURCE_UMMU_PENDING_NONE) {
    __atomic_add_fetch(&g_kvaf_ummu_pending_count, 1U, __ATOMIC_RELAXED);
  } else if (state->pending != KVAF_SOURCE_UMMU_PENDING_NONE &&
             pending == KVAF_SOURCE_UMMU_PENDING_NONE) {
    __atomic_sub_fetch(&g_kvaf_ummu_pending_count, 1U, __ATOMIC_RELAXED);
  }
  state->pending = pending;
}

static void kvaf_clear_pending(struct kvaf_ummu_tid_state *state)
{
  kvaf_set_pending(state, KVAF_SOURCE_UMMU_PENDING_NONE);
  state->pending_address = 0;
  state->pending_length = 0;
  state->pending_token = 0;
  state->plbi_next_ummu = 0;
  state->pending_by_token = false;
  state->pending_local_update_done = false;
  state->pending_state_unknown = false;
}

static struct kvaf_ummu_tid_state *kvaf_find_state_locked(uint32_t tid)
{
  struct kvaf_ummu_tid_state *state;

  for (state = g_kvaf_ummu_states; state != NULL; state = state->next) {
    if (state->tid == tid) {
      return state;
    }
  }
  return NULL;
}

static void kvaf_add_state_locked(struct kvaf_ummu_tid_state *state)
{
  state->next = g_kvaf_ummu_states;
  g_kvaf_ummu_states = state;
  ++g_kvaf_ummu_ctx->tid_cnt;
  __atomic_add_fetch(&g_kvaf_ummu_live_tid_count, 1U, __ATOMIC_RELAXED);
}

static void kvaf_remove_state_locked(struct kvaf_ummu_tid_state *state)
{
  struct kvaf_ummu_tid_state **cursor = &g_kvaf_ummu_states;

  while (*cursor != NULL) {
    if (*cursor == state) {
      *cursor = state->next;
      --g_kvaf_ummu_ctx->tid_cnt;
      __atomic_sub_fetch(
          &g_kvaf_ummu_live_tid_count, 1U, __ATOMIC_RELAXED);
      return;
    }
    cursor = &(*cursor)->next;
  }
}

static int kvaf_ensure_context(void)
{
  struct ummu_ctx_info *context;
  int status = 0;

  if (pthread_mutex_lock(&g_kvaf_ummu_init_lock) != 0) {
    return -EDEADLK;
  }
  if (g_kvaf_ummu_ctx != NULL) {
    if (g_kvaf_ummu_owner_pid != getpid()) {
      status = -ECHILD;
      kvaf_record_failure(NULL, KVAF_SOURCE_UMMU_FAILURE_FORKED_PROCESS,
          status, KVAF_SOURCE_UMMU_RETRY_UNSAFE, true,
          KVAF_SOURCE_UMMU_PENDING_NONE, 0, 0);
    }
    (void)pthread_mutex_unlock(&g_kvaf_ummu_init_lock);
    return status;
  }

  context = calloc(1, sizeof(*context));
  if (context == NULL) {
    status = -ENOMEM;
    goto error;
  }
  context->shared_fd = -1;
  if (pthread_mutex_init(&context->ctx_mutex, NULL) != 0) {
    status = -EDEADLK;
    goto free_context;
  }
  context->mapt_map = ummu_map_create();
  if (context->mapt_map == NULL) {
    status = -ENOMEM;
    goto destroy_mutex;
  }
  context->shared_fd = open(TID_DEVICE_NAME, O_RDWR | O_CLOEXEC);
  if (context->shared_fd < 0) {
    status = -errno;
    goto clear_map;
  }
  g_kvaf_ummu_ctx = context;
  g_kvaf_ummu_owner_pid = getpid();
  (void)pthread_mutex_unlock(&g_kvaf_ummu_init_lock);
  return 0;

clear_map:
  ummu_map_clear(context->mapt_map, NULL);
destroy_mutex:
  (void)pthread_mutex_destroy(&context->ctx_mutex);
free_context:
  free(context);
error:
  kvaf_record_failure(NULL, KVAF_SOURCE_UMMU_FAILURE_INIT, status,
      KVAF_SOURCE_UMMU_RETRY_SAFE, false,
      KVAF_SOURCE_UMMU_PENDING_NONE, 0, 0);
  (void)pthread_mutex_unlock(&g_kvaf_ummu_init_lock);
  return status;
}

static int kvaf_retain_mapping(
    struct kvaf_ummu_tid_state *state, void *address, size_t length)
{
  struct kvaf_ummu_retained_mapping *mapping;

  if (state == NULL || address == NULL || length == 0) {
    return -EINVAL;
  }
  for (mapping = state->retained; mapping != NULL;
       mapping = mapping->next) {
    if (mapping->address == address && mapping->length == length) {
      return 0;
    }
  }
  mapping = calloc(1, sizeof(*mapping));
  if (mapping == NULL) {
    return -ENOMEM;
  }
  mapping->address = address;
  mapping->length = length;
  mapping->next = state->retained;
  state->retained = mapping;
  __atomic_add_fetch(&g_kvaf_ummu_retained_count, 1U, __ATOMIC_RELAXED);
  return 0;
}

static int kvaf_checked_munmap(
    struct kvaf_ummu_tid_state *state, void *address, size_t length)
{
  int status;

  if (address == NULL || length == 0) {
    return -EINVAL;
  }
  if (munmap(address, length) == 0) {
    return 0;
  }
  status = -errno;
  (void)kvaf_retain_mapping(state, address, length);
  kvaf_record_failure(state, KVAF_SOURCE_UMMU_FAILURE_MUNMAP,
      status, KVAF_SOURCE_UMMU_RETRY_SAFE, false,
      state == NULL ? KVAF_SOURCE_UMMU_PENDING_NONE : state->pending,
      (uint64_t)(uintptr_t)address, length);
  return status;
}

static int kvaf_retry_retained_mappings(struct kvaf_ummu_tid_state *state)
{
  struct kvaf_ummu_retained_mapping **cursor = &state->retained;
  struct kvaf_ummu_retained_mapping *mapping;

  while (*cursor != NULL) {
    mapping = *cursor;
    if (munmap(mapping->address, mapping->length) != 0) {
      kvaf_record_failure(state, KVAF_SOURCE_UMMU_FAILURE_MUNMAP,
          -errno, KVAF_SOURCE_UMMU_RETRY_SAFE, false, state->pending,
          (uint64_t)(uintptr_t)mapping->address, mapping->length);
      return -errno;
    }
    *cursor = mapping->next;
    __atomic_sub_fetch(&g_kvaf_ummu_retained_count, 1U, __ATOMIC_RELAXED);
    free(mapping);
  }
  return 0;
}

void *ummu_get_core_buf(
    unsigned int tid, enum ummu_buf_mode mode, size_t size, int fd,
    unsigned long index)
{
  off_t offset;

  if (fd < 0 || size == 0) {
    errno = EINVAL;
    return MAP_FAILED;
  }
  switch (mode) {
    case BASE_MODE_ENTRY_BLOCK:
      offset = (off_t)UMMU_OFFSET_BLK(0UL, tid);
      break;
    case BASE_MODE_TABLE_BLOCK:
      if (size < BLOCK_SIZE_16K || size > BLOCK_SIZE_2M) {
        errno = EINVAL;
        return MAP_FAILED;
      }
      offset = (off_t)UMMU_OFFSET_BLK(index, tid);
      break;
    case BASE_MODE_QUEUE:
      offset = (off_t)UMMU_OFFSET_QUE(tid);
      break;
    default:
      errno = EINVAL;
      return MAP_FAILED;
  }
  return mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, offset);
}

void ummu_free_core_buf(enum ummu_buf_mode mode, void *buffer, size_t size)
{
  size_t length = mode == BASE_MODE_ENTRY_BLOCK ? BLOCK_SIZE_4K : size;

  if (kvaf_checked_munmap(g_kvaf_ummu_current_state, buffer, length) != 0 &&
      g_kvaf_ummu_current_state == NULL) {
    kvaf_record_failure(NULL, KVAF_SOURCE_UMMU_FAILURE_INTERNAL,
        -EUCLEAN, KVAF_SOURCE_UMMU_RETRY_UNSAFE, true,
        KVAF_SOURCE_UMMU_PENDING_NONE,
        (uint64_t)(uintptr_t)buffer, length);
  }
}

int ummu_get_tid(int fd, struct ummu_tid_info *info)
{
  if (fd < 0 || info == NULL) {
    return -EINVAL;
  }
  info->dev = ~0UL;
  return ioctl(fd, UMMU_IOCALLOC_TID, info) == 0 ? 0 : -errno;
}

void ummu_put_tid(int fd, unsigned int tid)
{
  struct ummu_tid_info info = {.tid = tid};

  if (fd < 0 || ioctl(fd, UMMU_IOCFREE_TID, &info) != 0) {
    kvaf_record_failure(g_kvaf_ummu_current_state,
        KVAF_SOURCE_UMMU_FAILURE_FREE_TID_IOCTL,
        fd < 0 ? -EINVAL : -errno, KVAF_SOURCE_UMMU_RETRY_SAFE, false,
        KVAF_SOURCE_UMMU_PENDING_FREE_TID, 0, 0);
  }
}

void ummu_kcmd_plbi(
    int fd, struct ummu_tid_info *info, uint32_t opcode)
{
  int status;

  if (fd < 0 || info == NULL) {
    status = -EINVAL;
  } else if (opcode == PLBI_VA_OP) {
    status = ioctl(fd, UMMU_IOCPLBI_VA, info) == 0 ? 0 : -errno;
  } else if (opcode == PLBI_ALL_OP) {
    status = ioctl(fd, UMMU_IOCPLBI_ALL, info) == 0 ? 0 : -errno;
  } else {
    status = -EINVAL;
  }
  if (status != 0) {
    kvaf_record_failure(g_kvaf_ummu_current_state,
        KVAF_SOURCE_UMMU_FAILURE_PLBI_KERNEL_IOCTL, status,
        KVAF_SOURCE_UMMU_RETRY_UNSAFE, true,
        g_kvaf_ummu_current_state == NULL
            ? KVAF_SOURCE_UMMU_PENDING_NONE
            : g_kvaf_ummu_current_state->pending,
        0, 0);
  }
}

bool queue_has_space(
    uint16_t producer, uint16_t consumer, uint16_t queue_size,
    uint16_t needed)
{
  uint16_t space;
  const uint16_t consumer_index = Q_IDX(consumer, queue_size);
  const uint16_t producer_index = Q_IDX(producer, queue_size);

  if (Q_WRAP(producer, queue_size) == Q_WRAP(consumer, queue_size)) {
    space = ((uint16_t)1U << queue_size) -
        (producer_index - consumer_index);
  } else {
    space = consumer_index - producer_index;
  }
  return space >= needed;
}

uint16_t ummu_update_queue_index(uint16_t index, uint16_t queue_size)
{
  const uint16_t index_mask = (uint16_t)MAX_CMD_QUE_INDEX(queue_size);
  uint16_t value = index & index_mask;
  uint16_t wrap = index >> queue_size;

  if (value == index_mask) {
    wrap ^= 1U;
    value = 0;
  } else {
    ++value;
  }
  return (uint16_t)((wrap << queue_size) | value);
}

int ummu_queue_create(
    struct ummu_mapt_info *mapt_info, struct ummu_tid_info *info, int fd)
{
  struct ummu_mapt_queue_args *queues;
  unsigned char *cursor;
  void *mapping;
  size_t bytes_per_queue;
  size_t queue_bytes;
  uint32_t index;

  if (mapt_info == NULL || info == NULL || fd < 0) {
    return -EINVAL;
  }
  if (mapt_info->kcmd_plbi) {
    return 0;
  }
  if (info->ummu_cnt == 0 || info->pcmdq_order > 15 ||
      info->pcplq_order > 15) {
    return -EINVAL;
  }
  bytes_per_queue = PAGE_SIZE *
      (1UL + (1UL << info->pcmdq_order) +
       (1UL << info->pcplq_order));
  if (bytes_per_queue > SIZE_MAX / info->ummu_cnt) {
    return -EOVERFLOW;
  }
  queue_bytes = bytes_per_queue * info->ummu_cnt;
  queues = calloc(info->ummu_cnt, sizeof(*queues));
  if (queues == NULL) {
    return -ENOMEM;
  }
  mapping = ummu_get_core_buf(
      mapt_info->tid, BASE_MODE_QUEUE, queue_bytes, fd, 0);
  if (CORE_BUF_CHECK_INVALID(mapping)) {
    free(queues);
    return errno == 0 ? -ENOMEM : -errno;
  }

  cursor = mapping;
  for (index = 0; index < info->ummu_cnt; ++index) {
    queues[index].cmd_entry = (struct ummu_mapt_cmd_entry *)cursor;
    queues[index].cpl_entry = (struct ummu_mapt_cpl_entry *)(
        cursor + PAGE_SIZE * (1UL << info->pcmdq_order));
    queues[index].ctrl_page = (struct ummu_mapt_ctrl_pages *)(
        cursor + PAGE_SIZE *
            ((1UL << info->pcmdq_order) +
             (1UL << info->pcplq_order)));
    cursor += bytes_per_queue;
  }
  mapt_info->queue_ctx.queue = queues;
  mapt_info->queue_ctx.ummu_cnt = info->ummu_cnt;
  mapt_info->queue_ctx.queue_size = queue_bytes;
  mapt_info->queue_ctx.cmd_que_pi_size =
      info->pcmdq_order + (uint8_t)PAGE_SHIFT - CMD_ENTRY_SHIFT;
  mapt_info->queue_ctx.cpl_que_size =
      info->pcplq_order + (uint8_t)PAGE_SHIFT - CPL_ENTRY_SHIFT;
  return 0;
}

int ummu_get_cpl_psync(
    struct ummu_mapt_info *mapt_info,
    struct ummu_mapt_queue_args *queue, uint16_t expected_command_index)
{
  struct ummu_mapt_cpl_entry *completion;
  uint16_t producer;
  uint16_t consumer;
  uint16_t mask;
  int first_error = 0;

  if (mapt_info == NULL || queue == NULL || queue->ctrl_page == NULL) {
    return -EINVAL;
  }
  producer = queue->ctrl_page->ucplq_pi;
  consumer = queue->ctrl_page->ucplq_ci;
  ummu_from_device_rmb();
  if (producer == consumer) {
    return -EAGAIN;
  }
  mask = (uint16_t)MAX_CMD_QUE_INDEX(
      mapt_info->queue_ctx.cpl_que_size);
  while (producer != consumer) {
    completion = queue->cpl_entry + (consumer & mask);
    if (completion->cpl_status != CPL_CMD_PSYNC_SUCC &&
        first_error == 0) {
      first_error = -EIO;
    }
    consumer = ummu_update_queue_index(
        consumer, mapt_info->queue_ctx.cpl_que_size);
    if (completion->cmdq_ci == expected_command_index) {
      queue->ctrl_page->ucplq_ci = consumer;
      return first_error;
    }
  }
  queue->ctrl_page->ucplq_ci = consumer;
  return first_error == 0 ? -EAGAIN : first_error;
}

int ummu_send_cmd(
    struct ummu_mapt_cmd_entry *plbi,
    struct ummu_mapt_info *mapt_info, uint32_t ummu_index)
{
  const struct ummu_mapt_cmd_entry psync = {
      .opcode = PSYNC_OP, .cqe_en = 1};
  struct ummu_mapt_queue_args *queue;
  struct ummu_mapt_ctrl_pages *control;
  uint32_t plbi_count;
  uint16_t producer;
  uint16_t expected = 0;
  uint16_t wait_count;
  int status = -EAGAIN;

  g_kvaf_queue_failure_stage = KVAF_SOURCE_UMMU_FAILURE_NONE;
  g_kvaf_queue_state_unknown = false;
  if (plbi == NULL || mapt_info == NULL ||
      ummu_index >= mapt_info->queue_ctx.ummu_cnt ||
      mapt_info->queue_ctx.queue == NULL) {
    return -EINVAL;
  }
  queue = &mapt_info->queue_ctx.queue[ummu_index];
  control = queue->ctrl_page;
  plbi_count = mapt_info->double_plbi ? 2U : 1U;
  wait_count = KVAF_UMMU_QUEUE_WAIT_COUNT;
  while (!queue_has_space(control->ucmdq_pi, control->ucmdq_ci,
             mapt_info->queue_ctx.cmd_que_pi_size,
             (uint16_t)(CMD_NUM_PER_PLBI * plbi_count))) {
    if (wait_count-- == 0) {
      g_kvaf_queue_failure_stage =
          KVAF_SOURCE_UMMU_FAILURE_PLBI_QUEUE_FULL;
      return -EAGAIN;
    }
    (void)usleep(KVAF_UMMU_QUEUE_WAIT_US);
  }

  producer = control->ucmdq_pi;
  for (uint32_t index = 0; index < plbi_count; ++index) {
    memcpy(queue->cmd_entry +
               Q_IDX(producer, mapt_info->queue_ctx.cmd_que_pi_size),
        plbi, sizeof(*plbi));
    producer = ummu_update_queue_index(
        producer, mapt_info->queue_ctx.cmd_que_pi_size);
    expected = producer;
    memcpy(queue->cmd_entry +
               Q_IDX(producer, mapt_info->queue_ctx.cmd_que_pi_size),
        &psync, sizeof(psync));
    producer = ummu_update_queue_index(
        producer, mapt_info->queue_ctx.cmd_que_pi_size);
  }
  ummu_to_device_wmb();
  control->ucmdq_pi = producer;

  wait_count = KVAF_UMMU_CPL_WAIT_COUNT;
  while (wait_count-- > 0) {
    status = ummu_get_cpl_psync(mapt_info, queue, expected);
    if (status == 0 || status == -EIO) {
      break;
    }
    (void)usleep(KVAF_UMMU_CPL_WAIT_US);
  }
  if (status == -EIO) {
    g_kvaf_queue_failure_stage =
        KVAF_SOURCE_UMMU_FAILURE_PLBI_COMPLETION;
    return status;
  }
  if (status != 0) {
    g_kvaf_queue_failure_stage =
        KVAF_SOURCE_UMMU_FAILURE_PLBI_PSYNC_TIMEOUT;
    g_kvaf_queue_state_unknown = true;
    return -ETIMEDOUT;
  }
  return 0;
}

static uint32_t kvaf_plbi_range(size_t length)
{
  uint32_t range = 0;
  size_t value = length == 0 ? 0 : length - 1U;

  while (value != 0) {
    value >>= 1U;
    ++range;
  }
  return range;
}

static int kvaf_send_kcmd_plbi_checked(
    struct kvaf_ummu_tid_state *state, uint32_t opcode,
    uint64_t address, size_t length)
{
  struct ummu_tid_info info = {
      .tid = state->tid,
      .va = address,
      .size = length,
  };
  bool retry_safe;
  int status;

  if (g_kvaf_ummu_ctx == NULL || g_kvaf_ummu_ctx->shared_fd < 0) {
    status = -ENODEV;
  } else if (opcode == PLBI_VA_OP) {
    status = ioctl(g_kvaf_ummu_ctx->shared_fd,
        UMMU_IOCPLBI_VA, &info) == 0 ? 0 : -errno;
  } else if (opcode == PLBI_ALL_OP) {
    status = ioctl(g_kvaf_ummu_ctx->shared_fd,
        UMMU_IOCPLBI_ALL, &info) == 0 ? 0 : -errno;
  } else {
    status = -EINVAL;
  }
  if (status == 0) {
    return 0;
  }
  retry_safe = status == -EINTR || status == -EAGAIN ||
      status == -EBUSY || status == -ENOSPC;
  state->pending_state_unknown = !retry_safe;
  kvaf_record_failure(state,
      KVAF_SOURCE_UMMU_FAILURE_PLBI_KERNEL_IOCTL, status,
      retry_safe ? KVAF_SOURCE_UMMU_RETRY_SAFE
                 : KVAF_SOURCE_UMMU_RETRY_UNSAFE,
      !retry_safe, state->pending, address, length);
  return status;
}

static int kvaf_send_plbi_checked(
    struct kvaf_ummu_tid_state *state, uint32_t opcode,
    uint64_t address, size_t length)
{
  struct ummu_mapt_cmd_entry command = {0};
  uint32_t index;
  int status;

  if (state == NULL || state->mapt == NULL) {
    return -EINVAL;
  }
  if (state->mapt->kcmd_plbi) {
    return kvaf_send_kcmd_plbi_checked(
        state, opcode, address, length);
  }
  command.opcode = opcode;
  if (opcode == PLBI_VA_OP) {
    command.addr = address;
    command.range = kvaf_plbi_range(length);
  }
  for (index = state->plbi_next_ummu;
       index < state->mapt->queue_ctx.ummu_cnt; ++index) {
    status = ummu_send_cmd(&command, state->mapt, index);
    if (status != 0) {
      const bool unknown = g_kvaf_queue_state_unknown;
      state->plbi_next_ummu = index;
      state->pending_state_unknown = unknown;
      kvaf_record_failure(state,
          g_kvaf_queue_failure_stage == KVAF_SOURCE_UMMU_FAILURE_NONE
              ? KVAF_SOURCE_UMMU_FAILURE_INTERNAL
              : g_kvaf_queue_failure_stage,
          status,
          unknown ? KVAF_SOURCE_UMMU_RETRY_UNSAFE
                  : KVAF_SOURCE_UMMU_RETRY_SAFE,
          unknown, state->pending, address, length);
      return status;
    }
    state->plbi_next_ummu = index + 1U;
  }
  state->plbi_next_ummu = 0;
  return 0;
}

void ummu_plbi_va_cmd(
    struct ummu_mapt_info *mapt_info, struct ummu_data_info *data_info)
{
  (void)mapt_info;
  if (g_kvaf_ummu_current_state == NULL || data_info == NULL) {
    kvaf_record_failure(NULL, KVAF_SOURCE_UMMU_FAILURE_INTERNAL,
        -EINVAL, KVAF_SOURCE_UMMU_RETRY_UNSAFE, true,
        KVAF_SOURCE_UMMU_PENDING_NONE, 0, 0);
    return;
  }
  (void)kvaf_send_plbi_checked(g_kvaf_ummu_current_state,
      PLBI_VA_OP, data_info->data_base, data_info->data_size);
}

void ummu_send_user_all(struct ummu_mapt_info *mapt_info)
{
  (void)mapt_info;
  if (g_kvaf_ummu_current_state != NULL) {
    (void)kvaf_send_plbi_checked(
        g_kvaf_ummu_current_state, PLBI_ALL_OP, 0, 0);
  }
}

void ummu_queue_destroy(struct ummu_mapt_info *mapt_info)
{
  struct ummu_mapt_queue_args *queues;

  if (mapt_info == NULL || mapt_info->kcmd_plbi ||
      mapt_info->queue_ctx.queue == NULL) {
    return;
  }
  queues = mapt_info->queue_ctx.queue;
  ummu_free_core_buf(BASE_MODE_QUEUE,
      queues[0].cmd_entry, mapt_info->queue_ctx.queue_size);
  free(queues);
  mapt_info->queue_ctx.queue = NULL;
}

#ifndef KVAF_UMMU_COMMON_SOURCE
#error "KVAF_UMMU_COMMON_SOURCE must name ummu_common.c"
#endif
#ifndef KVAF_UMMU_LEVEL_BLOCK_SOURCE
#error "KVAF_UMMU_LEVEL_BLOCK_SOURCE must name ummu_level_block.c"
#endif
#ifndef KVAF_UMMU_SEG_MNG_SOURCE
#error "KVAF_UMMU_SEG_MNG_SOURCE must name ummu_seg_mng.c"
#endif
#ifndef KVAF_UMMU_MAPT_SOURCE
#error "KVAF_UMMU_MAPT_SOURCE must name ummu_mapt.c"
#endif

#include KVAF_UMMU_COMMON_SOURCE
#include KVAF_UMMU_LEVEL_BLOCK_SOURCE
#include KVAF_UMMU_SEG_MNG_SOURCE

#define ummu_allocate_tid kvaf_legacy_ummu_allocate_tid
#define ummu_free_tid kvaf_legacy_ummu_free_tid
#define ummu_grant kvaf_legacy_ummu_grant
#define ummu_ungrant kvaf_legacy_ummu_ungrant
#define ummu_ungrant_by_token kvaf_legacy_ummu_ungrant_by_token
#define ummu_mapt_destroy kvaf_legacy_ummu_mapt_destroy
#include KVAF_UMMU_MAPT_SOURCE
#undef ummu_mapt_destroy
#undef ummu_ungrant_by_token
#undef ummu_ungrant
#undef ummu_grant
#undef ummu_free_tid
#undef ummu_allocate_tid

static int kvaf_pending_matches(
    const struct kvaf_ummu_tid_state *state,
    enum kvaf_source_ummu_pending_operation pending,
    uint64_t address, size_t length, bool by_token, uint32_t token)
{
  return state->pending == pending &&
      state->pending_address == address &&
      state->pending_length == length &&
      state->pending_by_token == by_token &&
      (!by_token || state->pending_token == token);
}

static int kvaf_retry_data_operation_locked(
    struct kvaf_ummu_tid_state *state)
{
  int status;

  if (state->pending == KVAF_SOURCE_UMMU_PENDING_NONE ||
      state->pending == KVAF_SOURCE_UMMU_PENDING_FREE_TID) {
    return -EINVAL;
  }
  if (!state->pending_local_update_done || state->pending_state_unknown) {
    kvaf_record_failure(state, KVAF_SOURCE_UMMU_FAILURE_INTERNAL,
        -EUCLEAN, KVAF_SOURCE_UMMU_RETRY_UNSAFE, true,
        state->pending, state->pending_address, state->pending_length);
    return -EUCLEAN;
  }
  status = kvaf_retry_retained_mappings(state);
  if (status != 0) {
    return status;
  }
  status = kvaf_send_plbi_checked(state, PLBI_VA_OP,
      state->pending_address, state->pending_length);
  if (status == 0) {
    kvaf_clear_pending(state);
  }
  return status;
}

static int kvaf_context_and_state_lock(
    uint32_t tid, struct ummu_ctx_info **context_out,
    struct kvaf_ummu_tid_state **state_out)
{
  struct ummu_ctx_info *context;
  struct kvaf_ummu_tid_state *state;
  int status;

  status = kvaf_ensure_context();
  if (status != 0) {
    return status;
  }
  context = g_kvaf_ummu_ctx;
  if (pthread_mutex_lock(&context->ctx_mutex) != 0) {
    return -EDEADLK;
  }
  state = kvaf_find_state_locked(tid);
  if (state == NULL || state->mapt == NULL || state->allocation_orphan) {
    (void)pthread_mutex_unlock(&context->ctx_mutex);
    return -ENOENT;
  }
  if (pthread_mutex_lock(&state->mapt->mapt_mutex) != 0) {
    (void)pthread_mutex_unlock(&context->ctx_mutex);
    return -EDEADLK;
  }
  *context_out = context;
  *state_out = state;
  g_kvaf_ummu_current_state = state;
  return 0;
}

static void kvaf_context_and_state_unlock(
    struct ummu_ctx_info *context, struct kvaf_ummu_tid_state *state)
{
  g_kvaf_ummu_current_state = NULL;
  (void)pthread_mutex_unlock(&state->mapt->mapt_mutex);
  (void)pthread_mutex_unlock(&context->ctx_mutex);
}

int ummu_grant(
    uint32_t tid, void *data, size_t data_size,
    enum ummu_mapt_perm permission, struct ummu_seg_attr *segment_attribute)
{
  struct ummu_ctx_info *context;
  struct kvaf_ummu_tid_state *state;
  struct ummu_data_info data_info = {0};
  enum ummu_grant_op_type operation;
  int status;

  if (segment_attribute == NULL) {
    return -EINVAL;
  }
  status = kvaf_context_and_state_lock(tid, &context, &state);
  if (status != 0) {
    return status;
  }
  if (state->pending != KVAF_SOURCE_UMMU_PENDING_NONE) {
    const bool has_token = segment_attribute->token != NULL;
    const uint32_t retry_token = has_token
        ? segment_attribute->token->tokenVal
        : 0U;
    if (kvaf_pending_matches(state,
            KVAF_SOURCE_UMMU_PENDING_GRANT_TOKEN_UPDATE,
            (uint64_t)(uintptr_t)data, data_size,
            has_token, retry_token)) {
      status = kvaf_retry_data_operation_locked(state);
    } else {
      status = -EBUSY;
    }
    kvaf_context_and_state_unlock(context, state);
    return status;
  }

  data_info.data = data;
  data_info.data_size = data_size;
  data_info.perm = permission;
  data_info.token = segment_attribute->token;
  data_info.e_bit = segment_attribute->e_bit;
  if (ummu_check_data(&data_info, state->mapt->mode) != 0 ||
      ummu_perm_data_preproc(&data_info) != 0) {
    status = -EINVAL;
    goto done;
  }
  operation = ummu_grant_check(state->mapt, &data_info);
  if (operation == UMMU_OP_END) {
    status = -EINVAL;
    goto clear_token;
  }
  data_info.op = operation;
  if (operation == UMMU_ADD_TOKEN) {
    kvaf_set_pending(
        state, KVAF_SOURCE_UMMU_PENDING_GRANT_TOKEN_UPDATE);
    state->pending_address = data_info.data_base;
    state->pending_length = data_info.data_size;
    state->pending_token = data_info.tokenval;
    state->pending_by_token = data_info.token != NULL;
  }
  if (state->mapt->mode == MAPT_MODE_ENTRY) {
    status = ummu_entry_op(state->mapt, &data_info);
  } else if (state->mapt->mode == MAPT_MODE_TABLE) {
    status = ummu_table_op(state->mapt, &data_info);
  } else {
    status = -EINVAL;
  }
  if (status == 0) {
    status = ummu_update_info(operation, state->mapt, &data_info);
  }
  if (status != 0) {
    if (state->pending != KVAF_SOURCE_UMMU_PENDING_NONE) {
      state->pending_state_unknown = true;
    }
    kvaf_record_failure(state, KVAF_SOURCE_UMMU_FAILURE_MAPT_UPDATE,
        status, KVAF_SOURCE_UMMU_RETRY_UNSAFE, true, state->pending,
        data_info.data_base, data_info.data_size);
    goto clear_token;
  }
  if (state->pending ==
      KVAF_SOURCE_UMMU_PENDING_GRANT_TOKEN_UPDATE) {
    state->pending_local_update_done = true;
    status = kvaf_send_plbi_checked(state, PLBI_VA_OP,
        data_info.data_base, data_info.data_size);
    if (status == 0) {
      kvaf_clear_pending(state);
    }
  }

clear_token:
  data_info.tokenval = 0;
done:
  kvaf_context_and_state_unlock(context, state);
  return status;
}

static int kvaf_ungrant_data(
    uint32_t tid, void *data, size_t data_size,
    uint32_t token, bool by_token)
{
  struct ummu_ctx_info *context;
  struct kvaf_ummu_tid_state *state;
  struct ummu_data_info data_info = {0};
  enum kvaf_source_ummu_pending_operation pending;
  enum ummu_grant_op_type operation;
  int status;

  if (data == NULL || data_size == 0) {
    return -EINVAL;
  }
  status = kvaf_context_and_state_lock(tid, &context, &state);
  if (status != 0) {
    return status;
  }
  pending = by_token
      ? KVAF_SOURCE_UMMU_PENDING_UNGRANT_TOKEN_UPDATE
      : KVAF_SOURCE_UMMU_PENDING_UNGRANT;
  if (state->pending != KVAF_SOURCE_UMMU_PENDING_NONE) {
    if (kvaf_pending_matches(state, pending,
            (uint64_t)(uintptr_t)data, data_size, by_token, token)) {
      status = kvaf_retry_data_operation_locked(state);
    } else {
      status = -EBUSY;
    }
    kvaf_context_and_state_unlock(context, state);
    return status;
  }
  data_info.data = data;
  data_info.data_size = data_size;
  data_info.tokenval = token;
  data_info.bytoken = by_token ? 1 : 0;
  data_info.data_base = (uint64_t)(uintptr_t)data;
  data_info.data_limit = data_info.data_base + data_size - 1U;
  if (state->mapt->mode == MAPT_MODE_TABLE &&
      !ALIGN_VPAGE_SHIFT_CHECK(data_info.data_base)) {
    status = -EINVAL;
    goto done;
  }
  operation = ummu_ungrant_check(state->mapt, &data_info);
  if (operation == UMMU_OP_END) {
    status = -EINVAL;
    goto done;
  }
  data_info.op = operation;
  kvaf_set_pending(state, pending);
  state->pending_address = data_info.data_base;
  state->pending_length = data_size;
  state->pending_token = token;
  state->pending_by_token = by_token;

  if (state->mapt->mode == MAPT_MODE_ENTRY) {
    status = ummu_entry_op(state->mapt, &data_info);
  } else {
    status = ummu_table_op(state->mapt, &data_info);
  }
  if (status == 0) {
    status = ummu_update_info(operation, state->mapt, &data_info);
  }
  if (status != 0) {
    state->pending_state_unknown = true;
    kvaf_record_failure(state, KVAF_SOURCE_UMMU_FAILURE_MAPT_UPDATE,
        status, KVAF_SOURCE_UMMU_RETRY_UNSAFE, true, state->pending,
        data_info.data_base, data_size);
    goto done;
  }
  state->pending_local_update_done = true;
  status = kvaf_retry_retained_mappings(state);
  if (status == 0) {
    status = kvaf_send_plbi_checked(
        state, PLBI_VA_OP, data_info.data_base, data_size);
  }
  if (status == 0) {
    kvaf_clear_pending(state);
  }

done:
  data_info.tokenval = 0;
  kvaf_context_and_state_unlock(context, state);
  return status;
}

int ummu_ungrant(uint32_t tid, void *data, size_t data_size)
{
  return kvaf_ungrant_data(tid, data, data_size, 0, false);
}

int ummu_ungrant_by_token(
    uint32_t tid, void *data, size_t data_size, uint32_t token)
{
  return kvaf_ungrant_data(tid, data, data_size, token, true);
}

static int kvaf_issue_free_tid_ioctl(
    struct kvaf_ummu_tid_state *state)
{
  struct ummu_tid_info info = {.tid = state->tid};

  if (g_kvaf_ummu_ctx == NULL || g_kvaf_ummu_ctx->shared_fd < 0) {
    return -ENODEV;
  }
  if (ioctl(g_kvaf_ummu_ctx->shared_fd, UMMU_IOCFREE_TID, &info) != 0) {
    const int status = -errno;
    const bool retry_safe = errno == EINTR || errno == EAGAIN ||
        errno == EBUSY || errno == ENOSPC;
    state->pending_state_unknown = !retry_safe;
    kvaf_record_failure(state,
        KVAF_SOURCE_UMMU_FAILURE_FREE_TID_IOCTL, status,
        retry_safe ? KVAF_SOURCE_UMMU_RETRY_SAFE
                   : KVAF_SOURCE_UMMU_RETRY_UNSAFE,
        !retry_safe,
        KVAF_SOURCE_UMMU_PENDING_FREE_TID, 0, 0);
    return status;
  }
  return 0;
}

static int kvaf_mapt_has_live_mapping(
    const struct kvaf_ummu_tid_state *state)
{
  if (state->mapt == NULL) {
    return 0;
  }
  if (state->mapt->mode == MAPT_MODE_ENTRY) {
    return state->mapt->block_base.entry_block != NULL &&
        state->mapt->block_base.entry_block->valid != 0;
  }
  if (state->mapt->mode == MAPT_MODE_TABLE) {
    return state->mapt->block_base.table_ctx == NULL ||
        !kvaf_source_ummu_seg_tree_is_empty(
            state->mapt->block_base.table_ctx->granted_addr_mng);
  }
  return 1;
}

static int kvaf_munmap_owned(
    struct kvaf_ummu_tid_state *state, void *address, size_t length)
{
  if (address == NULL || length == 0) {
    return -EINVAL;
  }
  if (munmap(address, length) == 0) {
    return 0;
  }
  kvaf_record_failure(state, KVAF_SOURCE_UMMU_FAILURE_MUNMAP,
      -errno, KVAF_SOURCE_UMMU_RETRY_SAFE, false,
      KVAF_SOURCE_UMMU_PENDING_FREE_TID,
      (uint64_t)(uintptr_t)address, length);
  return -errno;
}

static int kvaf_release_local_mapt_locked(
    struct kvaf_ummu_tid_state *state)
{
  struct ummu_mapt_table_ctx *table;
  struct ummu_mapt_block *block;
  struct ummu_mapt_queue_args *queues;
  uint32_t index;
  int status;

  status = kvaf_retry_retained_mappings(state);
  if (status != 0 || state->mapt == NULL) {
    return status;
  }
  if (state->mapt->mode == MAPT_MODE_ENTRY) {
    if (state->mapt->block_base.entry_block != NULL) {
      status = kvaf_munmap_owned(state,
          state->mapt->block_base.entry_block, BLOCK_SIZE_4K);
      if (status != 0) {
        return status;
      }
      state->mapt->block_base.entry_block = NULL;
    }
  } else if (state->mapt->mode == MAPT_MODE_TABLE) {
    table = state->mapt->block_base.table_ctx;
    if (table == NULL ||
        !kvaf_source_ummu_seg_tree_is_empty(table->granted_addr_mng)) {
      return -EBUSY;
    }
    for (index = 0; index < INDEX_MAX_SIZE; ++index) {
      block = table->mapt_block_array[index];
      if (block == NULL) {
        continue;
      }
      status = kvaf_munmap_owned(
          state, block->block_addr, table->blk_exp_size);
      if (status != 0) {
        return status;
      }
      table->mapt_block_array[index] = NULL;
      free(block);
    }
    ummu_destroy_seg_mng(&table->granted_addr_mng);
    free(table);
    state->mapt->block_base.table_ctx = NULL;
  } else {
    return -EINVAL;
  }

  if (!state->mapt->kcmd_plbi &&
      state->mapt->queue_ctx.queue != NULL) {
    queues = state->mapt->queue_ctx.queue;
    status = kvaf_munmap_owned(state,
        queues[0].cmd_entry, state->mapt->queue_ctx.queue_size);
    if (status != 0) {
      return status;
    }
    free(queues);
    state->mapt->queue_ctx.queue = NULL;
  }
  return 0;
}

static int kvaf_free_tid_state_locked(
    struct kvaf_ummu_tid_state *state)
{
  bool mapt_locked = false;
  int status;

  if (state->pending == KVAF_SOURCE_UMMU_PENDING_FREE_TID &&
      state->pending_state_unknown) {
    kvaf_record_failure(state, KVAF_SOURCE_UMMU_FAILURE_INTERNAL,
        -EUCLEAN, KVAF_SOURCE_UMMU_RETRY_UNSAFE, true,
        KVAF_SOURCE_UMMU_PENDING_FREE_TID, 0, 0);
    return -EUCLEAN;
  }

  if (state->pending != KVAF_SOURCE_UMMU_PENDING_NONE &&
      state->pending != KVAF_SOURCE_UMMU_PENDING_FREE_TID) {
    if (state->mapt == NULL ||
        pthread_mutex_lock(&state->mapt->mapt_mutex) != 0) {
      return -EDEADLK;
    }
    mapt_locked = true;
    g_kvaf_ummu_current_state = state;
    status = kvaf_retry_data_operation_locked(state);
    g_kvaf_ummu_current_state = NULL;
    (void)pthread_mutex_unlock(&state->mapt->mapt_mutex);
    mapt_locked = false;
    if (status != 0) {
      return status;
    }
  }
  if (state->pending == KVAF_SOURCE_UMMU_PENDING_NONE) {
    if (kvaf_mapt_has_live_mapping(state)) {
      kvaf_record_failure(state,
          KVAF_SOURCE_UMMU_FAILURE_LIVE_MAPPING, -EBUSY,
          KVAF_SOURCE_UMMU_RETRY_SAFE, false,
          KVAF_SOURCE_UMMU_PENDING_FREE_TID, 0, 0);
      return -EBUSY;
    }
    kvaf_set_pending(state, KVAF_SOURCE_UMMU_PENDING_FREE_TID);
    state->cleanup_phase = state->allocation_orphan
        ? KVAF_UMMU_CLEANUP_PLBI_ALL_DONE
        : KVAF_UMMU_CLEANUP_ACTIVE;
  }

  if (state->mapt != NULL) {
    if (pthread_mutex_lock(&state->mapt->mapt_mutex) != 0) {
      return -EDEADLK;
    }
    mapt_locked = true;
    g_kvaf_ummu_current_state = state;
  }
  if (state->cleanup_phase == KVAF_UMMU_CLEANUP_ACTIVE) {
    status = kvaf_send_plbi_checked(state, PLBI_ALL_OP, 0, 0);
    if (status != 0) {
      goto unlock;
    }
    state->cleanup_phase = KVAF_UMMU_CLEANUP_PLBI_ALL_DONE;
  }
  if (state->cleanup_phase == KVAF_UMMU_CLEANUP_PLBI_ALL_DONE) {
    status = kvaf_issue_free_tid_ioctl(state);
    if (status != 0) {
      goto unlock;
    }
    state->cleanup_phase = KVAF_UMMU_CLEANUP_KERNEL_TID_FREED;
  }
  if (state->cleanup_phase == KVAF_UMMU_CLEANUP_KERNEL_TID_FREED) {
    status = state->mapt == NULL
        ? kvaf_retry_retained_mappings(state)
        : kvaf_release_local_mapt_locked(state);
    if (status != 0) {
      goto unlock;
    }
    state->cleanup_phase = KVAF_UMMU_CLEANUP_LOCAL_RELEASED;
  }
  status = 0;

unlock:
  g_kvaf_ummu_current_state = NULL;
  if (mapt_locked) {
    (void)pthread_mutex_unlock(&state->mapt->mapt_mutex);
  }
  if (status != 0 ||
      state->cleanup_phase != KVAF_UMMU_CLEANUP_LOCAL_RELEASED) {
    return status == 0 ? -EUCLEAN : status;
  }
  if (state->mapt != NULL &&
      pthread_mutex_destroy(&state->mapt->mapt_mutex) != 0) {
    state->pending_state_unknown = true;
    kvaf_record_failure(state, KVAF_SOURCE_UMMU_FAILURE_INTERNAL,
        -EBUSY, KVAF_SOURCE_UMMU_RETRY_UNSAFE, true,
        KVAF_SOURCE_UMMU_PENDING_FREE_TID, 0, 0);
    return -EBUSY;
  }
  if (state->map_inserted &&
      ummu_map_del(g_kvaf_ummu_ctx->mapt_map, state->tid) != 0) {
    state->pending_state_unknown = true;
    kvaf_record_failure(state, KVAF_SOURCE_UMMU_FAILURE_MAP_DELETE,
        -EUCLEAN, KVAF_SOURCE_UMMU_RETRY_UNSAFE, true,
        KVAF_SOURCE_UMMU_PENDING_FREE_TID, 0, 0);
    return -EUCLEAN;
  }
  free(state->mapt);
  state->mapt = NULL;
  kvaf_clear_pending(state);
  kvaf_remove_state_locked(state);
  free(state);
  return 0;
}

int ummu_allocate_tid(struct ummu_tid_attr *attribute, uint32_t *tid_out)
{
  struct kvaf_ummu_tid_state *state;
  struct ummu_tid_info info = {0};
  struct ummu_mapt_info *mapt;
  int cleanup_status;
  int status;

  if (attribute == NULL || tid_out == NULL ||
      attribute->mode < MAPT_MODE_TABLE ||
      attribute->mode >= MAPT_MODE_END) {
    return -EINVAL;
  }
  /*
   * Private source contract: UINT32_MAX means that create failed without a
   * retained TID.  A different value on error is an exact cleanup handle that
   * the source UDMA wrapper must keep in its orphan sidecar and retry through
   * ummu_free_tid().
   */
  *tid_out = UINT32_MAX;
  status = kvaf_ensure_context();
  if (status != 0) {
    return status;
  }
  state = calloc(1, sizeof(*state));
  if (state == NULL) {
    return -ENOMEM;
  }
  /*
   * Take the owner-registry lock before the first resource-creating ioctl.
   * Once ALLOC_TID succeeds there must be no fallible lock acquisition
   * between the kernel handle and its registry entry; otherwise a mutex
   * failure could lose the only cleanup owner and make Close non-retryable.
   */
  if (pthread_mutex_lock(&g_kvaf_ummu_ctx->ctx_mutex) != 0) {
    free(state);
    return -EDEADLK;
  }
  info.mode = attribute->mode;
  status = ummu_get_tid(g_kvaf_ummu_ctx->shared_fd, &info);
  if (status != 0) {
    kvaf_record_failure(NULL,
        KVAF_SOURCE_UMMU_FAILURE_ALLOC_TID_IOCTL, status,
        KVAF_SOURCE_UMMU_RETRY_SAFE, false,
        KVAF_SOURCE_UMMU_PENDING_NONE, 0, 0);
    (void)pthread_mutex_unlock(&g_kvaf_ummu_ctx->ctx_mutex);
    free(state);
    return status;
  }
  (void)kvaf_ummu_uapi_hw_cap(&info);
  state->tid = info.tid;
  state->mode = attribute->mode;
  state->allocation_orphan = true;
  state->cleanup_phase = KVAF_UMMU_CLEANUP_PLBI_ALL_DONE;
  kvaf_set_pending(state, KVAF_SOURCE_UMMU_PENDING_FREE_TID);
  kvaf_add_state_locked(state);
  *tid_out = info.tid;

  g_kvaf_ummu_current_state = state;
  mapt = ummu_mapt_create(&info);
  g_kvaf_ummu_current_state = NULL;
  if (mapt == NULL) {
    cleanup_status = kvaf_free_tid_state_locked(state);
    if (cleanup_status == 0) {
      *tid_out = UINT32_MAX;
      kvaf_record_failure(NULL,
          KVAF_SOURCE_UMMU_FAILURE_MAPT_CREATE, -ENOMEM,
          KVAF_SOURCE_UMMU_RETRY_NOT_APPLICABLE, false,
          KVAF_SOURCE_UMMU_PENDING_NONE, 0, 0);
    } else if (!state->pending_state_unknown) {
      /* Preserve a lower-level unsafe classification instead of rewriting it
       * as retryable merely because MAPT creation was the initiating step. */
      kvaf_record_failure(state,
          KVAF_SOURCE_UMMU_FAILURE_MAPT_CREATE, cleanup_status,
          KVAF_SOURCE_UMMU_RETRY_SAFE, false,
          KVAF_SOURCE_UMMU_PENDING_FREE_TID, 0, 0);
    }
    (void)pthread_mutex_unlock(&g_kvaf_ummu_ctx->ctx_mutex);
    return -ENOMEM;
  }
  state->mapt = mapt;
  state->allocation_orphan = false;
  state->cleanup_phase = KVAF_UMMU_CLEANUP_ACTIVE;
  status = ummu_map_insert(
      g_kvaf_ummu_ctx->mapt_map, info.tid, mapt);
  if (status != 0) {
    kvaf_record_failure(state, KVAF_SOURCE_UMMU_FAILURE_MAP_INSERT,
        status, KVAF_SOURCE_UMMU_RETRY_SAFE, false,
        KVAF_SOURCE_UMMU_PENDING_FREE_TID, 0, 0);
    (void)pthread_mutex_unlock(&g_kvaf_ummu_ctx->ctx_mutex);
    return status;
  }
  state->map_inserted = true;
  kvaf_clear_pending(state);
  (void)pthread_mutex_unlock(&g_kvaf_ummu_ctx->ctx_mutex);
  return 0;
}

int ummu_free_tid(uint32_t tid)
{
  struct kvaf_ummu_tid_state *state;
  int status;

  status = kvaf_ensure_context();
  if (status != 0) {
    return status;
  }
  if (pthread_mutex_lock(&g_kvaf_ummu_ctx->ctx_mutex) != 0) {
    return -EDEADLK;
  }
  state = kvaf_find_state_locked(tid);
  status = state == NULL ? -ENOENT : kvaf_free_tid_state_locked(state);
  (void)pthread_mutex_unlock(&g_kvaf_ummu_ctx->ctx_mutex);
  return status;
}

uint64_t kvaf_source_ummu_capability(void)
{
  return KVAF_SOURCE_UMMU_CHECKED_RETRYABLE_V1 |
      KVAF_SOURCE_UMMU_PINNED_FULL_UAPI;
}

uint32_t kvaf_source_ummu_live_tid_count(void)
{
  return __atomic_load_n(&g_kvaf_ummu_live_tid_count, __ATOMIC_RELAXED);
}

uint32_t kvaf_source_ummu_pending_operation_count(void)
{
  return __atomic_load_n(&g_kvaf_ummu_pending_count, __ATOMIC_RELAXED);
}

uint32_t kvaf_source_ummu_retained_mapping_count(void)
{
  return __atomic_load_n(&g_kvaf_ummu_retained_count, __ATOMIC_RELAXED);
}

int kvaf_source_ummu_get_status(struct kvaf_source_ummu_status *status)
{
  if (status == NULL ||
      status->structure_size < sizeof(struct kvaf_source_ummu_status)) {
    return -EINVAL;
  }
  if (pthread_mutex_lock(&g_kvaf_ummu_status_lock) != 0) {
    return -EDEADLK;
  }
  *status = g_kvaf_ummu_last_status;
  status->abi_version = KVAF_SOURCE_UMMU_STATUS_ABI_VERSION;
  status->structure_size = sizeof(*status);
  status->capability_magic = KVAF_SOURCE_UMMU_CAPABILITY_MAGIC;
  status->capability_bits = kvaf_source_ummu_capability();
  status->live_tid_count = kvaf_source_ummu_live_tid_count();
  status->pending_operation_count =
      kvaf_source_ummu_pending_operation_count();
  status->retained_mapping_count =
      kvaf_source_ummu_retained_mapping_count();
  (void)pthread_mutex_unlock(&g_kvaf_ummu_status_lock);
  return 0;
}

int kvaf_source_ummu_retry_pending(uint32_t tid)
{
  struct kvaf_ummu_tid_state *state;
  int status;

  status = kvaf_ensure_context();
  if (status != 0) {
    return status;
  }
  if (pthread_mutex_lock(&g_kvaf_ummu_ctx->ctx_mutex) != 0) {
    return -EDEADLK;
  }
  state = kvaf_find_state_locked(tid);
  if (state == NULL || state->pending == KVAF_SOURCE_UMMU_PENDING_NONE) {
    status = -ENOENT;
  } else if (state->pending == KVAF_SOURCE_UMMU_PENDING_FREE_TID) {
    status = kvaf_free_tid_state_locked(state);
  } else if (state->mapt == NULL ||
             pthread_mutex_lock(&state->mapt->mapt_mutex) != 0) {
    status = -EDEADLK;
  } else {
    g_kvaf_ummu_current_state = state;
    status = kvaf_retry_data_operation_locked(state);
    g_kvaf_ummu_current_state = NULL;
    (void)pthread_mutex_unlock(&state->mapt->mapt_mutex);
  }
  (void)pthread_mutex_unlock(&g_kvaf_ummu_ctx->ctx_mutex);
  return status;
}

int kvaf_source_ummu_checked_contract_ready(void)
{
  /* This object never resolves any UMMU operation from libummu.so. */
  return kvaf_source_ummu_capability() ==
      (KVAF_SOURCE_UMMU_CHECKED_RETRYABLE_V1 |
       KVAF_SOURCE_UMMU_PINNED_FULL_UAPI);
}

static __attribute__((destructor)) void kvaf_source_ummu_uninit(void)
{
  struct ummu_ctx_info *context;

  if (pthread_mutex_lock(&g_kvaf_ummu_init_lock) != 0) {
    return;
  }
  context = g_kvaf_ummu_ctx;
  if (context == NULL) {
    (void)pthread_mutex_unlock(&g_kvaf_ummu_init_lock);
    return;
  }
  if (kvaf_source_ummu_live_tid_count() != 0U ||
      kvaf_source_ummu_pending_operation_count() != 0U ||
      kvaf_source_ummu_retained_mapping_count() != 0U) {
    /* Never make retry handles unreachable during a normal provider unload. */
    kvaf_record_failure(NULL, KVAF_SOURCE_UMMU_FAILURE_LIVE_MAPPING,
        -EBUSY, KVAF_SOURCE_UMMU_RETRY_UNSAFE, true,
        KVAF_SOURCE_UMMU_PENDING_NONE, 0, 0);
    (void)pthread_mutex_unlock(&g_kvaf_ummu_init_lock);
    return;
  }
  g_kvaf_ummu_ctx = NULL;
  if (context->shared_fd >= 0) {
    (void)close(context->shared_fd);
  }
  ummu_map_clear(context->mapt_map, NULL);
  (void)pthread_mutex_destroy(&context->ctx_mutex);
  free(context);
  (void)pthread_mutex_unlock(&g_kvaf_ummu_init_lock);
}
