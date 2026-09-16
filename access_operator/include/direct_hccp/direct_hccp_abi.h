#ifndef KVAF_ACCESS_OPERATOR_DIRECT_HCCP_ABI_H_
#define KVAF_ACCESS_OPERATOR_DIRECT_HCCP_ABI_H_

// Minimal C ABI used by KVAF's direct operator transport to call CANN's RA
// implementation directly.
// This is deliberately not an HCOMM interface: on A5 the symbols are resolved
// from libra.so (with libhccp compatibility fallbacks). Keep these layouts in
// one place because RA is an implementation ABI, not a stable CANN application
// API.

#include <cstddef>
#include <cstdint>

namespace kvaf::direct_hccp {

constexpr uint32_t kEidBytes = 16;
constexpr uint32_t kMemKeyBytes = 128;
constexpr uint32_t kQpKeyBytes = 64;
constexpr uint32_t kResvWords = 16;

enum NetworkMode : int32_t {
  kPeerOnline = 0,
  kOfflineHdc = 1,
};

enum TransportMode : uint32_t {
  kReliableMessage = 1,
};

enum JfcMode : uint32_t {
  kJfcNormal = 0,
  kJfcUserCtlNormal = 3,
};

enum JettyMode : uint32_t {
  kJettyStandard = 0,
  kJettyUserCtlNormal = 3,
};

enum JettyImportMode : uint32_t {
  kJettyImportNormal = 0,
  kJettyImportExp = 1,
};

enum JettyGroupPolicy : uint32_t {
  kJettyGroupRoundRobin = 0,
};

enum TargetType : uint32_t {
  kTargetJetty = 1,
};

// HccpEid carries a uint64_t representation in addition to raw bytes, so its
// ABI alignment is eight even when this demo uses only raw[16].
struct alignas(8) Eid {
  uint8_t raw[kEidBytes];
};

struct CtxInitCfg {
  int32_t mode;
};

struct CtxInitAttr {
  uint32_t phy_id;
  struct {
    uint32_t eid_index;
    uint32_t reserved0;
    Eid eid;
  } ub;
  uint32_t reserved[16];
};

struct TokenId {
  uint32_t token_id;
};

struct MemKey {
  uint8_t value[kMemKeyBytes];
  uint8_t size;
};

struct MemInfo {
  uint64_t addr;
  uint64_t size;
};

struct MemRegAttr {
  MemInfo mem;
  struct {
    uint32_t flags;
    uint32_t token_value;
    void *token_id_handle;
  } ub;
  uint32_t reserved[8];
};

struct MemRegInfo {
  MemKey key;
  struct {
    uint32_t token_id;
    uint64_t target_seg_handle;
  } ub;
  uint32_t reserved[8];
};

struct MrRegInfo {
  MemRegAttr in;
  MemRegInfo out;
};

struct MemImportAttr {
  MemKey key;
  struct {
    uint32_t flags;
    uint64_t mapping_addr;
    uint32_t token_value;
  } ub;
  uint32_t reserved[4];
};

struct MemImportInfo {
  struct {
    uint64_t target_seg_handle;
  } ub;
  uint32_t reserved[4];
};

struct MrImportInfo {
  MemImportAttr in;
  MemImportInfo out;
};

struct CqCreateAttr {
  void *channel_handle;
  uint32_t depth;
  uint32_t reserved_align;
  struct {
    uint64_t user_ctx;
    uint32_t mode;
    uint32_t ceqn;
    uint32_t flag;
    struct {
      bool valid;
      uint8_t reserved[3];
      uint32_t cqe_flag;
    } ccu_ex_cfg;
  } ub;
};

struct CqCreateInfo {
  uint64_t va;
  uint32_t id;
  uint32_t cqe_size;
  uint64_t buffer_addr;
  uint64_t swdb_addr;
};

struct CqInfo {
  CqCreateAttr in;
  CqCreateInfo out;
};

struct JettyQueueCfg {
  uint32_t buffer_size;
  uint32_t reserved_align;
  uint64_t buffer_va;
};

struct JettyExtMode {
  JettyQueueCfg sq;
  bool pi_type;
  uint8_t reserved0[3];
  // Raw HCCP ub.extMode.cstmFlag union. Bit 0 is sqCstm. A5 USER_CTL Jettys
  // require it clear so HCCP allocates and returns the SQ; a set bit is for
  // the CCU custom-SQ path and is rejected for a normal USER_CTL Jetty.
  uint32_t cstm_flag;
  uint32_t sqebb_num;
};

struct QpCreateAttr {
  void *scq_handle;
  void *rcq_handle;
  void *srq_handle;
  uint32_t sq_depth;
  uint32_t rq_depth;
  uint32_t transport_mode;
  uint32_t reserved_align;
  struct {
    uint32_t mode;
    uint32_t jetty_id;
    uint32_t flag;
    uint32_t jfs_flag;
    void *token_id_handle;
    uint32_t token_value;
    uint8_t priority;
    uint8_t rnr_retry;
    uint8_t err_timeout;
    uint8_t reserved0;
    JettyExtMode ext_mode;
  } ub;
  uint32_t reserved[kResvWords];
};

struct QpKey {
  uint8_t value[kQpKeyBytes];
  uint8_t size;
};

struct QpCreateInfo {
  QpKey key;
  struct {
    uint32_t uasid;
    uint32_t id;
    uint64_t sq_buffer_va;
    uint64_t wqebb_size;
    uint64_t db_addr;
    uint32_t db_token_id;
    uint32_t share_info_len;
    uint64_t share_info_addr;
  } ub;
  uint64_t va;
  uint32_t reserved[kResvWords];
};

struct TpInfo {
  uint64_t tp_handle;
  uint32_t reserved;
};

struct GetTpCfg {
  uint32_t flag;
  uint32_t transport_mode;
  Eid local_eid;
  Eid peer_eid;
};

// Optional diagnostic ABI from HCCP hccp_ctx.h / hccp_ctx_tp.h. These
// readbacks are best effort; they do not participate in channel creation.
struct DevBaseAttr {
  uint32_t sq_max_depth;
  uint32_t rq_max_depth;
  uint32_t sq_max_sge;
  uint32_t rq_max_sge;
  struct {
    uint32_t max_jfs_inline_len;
    uint32_t max_jfs_rsge;
    uint32_t die_id;
    uint32_t chip_id;
    uint32_t func_id;
    uint32_t rm_tp_cap;
    uint32_t rc_tp_cap;
    uint32_t um_tp_cap;
    uint32_t tp_feature;
    struct { uint32_t sl; uint32_t tp_type; } priority_info[16];
    uint32_t reserved0;
  } ub;
  uint32_t max_read_size;
  uint32_t max_write_size;
  uint64_t max_msg_size;
  uint32_t reserved[12];
};

struct JettyAttr {
  uint32_t mask;
  uint32_t rx_threshold;
  uint32_t state;
  uint32_t reserved[2];
};

#pragma pack(push, 1)
struct TpAttr {
  uint8_t retry_times_init : 3;
  uint8_t at : 5;
  uint8_t sip[16];
  uint8_t dip[16];
  uint8_t sma[6];
  uint8_t dma[6];
  uint16_t vlan_id : 12;
  uint8_t vlan_en : 1;
  uint8_t dscp : 6;
  uint8_t at_times : 5;
  uint8_t sl : 4;
  uint8_t ttl;
  uint16_t ack_udp_srcport;
  uint16_t data_udp_srcport;
  uint8_t udp_srcport_range : 4;
  uint8_t spray_en : 1;
  uint8_t udp_global_en : 1;
  uint8_t reserved0 : 2;
  uint16_t sl_bitmap;
  uint8_t dscp_config_mode : 1;
  uint8_t reserved1 : 7;
  uint8_t reserved[70];
};
#pragma pack(pop)

// The exported UB QpKey contains RsJettyKeyInfo (32 bytes on RA, or a
// 28-byte packed peer key). Only known sizes are decoded by diagnostics.
struct JettyKeyIdentity {
  Eid eid;
  uint32_t uasid;
  uint32_t id;
  uint32_t transport_mode;
};

static_assert(sizeof(DevBaseAttr) == 248);
static_assert(offsetof(DevBaseAttr, ub.priority_info) == 52);
static_assert(offsetof(DevBaseAttr, max_msg_size) == 192);
static_assert(sizeof(JettyAttr) == 20);
static_assert(sizeof(TpAttr) == 128);
static_assert(offsetof(TpAttr, sip) == 1);
static_assert(offsetof(TpAttr, ttl) == 49);
static_assert(offsetof(TpAttr, reserved) == 58);
static_assert(sizeof(JettyKeyIdentity) == 32);

struct JettyImportExpCfg {
  uint64_t tp_handle;
  uint64_t peer_tp_handle;
  uint64_t tag;
  uint32_t tx_psn;
  uint32_t rx_psn;
  uint32_t reserved[kResvWords];
};

struct QpImportAttr {
  QpKey key;
  struct {
    uint32_t mode;
    uint32_t token_value;
    uint32_t policy;
    uint32_t type;
    uint32_t flag;
    JettyImportExpCfg exp_import_cfg;
    uint32_t tp_type;
  } ub;
  uint32_t reserved[7];
};

struct QpImportInfo {
  struct {
    uint64_t target_jetty_handle;
    uint32_t tpn;
  } ub;
  uint32_t reserved[8];
};

struct QpImportInfoT {
  QpImportAttr in;
  QpImportInfo out;
};

struct RaInfo {
  int32_t mode;
  uint32_t phy_id;
};

// Bootstrap ABI for the HCCP resource service.  RaCtx* assumes its matching
// PEER/HDC resource agent has already been initialized.
struct RaInitConfig {
  uint32_t phy_id;
  uint32_t nic_position;
  int32_t hdc_type;
  bool enable_hdc_async;
  uint8_t reserved[3];
};

struct RtProcExtParam {
  const char *param_info;
  uint64_t param_len;
};

struct RtNetServiceOpenArgs {
  RtProcExtParam *ext_param_list;
  uint64_t ext_param_count;
};

// libhccp.so C entry points.  They are loaded with dlsym so this demo does
// not link HCOMM or need an HCOMM source tree.
using CtxInitFn = int (*)(CtxInitCfg *, CtxInitAttr *, void **);
using CtxDeinitFn = int (*)(void *);
using TokenAllocFn = int (*)(void *, TokenId *, void **);
using TokenFreeFn = int (*)(void *, void *);
using LmemRegisterFn = int (*)(void *, MrRegInfo *, void **);
using LmemUnregisterFn = int (*)(void *, void *);
using RmemImportFn = int (*)(void *, MrImportInfo *, void **);
using RmemUnimportFn = int (*)(void *, void *);
using CqCreateFn = int (*)(void *, CqInfo *, void **);
using CqDestroyFn = int (*)(void *, void *);
using QpCreateFn = int (*)(void *, QpCreateAttr *, QpCreateInfo *, void **);
using QpDestroyFn = int (*)(void *);
using QpImportFn = int (*)(void *, QpImportInfoT *, void **);
using QpUnimportFn = int (*)(void *, void *);
using GetTpInfoFn = int (*)(void *, GetTpCfg *, TpInfo *, uint32_t *);
// The synchronous TP query is PEER-only on A5.  HDC (the Device/AIV
// context) has a separate asynchronous request path.
using GetTpInfoAsyncFn = int (*)(void *, GetTpCfg *, TpInfo *, uint32_t *, void **);
using AsyncReqResultFn = int (*)(void *, int *);
using GetRandomFn = int (*)(RaInfo *, uint32_t *);
using GetDevBaseAttrFn = int (*)(void *, DevBaseAttr *);
using QpQueryBatchFn = int (*)(void **, JettyAttr *, uint32_t *);
using GetTpAttrAsyncFn = int (*)(void *, uint64_t, uint32_t *, TpAttr *, void **);
using RaInitFn = int (*)(RaInitConfig *);
using RaDeinitFn = int (*)(RaInitConfig *);
using RtOpenNetServiceFn = int (*)(RtNetServiceOpenArgs *);

inline constexpr uint32_t MemRegFlags(bool non_pin) {
  // token policy=plain text, access=READ|WRITE|ATOMIC, token id valid.
  return 1U | (0xEU << 5U) | (non_pin ? (1U << 11U) : 0U) | (1U << 13U);
}

inline constexpr uint32_t MemImportFlags() {
  // access=READ|WRITE|ATOMIC, no host-side address mapping required.
  return 0xEU << 1U;
}

inline constexpr uint32_t JettyFlagShareJfr() { return 1U; }
inline constexpr uint32_t JfsFlagErrorSuspend() { return 1U << 1U; }
inline constexpr uint32_t ImportFlagPlainTextToken() { return 1U; }
inline constexpr uint32_t CtpProtocolType() { return 1U; }
inline constexpr uint32_t CtpFlag() { return 1U; }

}  // namespace kvaf::direct_hccp

#endif  // KVAF_ACCESS_OPERATOR_DIRECT_HCCP_ABI_H_
