#ifndef KVAF_SOURCE_MATCHED_UMMU_MAPPING_H
#define KVAF_SOURCE_MATCHED_UMMU_MAPPING_H

#include <errno.h>
#include <stdbool.h>
#include <stddef.h>

/*
 * Intrusive ownership record for one successful UMMU mmap.
 *
 * The record is allocated before mmap and linked immediately after mmap
 * succeeds.  Consequently a void upstream rollback can transfer ownership to
 * the retained list after munmap fails without performing another allocation.
 */
struct kvaf_ummu_owned_mapping {
  void *address;
  size_t length;
  bool retained;
  struct kvaf_ummu_owned_mapping *next;
};

struct kvaf_ummu_mapping_ledger {
  struct kvaf_ummu_owned_mapping *head;
  size_t retained_count;
};

static inline void kvaf_ummu_mapping_attach(
    struct kvaf_ummu_mapping_ledger *ledger,
    struct kvaf_ummu_owned_mapping *mapping,
    void *address, size_t length)
{
  mapping->address = address;
  mapping->length = length;
  mapping->retained = false;
  mapping->next = ledger->head;
  ledger->head = mapping;
}

static inline struct kvaf_ummu_owned_mapping *kvaf_ummu_mapping_find(
    const struct kvaf_ummu_mapping_ledger *ledger,
    const void *address, size_t length)
{
  struct kvaf_ummu_owned_mapping *mapping;

  if (ledger == NULL || address == NULL || length == 0) {
    return NULL;
  }
  for (mapping = ledger->head; mapping != NULL; mapping = mapping->next) {
    if (mapping->address == address && mapping->length == length) {
      return mapping;
    }
  }
  return NULL;
}

/* Returns one only for the first active-to-retained ownership transition. */
static inline int kvaf_ummu_mapping_mark_retained(
    struct kvaf_ummu_mapping_ledger *ledger,
    void *address, size_t length)
{
  struct kvaf_ummu_owned_mapping *mapping =
      kvaf_ummu_mapping_find(ledger, address, length);

  if (mapping == NULL) {
    return -ENOENT;
  }
  if (mapping->retained) {
    return 0;
  }
  mapping->retained = true;
  ++ledger->retained_count;
  return 1;
}

/*
 * Detach only after munmap succeeds.  The returned record remains valid so
 * the caller can update process-wide counters before releasing its storage.
 */
static inline struct kvaf_ummu_owned_mapping *kvaf_ummu_mapping_detach(
    struct kvaf_ummu_mapping_ledger *ledger,
    const void *address, size_t length)
{
  struct kvaf_ummu_owned_mapping **cursor;
  struct kvaf_ummu_owned_mapping *mapping;

  if (ledger == NULL || address == NULL || length == 0) {
    return NULL;
  }
  for (cursor = &ledger->head; *cursor != NULL;
       cursor = &(*cursor)->next) {
    mapping = *cursor;
    if (mapping->address != address || mapping->length != length) {
      continue;
    }
    *cursor = mapping->next;
    if (mapping->retained) {
      --ledger->retained_count;
    }
    mapping->next = NULL;
    return mapping;
  }
  return NULL;
}

#endif
