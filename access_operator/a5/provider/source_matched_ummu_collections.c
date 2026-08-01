#include "kvaf_source_matched_ummu.h"

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>

#include "ummu_seg_tree.h"
#include "ummu_map.h"
#include "ummu_utils.h"

/*
 * The upstream libummu uses std::unordered_map and std::map for these two
 * tiny ownership indexes.  The provider keeps an equivalent C
 * implementation private so AICPU does not gain a libstdc++ dependency.
 */
struct kvaf_ummu_map_node {
  uint32_t key;
  void *value;
  struct kvaf_ummu_map_node *next;
};

struct kvaf_ummu_map {
  struct kvaf_ummu_map_node *head;
};

void *create_map(void)
{
  return calloc(1, sizeof(struct kvaf_ummu_map));
}

int hashmap_insert(void *opaque_map, uint32_t key, void *value)
{
  struct kvaf_ummu_map *map = opaque_map;
  struct kvaf_ummu_map_node *node;

  if (map == NULL || value == NULL) {
    return -EINVAL;
  }
  for (node = map->head; node != NULL; node = node->next) {
    if (node->key == key) {
      return -EEXIST;
    }
  }
  node = calloc(1, sizeof(*node));
  if (node == NULL) {
    return -ENOMEM;
  }
  node->key = key;
  node->value = value;
  node->next = map->head;
  map->head = node;
  return 0;
}

void *hashmap_get(void *opaque_map, uint32_t key)
{
  struct kvaf_ummu_map *map = opaque_map;
  struct kvaf_ummu_map_node *node;

  if (map == NULL) {
    return NULL;
  }
  for (node = map->head; node != NULL; node = node->next) {
    if (node->key == key) {
      return node->value;
    }
  }
  return NULL;
}

size_t hashmap_del(void *opaque_map, uint32_t key)
{
  struct kvaf_ummu_map *map = opaque_map;
  struct kvaf_ummu_map_node **cursor;
  struct kvaf_ummu_map_node *node;

  if (map == NULL) {
    return 0;
  }
  cursor = &map->head;
  while (*cursor != NULL) {
    node = *cursor;
    if (node->key == key) {
      *cursor = node->next;
      free(node);
      return 1;
    }
    cursor = &node->next;
  }
  return 0;
}

void hashmap_clear(void *opaque_map, value_destroy destroyer)
{
  struct kvaf_ummu_map *map = opaque_map;
  struct kvaf_ummu_map_node *node;

  if (map == NULL) {
    return;
  }
  while (map->head != NULL) {
    node = map->head;
    map->head = node->next;
    if (destroyer != NULL) {
      destroyer(node->value);
    } else {
      free(node->value);
    }
    free(node);
  }
  free(map);
}

void *ummu_map_create(void)
{
  return create_map();
}

int ummu_map_insert(void *map, uint32_t key, void *value)
{
  return hashmap_insert(map, key, value);
}

void *ummu_map_get(void *map, uint32_t key)
{
  return hashmap_get(map, key);
}

int ummu_map_del(void *map, uint32_t key)
{
  return hashmap_del(map, key) == 1U ? 0 : -ENOENT;
}

void ummu_map_clear(void *map, ummu_value_destroy destroyer)
{
  hashmap_clear(map, destroyer);
}

struct kvaf_ummu_range_node {
  uint64_t left;
  uint64_t right;
  void *data;
  struct kvaf_ummu_range_node *next;
};

struct kvaf_ummu_range_tree {
  struct kvaf_ummu_range_node *head;
};

static int kvaf_ummu_range_right(
    uint64_t left, uint64_t size, uint64_t *right)
{
  if (right == NULL || size > UINT64_MAX - left) {
    return -EOVERFLOW;
  }
  *right = left + size;
  return 0;
}

int ummu_create_seg_tree(uint64_t *output)
{
  struct kvaf_ummu_range_tree *tree;

  if (output == NULL) {
    return -EINVAL;
  }
  *output = 0;
  tree = calloc(1, sizeof(*tree));
  if (tree == NULL) {
    return -ENOMEM;
  }
  *output = (uint64_t)(uintptr_t)tree;
  return 0;
}

int ummu_insert_seg(
    void *data, uint64_t seg_tree, uint64_t left, uint64_t size)
{
  struct kvaf_ummu_range_tree *tree =
      (struct kvaf_ummu_range_tree *)(uintptr_t)seg_tree;
  struct kvaf_ummu_range_node **cursor;
  struct kvaf_ummu_range_node *node;
  uint64_t right;

  if (tree == NULL || data == NULL ||
      kvaf_ummu_range_right(left, size, &right) != 0) {
    return -EINVAL;
  }
  cursor = &tree->head;
  while (*cursor != NULL && (*cursor)->right < left) {
    cursor = &(*cursor)->next;
  }
  if (*cursor != NULL && (*cursor)->left <= right) {
    return -EEXIST;
  }
  node = calloc(1, sizeof(*node));
  if (node == NULL) {
    return -ENOMEM;
  }
  node->left = left;
  node->right = right;
  node->data = data;
  node->next = *cursor;
  *cursor = node;
  return 0;
}

void *ummu_find_seg(
    uint64_t left, uint64_t size, uint64_t seg_tree, int *comparison)
{
  struct kvaf_ummu_range_tree *tree =
      (struct kvaf_ummu_range_tree *)(uintptr_t)seg_tree;
  struct kvaf_ummu_range_node *node;
  uint64_t right;

  if (comparison == NULL) {
    return NULL;
  }
  *comparison = RANGE_NOT_EXIST;
  if (tree == NULL || kvaf_ummu_range_right(left, size, &right) != 0) {
    return NULL;
  }
  for (node = tree->head; node != NULL; node = node->next) {
    if (node->right < left) {
      continue;
    }
    if (node->left > right) {
      break;
    }
    if (node->left == left && node->right == right) {
      *comparison = RANGE_MATCH;
      return node->data;
    }
    *comparison = RANGE_OVERLAP;
    return NULL;
  }
  return NULL;
}

int ummu_delete_seg(
    uint64_t left, uint64_t size, uint64_t seg_tree,
    clear_seg_node cleaner)
{
  struct kvaf_ummu_range_tree *tree =
      (struct kvaf_ummu_range_tree *)(uintptr_t)seg_tree;
  struct kvaf_ummu_range_node **cursor;
  struct kvaf_ummu_range_node *node;
  uint64_t right;

  if (tree == NULL || kvaf_ummu_range_right(left, size, &right) != 0) {
    return -EINVAL;
  }
  cursor = &tree->head;
  while (*cursor != NULL) {
    node = *cursor;
    if (node->left == left && node->right == right) {
      *cursor = node->next;
      if (node->data != NULL) {
        if (cleaner != NULL) {
          cleaner(node->data);
        }
        free(node->data);
      }
      free(node);
      return 0;
    }
    if (node->left > right) {
      break;
    }
    cursor = &node->next;
  }
  return -ENOENT;
}

void ummu_destroy_seg_tree(uint64_t *seg_tree, clear_seg_node cleaner)
{
  struct kvaf_ummu_range_tree *tree;
  struct kvaf_ummu_range_node *node;

  if (seg_tree == NULL || *seg_tree == 0) {
    return;
  }
  tree = (struct kvaf_ummu_range_tree *)(uintptr_t)*seg_tree;
  while (tree->head != NULL) {
    node = tree->head;
    tree->head = node->next;
    if (node->data != NULL) {
      if (cleaner != NULL) {
        cleaner(node->data);
      }
      free(node->data);
    }
    free(node);
  }
  free(tree);
  *seg_tree = 0;
}

int kvaf_source_ummu_seg_tree_is_empty(uint64_t seg_tree)
{
  const struct kvaf_ummu_range_tree *tree =
      (const struct kvaf_ummu_range_tree *)(uintptr_t)seg_tree;

  return tree != NULL && tree->head == NULL;
}
