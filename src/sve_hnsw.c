#include "sve_hybrid/sve_hnsw.h"
#include "sve_hybrid/sve_numeric.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define SVE_HNSW_MAX_LEVEL 15U
#define SVE_HNSW_MAGIC 0x48455653U
#define SVE_HNSW_VERSION_LEGACY 1U
#define SVE_HNSW_VERSION 2U
#define SVE_SNAPSHOT_HEADER_BYTES 60U
#define SVE_SNAPSHOT_NODE_BYTES 17U
#define SVE_SNAPSHOT_CRC_BYTES 4U
#define SVE_SLOT_EMPTY UINT32_MAX
#define SVE_SLOT_TOMBSTONE (UINT32_MAX - 1U)
#define SVE_MAX_NODE_COUNT (UINT32_MAX - 2U)

typedef struct sve_hnsw_node {
  int64_t rowid;
  uint32_t level;
  bool alive;
  float sq_scale;
  double norm;
  uint8_t *vec;
  uint32_t *links;
} sve_hnsw_node_t;

typedef struct sve_map_slot {
  int64_t rowid;
  uint32_t val; /* node index, SVE_SLOT_EMPTY, or SVE_SLOT_TOMBSTONE */
} sve_map_slot_t;

typedef struct sve_cand {
  uint32_t node_idx;
  double dist;
} sve_cand_t;

typedef struct sve_heap {
  sve_cand_t *items;
  size_t size;
  size_t cap;
} sve_heap_t;

typedef struct sve_layer_work {
  uint64_t *visited;
  size_t visited_cap;
  sve_heap_t candidates;
  sve_heap_t results;
} sve_layer_work_t;

struct sve_hnsw {
  sve_hnsw_config_t cfg;
  size_t vec_stride;
  bool use_sq8;
  uint64_t rng_state;

  sve_hnsw_node_t *nodes;
  uint32_t node_count;
  uint32_t node_cap;
  size_t live_count;
  uint32_t entry_point;
  uint32_t max_level;

  sve_map_slot_t *map_slots;
  size_t map_cap;
  size_t map_used;
  size_t map_tombstones;
  sve_layer_work_t insert_work;
};

static bool array_fits(size_t count, size_t elem_size) {
  return count <= SIZE_MAX / elem_size;
}

static uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31U);
}

static uint64_t rng_next(uint64_t *state) {
  uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
  if (z == 0ULL) {
    z = 0x9E3779B97F4A7C15ULL;
  }
  return splitmix64(z);
}

static uint32_t random_level(sve_hnsw_t *idx) {
  uint64_t r = rng_next(&idx->rng_state);
  /* Map 53-bit integer to (0, 1] safely */
  double u = ((double)(r >> 11U) + 1.0) / 9007199254740992.0;
  double ml = 1.0 / log((double)idx->cfg.m);
  double lv = -log(u) * ml;
  if (lv < 0.0) {
    lv = 0.0;
  }
  uint32_t level = (uint32_t)lv;
  if (level > SVE_HNSW_MAX_LEVEL) {
    level = SVE_HNSW_MAX_LEVEL;
  }
  return level;
}

static size_t layer_max_deg(uint32_t m, uint32_t layer) {
  return (layer == 0U) ? (size_t)(m * 2U) : (size_t)m;
}

static size_t total_link_u32_count(uint32_t m, uint32_t level) {
  /* Layer 0: [count, 2*M slots]; Layer 1..level: [count, M slots] */
  return (1U + (size_t)(m * 2U)) + (size_t)level * (1U + (size_t)m);
}

static uint32_t *node_layer_ptr(const sve_hnsw_node_t *node, uint32_t m,
                                uint32_t layer) {
  if (!node->links || layer > node->level) {
    return NULL;
  }
  if (layer == 0U) {
    return node->links;
  }
  size_t off = (1U + (size_t)(m * 2U)) + (size_t)(layer - 1U) * (1U + (size_t)m);
  return node->links + off;
}

static void heap_swap(sve_cand_t *a, sve_cand_t *b) {
  sve_cand_t tmp = *a;
  *a = *b;
  *b = tmp;
}

static int heap_reserve(sve_heap_t *h, size_t cap) {
  if (cap <= h->cap) {
    return SVE_OK;
  }
  if (!array_fits(cap, sizeof(*h->items))) {
    return SVE_ERR_LIMIT;
  }
  sve_cand_t *items = (sve_cand_t *)realloc(h->items, cap * sizeof(*items));
  if (!items) {
    return SVE_ERR_NOMEM;
  }
  h->items = items;
  h->cap = cap;
  return SVE_OK;
}

static void heap_free(sve_heap_t *h) {
  free(h->items);
  h->items = NULL;
  h->size = 0U;
  h->cap = 0U;
}

static int heap_ensure(sve_heap_t *h) {
  if (h->size < h->cap) {
    return SVE_OK;
  }
  if (h->cap > SIZE_MAX / 2U) {
    return SVE_ERR_LIMIT;
  }
  return heap_reserve(h, h->cap > 0U ? h->cap * 2U : 16U);
}

static void layer_work_free(sve_layer_work_t *work) {
  free(work->visited);
  heap_free(&work->candidates);
  heap_free(&work->results);
}

static int heap_push_min(sve_heap_t *h, sve_cand_t c) {
  int rc = heap_ensure(h);
  if (rc != SVE_OK) {
    return rc;
  }
  size_t i = h->size++;
  h->items[i] = c;
  while (i > 0U) {
    size_t p = (i - 1U) / 2U;
    if (h->items[p].dist <= h->items[i].dist) {
      break;
    }
    heap_swap(&h->items[p], &h->items[i]);
    i = p;
  }
  return SVE_OK;
}

static sve_cand_t heap_pop_min(sve_heap_t *h) {
  sve_cand_t top = h->items[0];
  h->items[0] = h->items[--h->size];
  size_t i = 0U;
  for (;;) {
    size_t l = 2U * i + 1U;
    size_t r = 2U * i + 2U;
    size_t best = i;
    if (l < h->size && h->items[l].dist < h->items[best].dist) {
      best = l;
    }
    if (r < h->size && h->items[r].dist < h->items[best].dist) {
      best = r;
    }
    if (best == i) {
      break;
    }
    heap_swap(&h->items[i], &h->items[best]);
    i = best;
  }
  return top;
}

static int heap_push_max(sve_heap_t *h, sve_cand_t c) {
  int rc = heap_ensure(h);
  if (rc != SVE_OK) {
    return rc;
  }
  size_t i = h->size++;
  h->items[i] = c;
  while (i > 0U) {
    size_t p = (i - 1U) / 2U;
    if (h->items[p].dist >= h->items[i].dist) {
      break;
    }
    heap_swap(&h->items[p], &h->items[i]);
    i = p;
  }
  return SVE_OK;
}

static sve_cand_t heap_pop_max(sve_heap_t *h) {
  sve_cand_t top = h->items[0];
  h->items[0] = h->items[--h->size];
  size_t i = 0U;
  for (;;) {
    size_t l = 2U * i + 1U;
    size_t r = 2U * i + 2U;
    size_t best = i;
    if (l < h->size && h->items[l].dist > h->items[best].dist) {
      best = l;
    }
    if (r < h->size && h->items[r].dist > h->items[best].dist) {
      best = r;
    }
    if (best == i) {
      break;
    }
    heap_swap(&h->items[i], &h->items[best]);
    i = best;
  }
  return top;
}

static int cmp_cand_asc(const void *pa, const void *pb) {
  const sve_cand_t *a = (const sve_cand_t *)pa;
  const sve_cand_t *b = (const sve_cand_t *)pb;
  if (a->dist < b->dist) {
    return -1;
  }
  if (a->dist > b->dist) {
    return 1;
  }
  if (a->node_idx < b->node_idx) {
    return -1;
  }
  if (a->node_idx > b->node_idx) {
    return 1;
  }
  return 0;
}

static int cmp_hit_asc(const void *pa, const void *pb) {
  const sve_search_hit_t *a = (const sve_search_hit_t *)pa;
  const sve_search_hit_t *b = (const sve_search_hit_t *)pb;
  if (a->score_or_dist < b->score_or_dist) {
    return -1;
  }
  if (a->score_or_dist > b->score_or_dist) {
    return 1;
  }
  if (a->rowid < b->rowid) {
    return -1;
  }
  if (a->rowid > b->rowid) {
    return 1;
  }
  return 0;
}

static int map_init(sve_hnsw_t *idx, size_t cap) {
  if (!array_fits(cap, sizeof(*idx->map_slots))) {
    return SVE_ERR_LIMIT;
  }
  idx->map_slots = (sve_map_slot_t *)malloc(cap * sizeof(sve_map_slot_t));
  if (!idx->map_slots) {
    return SVE_ERR_NOMEM;
  }
  for (size_t i = 0; i < cap; i++) {
    idx->map_slots[i].rowid = 0;
    idx->map_slots[i].val = SVE_SLOT_EMPTY;
  }
  idx->map_cap = cap;
  idx->map_used = 0U;
  idx->map_tombstones = 0U;
  return SVE_OK;
}

static int map_rehash(sve_hnsw_t *idx, size_t new_cap) {
  if (!array_fits(new_cap, sizeof(*idx->map_slots))) {
    return SVE_ERR_LIMIT;
  }
  sve_map_slot_t *new_slots =
      (sve_map_slot_t *)malloc(new_cap * sizeof(sve_map_slot_t));
  if (!new_slots) {
    return SVE_ERR_NOMEM;
  }
  for (size_t i = 0; i < new_cap; i++) {
    new_slots[i].rowid = 0;
    new_slots[i].val = SVE_SLOT_EMPTY;
  }
  size_t mask = new_cap - 1U;
  size_t used = 0U;
  for (size_t i = 0; i < idx->map_cap; i++) {
    uint32_t v = idx->map_slots[i].val;
    if (v != SVE_SLOT_EMPTY && v != SVE_SLOT_TOMBSTONE) {
      size_t pos = (size_t)splitmix64((uint64_t)idx->map_slots[i].rowid) & mask;
      while (new_slots[pos].val != SVE_SLOT_EMPTY) {
        pos = (pos + 1U) & mask;
      }
      new_slots[pos].rowid = idx->map_slots[i].rowid;
      new_slots[pos].val = v;
      used++;
    }
  }
  free(idx->map_slots);
  idx->map_slots = new_slots;
  idx->map_cap = new_cap;
  idx->map_used = used;
  idx->map_tombstones = 0U;
  return SVE_OK;
}

static bool map_lookup(const sve_hnsw_t *idx, int64_t rowid,
                       uint32_t *out_node_idx) {
  if (!idx->map_slots || idx->map_cap == 0U) {
    return false;
  }
  size_t mask = idx->map_cap - 1U;
  size_t pos = (size_t)splitmix64((uint64_t)rowid) & mask;
  for (size_t step = 0U; step < idx->map_cap; step++) {
    uint32_t v = idx->map_slots[pos].val;
    if (v == SVE_SLOT_EMPTY) {
      return false;
    }
    if (v != SVE_SLOT_TOMBSTONE && idx->map_slots[pos].rowid == rowid) {
      if (out_node_idx) {
        *out_node_idx = v;
      }
      return true;
    }
    pos = (pos + 1U) & mask;
  }
  return false;
}

static int map_put(sve_hnsw_t *idx, int64_t rowid, uint32_t node_idx) {
  if (idx->map_used + idx->map_tombstones + 1U >=
      idx->map_cap - idx->map_cap / 4U) {
    size_t target = idx->map_cap;
    if (idx->map_used >= idx->map_cap / 4U) {
      if (target > SIZE_MAX / 2U) {
        return SVE_ERR_LIMIT;
      }
      target *= 2U;
    }
    int rc = map_rehash(idx, target);
    if (rc != SVE_OK) {
      return rc;
    }
  }
  size_t mask = idx->map_cap - 1U;
  size_t pos = (size_t)splitmix64((uint64_t)rowid) & mask;
  size_t first_tomb = SIZE_MAX;
  for (;;) {
    uint32_t v = idx->map_slots[pos].val;
    if (v == SVE_SLOT_EMPTY) {
      size_t ins = (first_tomb != SIZE_MAX) ? first_tomb : pos;
      if (idx->map_slots[ins].val == SVE_SLOT_TOMBSTONE) {
        idx->map_tombstones--;
      }
      idx->map_slots[ins].rowid = rowid;
      idx->map_slots[ins].val = node_idx;
      idx->map_used++;
      return SVE_OK;
    }
    if (v == SVE_SLOT_TOMBSTONE && first_tomb == SIZE_MAX) {
      first_tomb = pos;
    } else if (v != SVE_SLOT_TOMBSTONE && idx->map_slots[pos].rowid == rowid) {
      return SVE_ERR_INVALID;
    }
    pos = (pos + 1U) & mask;
  }
}

static bool map_remove(sve_hnsw_t *idx, int64_t rowid) {
  size_t mask = idx->map_cap - 1U;
  size_t pos = (size_t)splitmix64((uint64_t)rowid) & mask;
  for (size_t step = 0U; step < idx->map_cap; step++) {
    uint32_t v = idx->map_slots[pos].val;
    if (v == SVE_SLOT_EMPTY) {
      return false;
    }
    if (v != SVE_SLOT_TOMBSTONE && idx->map_slots[pos].rowid == rowid) {
      idx->map_slots[pos].val = SVE_SLOT_TOMBSTONE;
      idx->map_used--;
      idx->map_tombstones++;
      return true;
    }
    pos = (pos + 1U) & mask;
  }
  return false;
}

typedef struct query_repr {
  const void *raw_vec;
  int8_t *sq8_buf;
  float sq8_scale;
  bool is_sq8;
  double norm;
} query_repr_t;

/* A cosine norm depends only on the vector. Compute it once on ingest
   or query preparation, using the scalar kernel's accumulation order. */
static double cosine_norm(const sve_hnsw_t *idx, const void *vector) {
  double sum = 0.0;
  if (idx->use_sq8 || idx->cfg.elem_type == SVE_ELEM_INT8) {
    const int8_t *values = (const int8_t *)vector;
    int32_t squares = 0;
    for (size_t i = 0U; i < idx->cfg.dims; i++) {
      int32_t v = (int32_t)values[i];
      squares += v * v;
    }
    sum = (double)squares;
  } else {
    const float *values = (const float *)vector;
    for (size_t i = 0U; i < idx->cfg.dims; i++) {
      double v = (double)values[i];
      sum += v * v;
    }
  }
  return sqrt(sum);
}

static int prepare_query_repr(const sve_hnsw_t *idx, const void *query_vec,
                              query_repr_t *out) {
  *out = (query_repr_t){query_vec, NULL, 0.0f, idx->use_sq8, 0.0};
  if (idx->use_sq8) {
    out->sq8_buf = (int8_t *)malloc(idx->cfg.dims);
    if (!out->sq8_buf) {
      return SVE_ERR_NOMEM;
    }
    int rc = sve_quantize_f32_to_sq8((const float *)query_vec, idx->cfg.dims,
                                     out->sq8_buf, &out->sq8_scale);
    if (rc != SVE_OK) {
      free(out->sq8_buf);
      out->sq8_buf = NULL;
      return rc;
    }
  } else if (idx->cfg.elem_type == SVE_ELEM_FLOAT32 &&
             !sve_f32_array_is_finite((const float *)query_vec, idx->cfg.dims)) {
    return SVE_ERR_INVALID;
  }
  if (idx->cfg.metric == SVE_METRIC_COSINE) {
    out->norm = cosine_norm(idx, idx->use_sq8 ? (const void *)out->sq8_buf
                                            : query_vec);
  }
  return SVE_OK;
}

static void free_query_repr(query_repr_t *qr) {
  free(qr->sq8_buf);
}

static double cosine_distance(const sve_hnsw_t *idx, const query_repr_t *qr,
                               const sve_hnsw_node_t *node) {
  if (qr->norm == 0.0 || node->norm == 0.0) {
    return 1.0;
  }
  double dot = 0.0;
  if (idx->use_sq8 || idx->cfg.elem_type == SVE_ELEM_INT8) {
    const int8_t *a = idx->use_sq8 ? qr->sq8_buf : (const int8_t *)qr->raw_vec;
    const int8_t *b = (const int8_t *)node->vec;
    int32_t product = 0;
    for (size_t i = 0U; i < idx->cfg.dims; i++) {
      product += (int32_t)a[i] * (int32_t)b[i];
    }
    dot = (double)product;
  } else {
    const float *a = (const float *)qr->raw_vec;
    const float *b = (const float *)node->vec;
    for (size_t i = 0U; i < idx->cfg.dims; i++) {
      dot += (double)a[i] * (double)b[i];
    }
  }
  double sim = dot / (qr->norm * node->norm);
  if (sim > 1.0) {
    sim = 1.0;
  } else if (sim < -1.0) {
    sim = -1.0;
  }
  return 1.0 - sim;
}

static double dist_query_node(const sve_hnsw_t *idx, const query_repr_t *qr,
                              uint32_t node_idx) {
  const sve_hnsw_node_t *nd = &idx->nodes[node_idx];
  if (idx->cfg.metric == SVE_METRIC_COSINE) {
    return cosine_distance(idx, qr, nd);
  }
  if (qr->is_sq8) {
    const int8_t *nb = (const int8_t *)nd->vec;
    if (idx->cfg.metric == SVE_METRIC_L2) {
      return sve_distance_sq8_l2(qr->sq8_buf, qr->sq8_scale, nb, nd->sq_scale,
                                 idx->cfg.dims);
    }
    double acc = 0.0;
    for (size_t i = 0; i < idx->cfg.dims; i++) {
      double va = (double)qr->sq8_buf[i] * (double)qr->sq8_scale;
      double vb = (double)nb[i] * (double)nd->sq_scale;
      acc += (idx->cfg.metric == SVE_METRIC_L1) ? fabs(va - vb) : (va * vb);
    }
    return (idx->cfg.metric == SVE_METRIC_IP) ? -acc : acc;
  }
  double d = 0.0;
  int rc = sve_compute_distance(idx->cfg.elem_type, idx->cfg.metric,
                                qr->raw_vec, nd->vec, idx->cfg.dims, &d);
  return (rc == SVE_OK) ? d : DBL_MAX;
}

static query_repr_t node_query_repr(const sve_hnsw_t *idx,
                                     const sve_hnsw_node_t *node) {
  return (query_repr_t){node->vec, (int8_t *)node->vec, node->sq_scale,
                        idx->use_sq8, node->norm};
}

static double dist_node_node(const sve_hnsw_t *idx, uint32_t a_idx,
                             uint32_t b_idx) {
  query_repr_t qr = node_query_repr(idx, &idx->nodes[a_idx]);
  return dist_query_node(idx, &qr, b_idx);
}

sve_hnsw_config_t sve_hnsw_default_config(size_t dims,
                                          sve_elem_type_t elem_type,
                                          sve_metric_t metric,
                                          sve_index_kind_t index_kind) {
  sve_hnsw_config_t cfg;
  cfg.dims = dims;
  cfg.elem_type = elem_type;
  cfg.metric = metric;
  cfg.index_kind = index_kind;
  cfg.m = SVE_DEFAULT_HNSW_M;
  cfg.ef_construction = SVE_DEFAULT_HNSW_EF_CONSTRUCTION;
  cfg.ef_search = SVE_DEFAULT_HNSW_EF_SEARCH;
  cfg.rng_seed = 0x535645484E5357ULL;
  return cfg;
}

bool sve_hnsw_config_is_valid(const sve_hnsw_config_t *cfg) {
  if (!cfg || cfg->dims == 0U || cfg->dims > SVE_MAX_TABLE_DIMS) {
    return false;
  }
  if (cfg->elem_type != SVE_ELEM_FLOAT32 && cfg->elem_type != SVE_ELEM_INT8 &&
      cfg->elem_type != SVE_ELEM_BIT) {
    return false;
  }
  if (cfg->index_kind != SVE_INDEX_FLAT && cfg->index_kind != SVE_INDEX_HNSW &&
      cfg->index_kind != SVE_INDEX_HNSW_SQ8) {
    return false;
  }
  if (cfg->metric == SVE_METRIC_HAMMING) {
    if (cfg->elem_type != SVE_ELEM_BIT) {
      return false;
    }
  } else if (cfg->metric != SVE_METRIC_L2 &&
             !((cfg->metric == SVE_METRIC_COSINE || cfg->metric == SVE_METRIC_L1) &&
               cfg->elem_type != SVE_ELEM_BIT) &&
             !(cfg->metric == SVE_METRIC_IP && cfg->elem_type == SVE_ELEM_FLOAT32)) {
    return false;
  }
  if (cfg->elem_type == SVE_ELEM_BIT && (cfg->dims % 8U) != 0U) {
    return false;
  }
  if (cfg->index_kind == SVE_INDEX_HNSW_SQ8 &&
      cfg->elem_type != SVE_ELEM_FLOAT32) {
    return false;
  }
  if (cfg->m < 2U || cfg->m > SVE_MAX_HNSW_M) {
    return false;
  }
  if (cfg->ef_construction < 4U || cfg->ef_construction > SVE_MAX_HNSW_EF ||
      cfg->ef_search < 1U || cfg->ef_search > SVE_MAX_HNSW_EF) {
    return false;
  }
  return true;
}

sve_hnsw_t *sve_hnsw_create(const sve_hnsw_config_t *cfg) {
  if (!sve_hnsw_config_is_valid(cfg)) {
    return NULL;
  }
  sve_hnsw_t *idx = (sve_hnsw_t *)calloc(1U, sizeof(sve_hnsw_t));
  if (!idx) {
    return NULL;
  }
  idx->cfg = *cfg;
  idx->use_sq8 = (cfg->index_kind == SVE_INDEX_HNSW_SQ8 &&
                  cfg->elem_type == SVE_ELEM_FLOAT32);
  idx->vec_stride = idx->use_sq8 ? cfg->dims
                                 : sve_elem_byte_size(cfg->elem_type, cfg->dims);
  idx->rng_state = cfg->rng_seed ? cfg->rng_seed : 1ULL;
  idx->entry_point = UINT32_MAX;
  idx->max_level = 0U;

  idx->node_cap = 64U;
  idx->nodes = (sve_hnsw_node_t *)calloc(idx->node_cap, sizeof(sve_hnsw_node_t));
  if (!idx->nodes || map_init(idx, 128U) != SVE_OK) {
    sve_hnsw_destroy(idx);
    return NULL;
  }
  return idx;
}

void sve_hnsw_destroy(sve_hnsw_t *idx) {
  if (!idx) {
    return;
  }
  if (idx->nodes) {
    for (uint32_t i = 0U; i < idx->node_count; i++) {
      free(idx->nodes[i].vec);
      free(idx->nodes[i].links);
    }
    free(idx->nodes);
  }
  free(idx->map_slots);
  layer_work_free(&idx->insert_work);
  free(idx);
}

static bool visited_test_and_set(uint64_t *bitset, uint32_t node_idx) {
  uint64_t mask = 1ULL << (node_idx & 63U);
  size_t word = (size_t)(node_idx >> 6U);
  bool was = (bitset[word] & mask) != 0ULL;
  bitset[word] |= mask;
  return was;
}

static uint32_t greedy_search_layer_query(const sve_hnsw_t *idx,
                                          const query_repr_t *qr,
                                          uint32_t ep, uint32_t layer,
                                          double *out_ep_dist) {
  uint32_t curr = ep;
  double curr_dist = dist_query_node(idx, qr, curr);
  for (;;) {
    bool changed = false;
    const uint32_t *lp = node_layer_ptr(&idx->nodes[curr], idx->cfg.m, layer);
    if (!lp) {
      break;
    }
    uint32_t cnt = lp[0];
    for (uint32_t i = 1U; i <= cnt; i++) {
      uint32_t nbr = lp[i];
      if (nbr >= idx->node_count || !idx->nodes[nbr].alive) {
        continue;
      }
      double d = dist_query_node(idx, qr, nbr);
      if (d < curr_dist) {
        curr_dist = d;
        curr = nbr;
        changed = true;
      }
    }
    if (!changed) {
      break;
    }
  }
  if (out_ep_dist) {
    *out_ep_dist = curr_dist;
  }
  return curr;
}

static int search_layer_ef(const sve_hnsw_t *idx, const query_repr_t *qr,
                           uint32_t ep, uint32_t layer, uint32_t ef,
                           sve_layer_work_t *work) {
  sve_heap_t *cands = &work->candidates;
  sve_heap_t *results = &work->results;
  cands->size = 0U;
  results->size = 0U;
  if (ep >= idx->node_count || !idx->nodes[ep].alive) {
    return SVE_OK;
  }
  if (ef > idx->live_count) {
    ef = (uint32_t)idx->live_count;
  }
  size_t words = idx->node_count / 64U + (idx->node_count % 64U != 0U);
  if (words > work->visited_cap) {
    size_t cap = work->visited_cap == 0U ? 1U : work->visited_cap * 2U;
    if (cap < words) {
      cap = words;
    }
    if (!array_fits(cap, sizeof(*work->visited))) {
      return SVE_ERR_LIMIT;
    }
    uint64_t *tmp = (uint64_t *)realloc(work->visited, cap * sizeof(*tmp));
    if (!tmp) {
      return SVE_ERR_NOMEM;
    }
    work->visited = tmp;
    work->visited_cap = cap;
  }
  memset(work->visited, 0, words * sizeof(*work->visited));
  int rc = heap_reserve(cands, (size_t)ef * 2U + 16U);
  if (rc != SVE_OK) {
    return rc;
  }
  rc = heap_reserve(results, (size_t)ef + 1U);
  if (rc != SVE_OK) {
    return rc;
  }

  sve_cand_t ep_c = {ep, dist_query_node(idx, qr, ep)};
  (void)visited_test_and_set(work->visited, ep);
  cands->items[cands->size++] = ep_c;
  results->items[results->size++] = ep_c;

  while (cands->size > 0U) {
    sve_cand_t curr = heap_pop_min(cands);
    if (results->size >= ef && curr.dist > results->items[0].dist) {
      break;
    }
    const uint32_t *lp =
        node_layer_ptr(&idx->nodes[curr.node_idx], idx->cfg.m, layer);
    if (!lp) {
      continue;
    }
    for (uint32_t i = 1U; i <= lp[0]; i++) {
      uint32_t nbr = lp[i];
      if (nbr >= idx->node_count || visited_test_and_set(work->visited, nbr) ||
          !idx->nodes[nbr].alive) {
        continue;
      }
      double d = dist_query_node(idx, qr, nbr);
      if (results->size < ef || d < results->items[0].dist) {
        sve_cand_t nc = {nbr, d};
        rc = heap_push_min(cands, nc);
        if (rc != SVE_OK) {
          return rc;
        }
        /* The result heap reserved ef + 1 slots. */
        (void)heap_push_max(results, nc);
        if (results->size > ef) {
          (void)heap_pop_max(results);
        }
      }
    }
  }

  qsort(results->items, results->size, sizeof(*results->items), cmp_cand_asc);
  return SVE_OK;
}

static size_t select_neighbors_heuristic(const sve_hnsw_t *idx,
                                         sve_cand_t *cands, size_t n_cands,
                                         size_t max_deg, uint32_t *out_ids) {
  size_t selected = 0U;
  size_t n_discard = 0U;

  /* The candidates are sorted. Save rejected entries in their consumed prefix. */
  for (size_t i = 0; i < n_cands && selected < max_deg; i++) {
    uint32_t c_idx = cands[i].node_idx;
    bool good = true;
    for (size_t j = 0; j < selected; j++) {
      if (dist_node_node(idx, c_idx, out_ids[j]) < cands[i].dist) {
        good = false;
        break;
      }
    }
    if (good) {
      out_ids[selected++] = c_idx;
    } else {
      cands[n_discard++] = cands[i];
    }
  }

  for (size_t i = 0; i < n_discard && selected < max_deg; i++) {
    out_ids[selected++] = cands[i].node_idx;
  }
  return selected;
}

static void connect_reverse_edge(sve_hnsw_t *idx, uint32_t nbr_idx,
                                 uint32_t new_idx, uint32_t layer) {
  uint32_t *lp = node_layer_ptr(&idx->nodes[nbr_idx], idx->cfg.m, layer);
  size_t max_d = layer_max_deg(idx->cfg.m, layer);
  uint32_t kept = 0U;
  for (uint32_t i = 1U; i <= lp[0]; i++) {
    uint32_t id = lp[i];
    if (idx->nodes[id].alive) {
      lp[1U + kept++] = id;
    }
  }

  if ((size_t)kept < max_d) {
    lp[1U + kept] = new_idx;
    lp[0] = kept + 1U;
    return;
  }

  sve_cand_t cands[SVE_MAX_HNSW_M * 2U + 1U];
  for (uint32_t i = 0U; i < kept; i++) {
    uint32_t id = lp[1U + i];
    cands[i].node_idx = id;
    cands[i].dist = dist_node_node(idx, nbr_idx, id);
  }
  cands[kept].node_idx = new_idx;
  cands[kept].dist = dist_node_node(idx, nbr_idx, new_idx);
  size_t total = (size_t)kept + 1U;
  qsort(cands, total, sizeof(*cands), cmp_cand_asc);
  lp[0] = (uint32_t)select_neighbors_heuristic(idx, cands, total, max_d, lp + 1U);
}

int sve_hnsw_insert(sve_hnsw_t *idx, int64_t rowid, const void *vector_data) {
  if (!idx || !vector_data || idx->node_count >= SVE_MAX_NODE_COUNT) {
    return SVE_ERR_INVALID;
  }
  if (map_lookup(idx, rowid, NULL)) {
    return SVE_ERR_INVALID;
  }
  if (idx->cfg.elem_type == SVE_ELEM_FLOAT32 &&
      !sve_f32_array_is_finite((const float *)vector_data, idx->cfg.dims)) {
    return SVE_ERR_INVALID;
  }

  if (idx->node_count == idx->node_cap) {
    uint32_t next_cap = idx->node_cap <= SVE_MAX_NODE_COUNT / 2U
                            ? idx->node_cap * 2U : SVE_MAX_NODE_COUNT;
    if (!array_fits(next_cap, sizeof(*idx->nodes))) {
      return SVE_ERR_LIMIT;
    }
    sve_hnsw_node_t *tmp = (sve_hnsw_node_t *)realloc(
        idx->nodes, (size_t)next_cap * sizeof(*tmp));
    if (!tmp) {
      return SVE_ERR_NOMEM;
    }
    idx->nodes = tmp;
    idx->node_cap = next_cap;
  }

  uint32_t node_idx = idx->node_count;
  sve_hnsw_node_t *node = &idx->nodes[node_idx];
  memset(node, 0, sizeof(*node));
  node->vec = (uint8_t *)malloc(idx->vec_stride);
  if (!node->vec) {
    return SVE_ERR_NOMEM;
  }

  uint64_t prev_rng = idx->rng_state;
  node->rowid = rowid;
  node->alive = true;
  int rc = SVE_OK;
  if (idx->use_sq8) {
    rc = sve_quantize_f32_to_sq8((const float *)vector_data, idx->cfg.dims,
                                 (int8_t *)node->vec, &node->sq_scale);
    if (rc != SVE_OK) {
      goto fail;
    }
  } else {
    memcpy(node->vec, vector_data, idx->vec_stride);
  }

  if (idx->cfg.metric == SVE_METRIC_COSINE) {
    node->norm = cosine_norm(idx, node->vec);
  }

  uint32_t top_connect = 0U;
  if (idx->cfg.index_kind != SVE_INDEX_FLAT) {
    node->level = random_level(idx);
    size_t nu32 = total_link_u32_count(idx->cfg.m, node->level);
    node->links = (uint32_t *)calloc(nu32, sizeof(*node->links));
    if (!node->links) {
      rc = SVE_ERR_NOMEM;
      goto fail;
    }
    if (idx->entry_point != UINT32_MAX) {
      query_repr_t qr = node_query_repr(idx, node);
      uint32_t curr = idx->entry_point;
      for (uint32_t lv = idx->max_level; lv > node->level; lv--) {
        curr = greedy_search_layer_query(idx, &qr, curr, lv, NULL);
      }
      top_connect = node->level < idx->max_level ? node->level : idx->max_level;
      for (int lv_i = (int)top_connect; lv_i >= 0; lv_i--) {
        uint32_t lv = (uint32_t)lv_i;
        rc = search_layer_ef(idx, &qr, curr, lv, idx->cfg.ef_construction,
                             &idx->insert_work);
        if (rc != SVE_OK) {
          goto fail;
        }
        sve_heap_t *results = &idx->insert_work.results;
        if (results->size != 0U) {
          curr = results->items[0].node_idx;
          uint32_t *lp = node_layer_ptr(node, idx->cfg.m, lv);
          lp[0] = (uint32_t)select_neighbors_heuristic(
              idx, results->items, results->size, layer_max_deg(idx->cfg.m, lv),
              lp + 1U);
        }
      }
    }
  }

  /* All fallible work precedes publication and changes to existing edges. */
  rc = map_put(idx, rowid, node_idx);
  if (rc != SVE_OK) {
    goto fail;
  }
  idx->node_count++;
  idx->live_count++;
  if (node->links && idx->entry_point != UINT32_MAX) {
    for (uint32_t lv = 0U; lv <= top_connect; lv++) {
      const uint32_t *lp = node_layer_ptr(node, idx->cfg.m, lv);
      for (uint32_t i = 1U; i <= lp[0]; i++) {
        connect_reverse_edge(idx, lp[i], node_idx, lv);
      }
    }
  }
  if (idx->entry_point == UINT32_MAX || node->level > idx->max_level) {
    idx->max_level = node->level;
    idx->entry_point = node_idx;
  }
  return SVE_OK;

fail:
  free(node->vec);
  free(node->links);
  memset(node, 0, sizeof(*node));
  idx->rng_state = prev_rng;
  return rc;
}

int sve_hnsw_delete(sve_hnsw_t *idx, int64_t rowid) {
  if (!idx) {
    return SVE_ERR_INVALID;
  }
  uint32_t node_idx = 0U;
  if (!map_lookup(idx, rowid, &node_idx)) {
    return SVE_ERR_NOTFOUND;
  }
  if (!idx->nodes[node_idx].alive) {
    return SVE_ERR_NOTFOUND;
  }
  sve_hnsw_node_t *node = &idx->nodes[node_idx];
  node->alive = false;
  free(node->vec);
  node->vec = NULL;
  free(node->links);
  node->links = NULL;
  idx->live_count--;
  (void)map_remove(idx, rowid);

  if (idx->entry_point == node_idx) {
    idx->entry_point = UINT32_MAX;
    idx->max_level = 0U;
    for (uint32_t i = 0U; i < idx->node_count; i++) {
      if (idx->nodes[i].alive) {
        if (idx->entry_point == UINT32_MAX ||
            idx->nodes[i].level > idx->max_level) {
          idx->entry_point = i;
          idx->max_level = idx->nodes[i].level;
        }
      }
    }
  }
  return SVE_OK;
}

bool sve_hnsw_contains(const sve_hnsw_t *idx, int64_t rowid) {
  if (!idx) {
    return false;
  }
  uint32_t node_idx = 0U;
  return map_lookup(idx, rowid, &node_idx) && idx->nodes[node_idx].alive;
}

size_t sve_hnsw_live_count(const sve_hnsw_t *idx) {
  return idx ? idx->live_count : 0U;
}

static bool allowlist_contains(const int64_t *arr, size_t n, int64_t target) {
  size_t lo = 0U;
  size_t hi = n;
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2U;
    if (arr[mid] == target) {
      return true;
    }
    if (arr[mid] < target) {
      lo = mid + 1U;
    } else {
      hi = mid;
    }
  }
  return false;
}

int sve_hnsw_exact_search(const sve_hnsw_t *idx, const void *query_vec,
                          size_t k, const int64_t *allowed_rowids_sorted,
                          size_t n_allowed, sve_search_hit_t *out_hits,
                          size_t *out_count) {
  if (!idx || !query_vec || !out_count || (k > 0U && !out_hits) ||
      k > SVE_MAX_K) {
    return SVE_ERR_INVALID;
  }
  *out_count = 0U;
  if (k == 0U || idx->live_count == 0U ||
      (allowed_rowids_sorted != NULL && n_allowed == 0U)) {
    return SVE_OK;
  }

  query_repr_t qr;
  int rc = prepare_query_repr(idx, query_vec, &qr);
  if (rc != SVE_OK) {
    return rc;
  }

  size_t limit = idx->live_count;
  if (allowed_rowids_sorted && n_allowed < limit) {
    limit = n_allowed;
  }
  if (k > limit) {
    k = limit;
  }
  sve_heap_t top_k = {0};
  if (heap_reserve(&top_k, k + 1U) != SVE_OK) {
    free_query_repr(&qr);
    return SVE_ERR_NOMEM;
  }

  if (allowed_rowids_sorted != NULL) {
    for (size_t i = 0; i < n_allowed; i++) {
      uint32_t node_idx = 0U;
      if (!map_lookup(idx, allowed_rowids_sorted[i], &node_idx) ||
          !idx->nodes[node_idx].alive) {
        continue;
      }
      double d = dist_query_node(idx, &qr, node_idx);
      if (top_k.size < k || d < top_k.items[0].dist) {
        sve_cand_t c = {node_idx, d};
        if (heap_push_max(&top_k, c) != SVE_OK) {
          heap_free(&top_k);
          free_query_repr(&qr);
          return SVE_ERR_NOMEM;
        }
        if (top_k.size > k) {
          (void)heap_pop_max(&top_k);
        }
      }
    }
  } else {
    for (uint32_t i = 0U; i < idx->node_count; i++) {
      if (!idx->nodes[i].alive) {
        continue;
      }
      double d = dist_query_node(idx, &qr, i);
      if (top_k.size < k || d < top_k.items[0].dist) {
        sve_cand_t c = {i, d};
        if (heap_push_max(&top_k, c) != SVE_OK) {
          heap_free(&top_k);
          free_query_repr(&qr);
          return SVE_ERR_NOMEM;
        }
        if (top_k.size > k) {
          (void)heap_pop_max(&top_k);
        }
      }
    }
  }

  size_t emit = top_k.size;
  for (size_t i = 0; i < emit; i++) {
    out_hits[i].rowid = idx->nodes[top_k.items[i].node_idx].rowid;
    out_hits[i].score_or_dist = top_k.items[i].dist;
  }
  qsort(out_hits, emit, sizeof(sve_search_hit_t), cmp_hit_asc);
  *out_count = emit;

  heap_free(&top_k);
  free_query_repr(&qr);
  return SVE_OK;
}

int sve_hnsw_search_allowlist(const sve_hnsw_t *idx, const void *query_vec,
                              size_t k, uint32_t ef_override,
                              const int64_t *allowed_rowids_sorted,
                              size_t n_allowed, sve_search_hit_t *out_hits,
                              size_t *out_count) {
  if (!idx || !query_vec || !out_count || (k > 0U && !out_hits) ||
      k > SVE_MAX_K) {
    return SVE_ERR_INVALID;
  }
  *out_count = 0U;
  if (k == 0U || idx->live_count == 0U || idx->entry_point == UINT32_MAX ||
      (allowed_rowids_sorted != NULL && n_allowed == 0U)) {
    return SVE_OK;
  }

  if (idx->cfg.index_kind == SVE_INDEX_FLAT ||
      (allowed_rowids_sorted != NULL &&
       n_allowed <= SVE_EXACT_FILTER_THRESHOLD)) {
    return sve_hnsw_exact_search(idx, query_vec, k, allowed_rowids_sorted,
                                 n_allowed, out_hits, out_count);
  }

  size_t available = idx->live_count;
  if (allowed_rowids_sorted && n_allowed < available) {
    available = n_allowed;
  }
  if (k > available) {
    k = available;
  }
  if (k == available || k > SVE_MAX_HNSW_EF) {
    return sve_hnsw_exact_search(idx, query_vec, k, allowed_rowids_sorted,
                                 n_allowed, out_hits, out_count);
  }
  uint32_t ef = ef_override > 0U ? ef_override : idx->cfg.ef_search;
  if (ef > SVE_MAX_HNSW_EF) {
    ef = SVE_MAX_HNSW_EF;
  }
  if (ef < (uint32_t)k) {
    ef = (uint32_t)k;
  }
  if (allowed_rowids_sorted != NULL) {
    ef = ef > SVE_MAX_HNSW_EF / 4U ? SVE_MAX_HNSW_EF : ef * 4U;
  }

  query_repr_t qr;
  int rc = prepare_query_repr(idx, query_vec, &qr);
  if (rc != SVE_OK) {
    return rc;
  }

  uint32_t curr = idx->entry_point;
  for (uint32_t lv = idx->max_level; lv > 0U; lv--) {
    curr = greedy_search_layer_query(idx, &qr, curr, lv, NULL);
  }

  sve_layer_work_t work = {0};
  rc = search_layer_ef(idx, &qr, curr, 0U, ef, &work);
  size_t emit = 0U;
  if (rc == SVE_OK) {
    for (size_t i = 0; i < work.results.size && emit < k; i++) {
      const sve_cand_t *cand = &work.results.items[i];
      int64_t rid = idx->nodes[cand->node_idx].rowid;
      if (allowed_rowids_sorted &&
          !allowlist_contains(allowed_rowids_sorted, n_allowed, rid)) {
        continue;
      }
      out_hits[emit].rowid = rid;
      out_hits[emit].score_or_dist = cand->dist;
      emit++;
    }
  }
  layer_work_free(&work);
  free_query_repr(&qr);
  if (rc != SVE_OK) {
    return rc;
  }
  if (emit < k) {
    return sve_hnsw_exact_search(idx, query_vec, k, allowed_rowids_sorted,
                                 n_allowed, out_hits, out_count);
  }

  qsort(out_hits, emit, sizeof(*out_hits), cmp_hit_asc);
  *out_count = emit;
  return SVE_OK;
}

int sve_hnsw_search(const sve_hnsw_t *idx, const void *query_vec, size_t k,
                    uint32_t ef_override, sve_search_hit_t *out_hits,
                    size_t *out_count) {
  return sve_hnsw_search_allowlist(idx, query_vec, k, ef_override, NULL, 0U,
                                   out_hits, out_count);
}

static void put_u8(uint8_t **p, uint8_t v) {
  (*p)[0] = v;
  *p += 1U;
}

static void put_u32(uint8_t **p, uint32_t v) {
  uint8_t *b = *p;
  b[0] = (uint8_t)(v & 0xFFU);
  b[1] = (uint8_t)((v >> 8U) & 0xFFU);
  b[2] = (uint8_t)((v >> 16U) & 0xFFU);
  b[3] = (uint8_t)((v >> 24U) & 0xFFU);
  *p += 4U;
}

static void put_u64(uint8_t **p, uint64_t v) {
  put_u32(p, (uint32_t)(v & 0xFFFFFFFFULL));
  put_u32(p, (uint32_t)(v >> 32U));
}

static void put_f32(uint8_t **p, float v) {
  uint32_t bits = 0U;
  memcpy(&bits, &v, sizeof(bits));
  put_u32(p, bits);
}

static bool get_u8(const uint8_t **p, const uint8_t *end, uint8_t *out) {
  if ((size_t)(end - *p) < 1U) {
    return false;
  }
  *out = (*p)[0];
  *p += 1U;
  return true;
}

static bool get_u32(const uint8_t **p, const uint8_t *end, uint32_t *out) {
  if ((size_t)(end - *p) < 4U) {
    return false;
  }
  const uint8_t *b = *p;
  *out = (uint32_t)b[0] | ((uint32_t)b[1] << 8U) | ((uint32_t)b[2] << 16U) |
         ((uint32_t)b[3] << 24U);
  *p += 4U;
  return true;
}

static bool get_u64(const uint8_t **p, const uint8_t *end, uint64_t *out) {
  uint32_t lo = 0U, hi = 0U;
  if (!get_u32(p, end, &lo) || !get_u32(p, end, &hi)) {
    return false;
  }
  *out = (uint64_t)lo | ((uint64_t)hi << 32U);
  return true;
}

static bool get_f32(const uint8_t **p, const uint8_t *end, float *out) {
  uint32_t bits = 0U;
  if (!get_u32(p, end, &bits)) {
    return false;
  }
  memcpy(out, &bits, sizeof(*out));
  return true;
}

int sve_hnsw_serialize(const sve_hnsw_t *idx, uint8_t **out_buf,
                       size_t *out_len) {
  if (!idx || !out_buf || !out_len) {
    return SVE_ERR_INVALID;
  }
  *out_buf = NULL;
  *out_len = 0U;

  size_t need = SVE_SNAPSHOT_HEADER_BYTES + SVE_SNAPSHOT_CRC_BYTES;
  for (uint32_t i = 0U; i < idx->node_count; i++) {
    const sve_hnsw_node_t *nd = &idx->nodes[i];
    size_t record_bytes = SVE_SNAPSHOT_NODE_BYTES;
    if (nd->alive) {
      record_bytes += idx->vec_stride;
      if (nd->links) {
        for (uint32_t lv = 0U; lv <= nd->level; lv++) {
          const uint32_t *lp = node_layer_ptr(nd, idx->cfg.m, lv);
          record_bytes += sizeof(uint32_t);
          for (uint32_t j = 1U; j <= lp[0]; j++) {
            if (idx->nodes[lp[j]].alive) {
              record_bytes += sizeof(uint32_t);
            }
          }
        }
      }
    }
    if (need > (size_t)PTRDIFF_MAX - record_bytes) {
      return SVE_ERR_LIMIT;
    }
    need += record_bytes;
  }

  uint8_t *buf = (uint8_t *)malloc(need);
  if (!buf) {
    return SVE_ERR_NOMEM;
  }
  uint8_t *w = buf;
  put_u32(&w, SVE_HNSW_MAGIC);
  put_u32(&w, SVE_HNSW_VERSION);
  put_u32(&w, (uint32_t)idx->cfg.dims);
  put_u32(&w, (uint32_t)idx->cfg.elem_type);
  put_u32(&w, (uint32_t)idx->cfg.metric);
  put_u32(&w, (uint32_t)idx->cfg.index_kind);
  put_u32(&w, idx->cfg.m);
  put_u32(&w, idx->cfg.ef_construction);
  put_u32(&w, idx->cfg.ef_search);
  put_u64(&w, idx->rng_state);
  put_u32(&w, idx->node_count);
  put_u32(&w, (uint32_t)idx->live_count);
  put_u32(&w, idx->entry_point);
  put_u32(&w, idx->max_level);

  for (uint32_t i = 0U; i < idx->node_count; i++) {
    const sve_hnsw_node_t *nd = &idx->nodes[i];
    put_u64(&w, (uint64_t)nd->rowid);
    put_u32(&w, nd->level);
    put_u8(&w, nd->alive ? 1U : 0U);
    put_f32(&w, nd->alive ? nd->sq_scale : 0.0f);
    if (!nd->alive) {
      continue;
    }
    if (!idx->use_sq8 && idx->cfg.elem_type == SVE_ELEM_FLOAT32) {
      const float *vec = (const float *)nd->vec;
      for (size_t j = 0U; j < idx->cfg.dims; j++) {
        put_f32(&w, vec[j]);
      }
    } else {
      memcpy(w, nd->vec, idx->vec_stride);
      w += idx->vec_stride;
    }
    if (nd->links) {
      for (uint32_t lv = 0U; lv <= nd->level; lv++) {
        const uint32_t *lp = node_layer_ptr(nd, idx->cfg.m, lv);
        uint8_t *count_pos = w;
        w += sizeof(uint32_t);
        uint32_t count = 0U;
        for (uint32_t j = 1U; j <= lp[0]; j++) {
          if (idx->nodes[lp[j]].alive) {
            put_u32(&w, lp[j]);
            count++;
          }
        }
        put_u32(&count_pos, count);
      }
    }
  }

  size_t payload_len = (size_t)(w - buf);
  uint32_t crc = sve_crc32c(buf, payload_len);
  put_u32(&w, crc);

  *out_buf = buf;
  *out_len = (size_t)(w - buf);
  return SVE_OK;
}

static bool snapshot_links_are_valid(sve_hnsw_t *idx, bool legacy) {
  for (uint32_t i = 0U; i < idx->node_count; i++) {
    const sve_hnsw_node_t *node = &idx->nodes[i];
    if (!node->links) {
      continue;
    }
    for (uint32_t lv = 0U; lv <= node->level; lv++) {
      uint32_t *lp = node_layer_ptr(node, idx->cfg.m, lv);
      uint32_t kept = 0U;
      for (uint32_t j = 1U; j <= lp[0]; j++) {
        const sve_hnsw_node_t *target = &idx->nodes[lp[j]];
        if (target->level < lv || (!legacy && !target->alive)) {
          return false;
        }
        if (target->alive) {
          lp[1U + kept++] = lp[j];
        }
      }
      lp[0] = kept;
    }
  }
  return true;
}

static bool same_index_config(const sve_hnsw_config_t *a,
                              const sve_hnsw_config_t *b) {
  return a->dims == b->dims && a->elem_type == b->elem_type &&
         a->metric == b->metric && a->index_kind == b->index_kind &&
         a->m == b->m && a->ef_construction == b->ef_construction &&
         a->ef_search == b->ef_search;
}

int sve_hnsw_deserialize_for_config(const uint8_t *buf, size_t len,
                                    const sve_hnsw_config_t *expected,
                                    sve_hnsw_t **out_idx) {
  if (!out_idx) {
    return SVE_ERR_INVALID;
  }
  *out_idx = NULL;
  if (!buf || len < SVE_SNAPSHOT_HEADER_BYTES + SVE_SNAPSHOT_CRC_BYTES ||
      len > (size_t)PTRDIFF_MAX) {
    return SVE_ERR_CORRUPT;
  }

  const uint8_t *p = buf;
  const uint8_t *end = buf + len - SVE_SNAPSHOT_CRC_BYTES;
  uint32_t magic = 0U, ver = 0U, dims = 0U, elem_t = 0U, metric = 0U, kind = 0U;
  uint32_t m = 0U, ef_c = 0U, ef_s = 0U;
  uint64_t rng_state = 0ULL;
  uint32_t node_count = 0U, live_count = 0U, ep = 0U, max_lv = 0U;

  if (!get_u32(&p, end, &magic) || !get_u32(&p, end, &ver) ||
      magic != SVE_HNSW_MAGIC ||
      (ver != SVE_HNSW_VERSION_LEGACY && ver != SVE_HNSW_VERSION)) {
    return SVE_ERR_CORRUPT;
  }
  const uint8_t *tail = end;
  uint32_t stored_crc = 0U;
  if (!get_u32(&tail, buf + len, &stored_crc) ||
      stored_crc != sve_crc32c(buf, len - SVE_SNAPSHOT_CRC_BYTES)) {
    return SVE_ERR_CORRUPT;
  }
  if (!get_u32(&p, end, &dims) || !get_u32(&p, end, &elem_t) ||
      !get_u32(&p, end, &metric) || !get_u32(&p, end, &kind) ||
      !get_u32(&p, end, &m) || !get_u32(&p, end, &ef_c) ||
      !get_u32(&p, end, &ef_s) || !get_u64(&p, end, &rng_state) ||
      !get_u32(&p, end, &node_count) || !get_u32(&p, end, &live_count) ||
      !get_u32(&p, end, &ep) || !get_u32(&p, end, &max_lv) ||
      node_count > SVE_MAX_NODE_COUNT || max_lv > SVE_HNSW_MAX_LEVEL ||
      live_count > node_count ||
      (live_count == 0U ? (ep != UINT32_MAX || max_lv != 0U) : ep >= node_count)) {
    return SVE_ERR_CORRUPT;
  }

  sve_hnsw_config_t cfg = {(size_t)dims, (sve_elem_type_t)elem_t,
                          (sve_metric_t)metric, (sve_index_kind_t)kind,
                          m, ef_c, ef_s, rng_state};
  if (!sve_hnsw_config_is_valid(&cfg) ||
      (expected && !same_index_config(&cfg, expected)) ||
      (cfg.index_kind == SVE_INDEX_FLAT && max_lv != 0U)) {
    return SVE_ERR_CORRUPT;
  }

  bool legacy = ver == SVE_HNSW_VERSION_LEGACY;
  bool use_sq8 = cfg.index_kind == SVE_INDEX_HNSW_SQ8;
  size_t stride = use_sq8 ? cfg.dims : sve_elem_byte_size(cfg.elem_type, cfg.dims);
  size_t remaining = (size_t)(end - p);
  uint32_t node_cap = node_count > 64U ? node_count : 64U;
  if (node_count > remaining / SVE_SNAPSHOT_NODE_BYTES ||
      !array_fits(node_cap, sizeof(sve_hnsw_node_t))) {
    return SVE_ERR_CORRUPT;
  }
  remaining -= (size_t)node_count * SVE_SNAPSHOT_NODE_BYTES;
  size_t vectors = legacy ? (size_t)node_count : (size_t)live_count;
  if (vectors > remaining / stride) {
    return SVE_ERR_CORRUPT;
  }
  remaining -= vectors * stride;
  if (cfg.index_kind != SVE_INDEX_FLAT && live_count > remaining / sizeof(uint32_t)) {
    return SVE_ERR_CORRUPT;
  }

  size_t mcap = 128U;
  while (mcap / 2U < live_count) {
    if (mcap > SIZE_MAX / (2U * sizeof(sve_map_slot_t))) {
      return SVE_ERR_CORRUPT;
    }
    mcap *= 2U;
  }

  sve_hnsw_t *idx = (sve_hnsw_t *)calloc(1U, sizeof(*idx));
  if (!idx) {
    return SVE_ERR_NOMEM;
  }
  idx->cfg = cfg;
  idx->use_sq8 = use_sq8;
  idx->vec_stride = stride;
  idx->rng_state = rng_state;
  idx->node_cap = node_cap;
  idx->nodes = (sve_hnsw_node_t *)calloc(node_cap, sizeof(*idx->nodes));
  if (!idx->nodes || map_init(idx, mcap) != SVE_OK) {
    sve_hnsw_destroy(idx);
    return SVE_ERR_NOMEM;
  }
  idx->entry_point = ep;
  idx->max_level = max_lv;

  int status = SVE_ERR_CORRUPT;
  uint32_t observed_max = 0U;
  for (uint32_t i = 0U; i < node_count; i++) {
    sve_hnsw_node_t *nd = &idx->nodes[i];
    uint64_t rowid = 0ULL;
    uint8_t alive = 0U;
    idx->node_count++;
    if (!get_u64(&p, end, &rowid) || !get_u32(&p, end, &nd->level) ||
        !get_u8(&p, end, &alive) || !get_f32(&p, end, &nd->sq_scale) ||
        nd->level > SVE_HNSW_MAX_LEVEL || alive > 1U ||
        (cfg.index_kind == SVE_INDEX_FLAT && nd->level != 0U) ||
        !isfinite(nd->sq_scale) || nd->sq_scale < 0.0f ||
        (!use_sq8 && nd->sq_scale != 0.0f)) {
      goto fail;
    }
    nd->rowid = rowid <= INT64_MAX ? (int64_t)rowid
                                  : -1 - (int64_t)(UINT64_MAX - rowid);
    nd->alive = alive != 0U;
    if (!nd->alive) {
      if (legacy) {
        if ((size_t)(end - p) < stride) {
          goto fail;
        }
        p += stride;
      }
      continue;
    }
    if (idx->live_count >= live_count || (size_t)(end - p) < stride) {
      goto fail;
    }
    idx->live_count++;
    if (nd->level > observed_max) {
      observed_max = nd->level;
    }
    int map_rc = map_put(idx, nd->rowid, i);
    if (map_rc != SVE_OK) {
      status = map_rc == SVE_ERR_NOMEM ? SVE_ERR_NOMEM : SVE_ERR_CORRUPT;
      goto fail;
    }

    nd->vec = (uint8_t *)malloc(stride);
    if (!nd->vec) {
      status = SVE_ERR_NOMEM;
      goto fail;
    }
    if (!legacy && !use_sq8 && cfg.elem_type == SVE_ELEM_FLOAT32) {
      float *vec = (float *)nd->vec;
      for (size_t j = 0U; j < cfg.dims; j++) {
        if (!get_f32(&p, end, &vec[j])) {
          goto fail;
        }
      }
    } else {
      memcpy(nd->vec, p, stride);
      p += stride;
    }
    if (!use_sq8 && cfg.elem_type == SVE_ELEM_FLOAT32 &&
        !sve_f32_array_is_finite((const float *)nd->vec, cfg.dims)) {
      goto fail;
    }
    if (use_sq8 && memchr(nd->vec, (unsigned char)INT8_MIN, stride) != NULL) {
      goto fail;
    }
    if (cfg.metric == SVE_METRIC_COSINE) {
      nd->norm = cosine_norm(idx, nd->vec);
    }

    if (cfg.index_kind != SVE_INDEX_FLAT) {
      size_t nu32 = total_link_u32_count(cfg.m, nd->level);
      nd->links = (uint32_t *)calloc(nu32, sizeof(*nd->links));
      if (!nd->links) {
        status = SVE_ERR_NOMEM;
        goto fail;
      }
      for (uint32_t lv = 0U; lv <= nd->level; lv++) {
        uint32_t *lp = node_layer_ptr(nd, cfg.m, lv);
        if (!get_u32(&p, end, &lp[0]) || lp[0] > layer_max_deg(cfg.m, lv) ||
            lp[0] > (size_t)(end - p) / sizeof(uint32_t)) {
          goto fail;
        }
        for (uint32_t j = 1U; j <= lp[0]; j++) {
          if (!get_u32(&p, end, &lp[j]) || lp[j] >= node_count || lp[j] == i) {
            goto fail;
          }
          for (uint32_t k = 1U; k < j; k++) {
            if (lp[k] == lp[j]) {
              goto fail;
            }
          }
        }
      }
    }
  }

  if (p != end || idx->live_count != live_count || observed_max != max_lv ||
      (ep != UINT32_MAX && (!idx->nodes[ep].alive || idx->nodes[ep].level != max_lv)) ||
      !snapshot_links_are_valid(idx, legacy)) {
    goto fail;
  }
  *out_idx = idx;
  return SVE_OK;

fail:
  sve_hnsw_destroy(idx);
  return status;
}

int sve_hnsw_deserialize(const uint8_t *buf, size_t len, sve_hnsw_t **out_idx) {
  return sve_hnsw_deserialize_for_config(buf, len, NULL, out_idx);
}
