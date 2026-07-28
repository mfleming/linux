// SPDX-License-Identifier: GPL-2.0-only
/*
 * Stack depot - a stack trace storage that avoids duplication.
 *
 * Internally, stack depot has two storage backends. Refcounted entries and
 * callers that request STACK_DEPOT_FLAG_COUNTABLE use the legacy hash table with
 * contiguous stack records in stack pools. Persistent non-refcounted entries
 * can use trie storage when enabled; trie nodes share common frame prefixes and
 * are published through RCU/COW child arrays.
 *
 * Author: Alexander Potapenko <glider@google.com>
 * Copyright (C) 2016 Google, Inc.
 *
 * Based on the code by Dmitry Chernenkov.
 */

#define pr_fmt(fmt) "stackdepot: " fmt

#include <linux/bitmap.h>
#include <linux/build_bug.h>
#include <linux/debugfs.h>
#include <linux/errno.h>
#include <linux/gfp.h>
#include <linux/jhash.h>
#include <linux/jump_label.h>
#include <linux/kernel.h>
#include <linux/log2.h>
#include <linux/kmsan.h>
#include <linux/list.h>
#include <linux/mm.h>
#include <linux/moduleparam.h>
#include <linux/mutex.h>
#include <linux/poison.h>
#include <linux/printk.h>
#include <linux/rculist.h>
#include <linux/rcupdate.h>
#include <linux/refcount.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/stacktrace.h>
#include <linux/stackdepot.h>
#include <linux/string.h>
#include <linux/types.h>
#include <linux/memblock.h>
#include <linux/kasan-enabled.h>

#include <asm/stackdepot.h>

/*
 * The pool_index is offset by 1 so the first record does not have a 0 handle.
 */
/* Parsed before mm_core_init(); trie handle decoding assumes this is then fixed. */
static unsigned int stack_max_pools __read_mostly =
	MIN((1LL << DEPOT_POOL_INDEX_BITS) - 1, 8192);

static bool stack_depot_disabled;
static bool __stack_depot_early_init_requested __initdata = IS_ENABLED(CONFIG_STACKDEPOT_ALWAYS_INIT);
static bool __stack_depot_early_init_passed __initdata;

/* Use one hash table bucket per 16 KB of memory. */
#define STACK_HASH_TABLE_SCALE 14
/* Limit the number of buckets between 4K and 1M. */
#define STACK_BUCKET_NUMBER_ORDER_MIN 12
#define STACK_BUCKET_NUMBER_ORDER_MAX 20
/* Initial seed for jhash2. */
#define STACK_HASH_SEED 0x9747b28c

/* Hash table of stored stack records. */
static struct list_head *stack_table;
/* Fixed order of the number of table buckets. Used when KASAN is enabled. */
static unsigned int stack_bucket_number_order;
/* Hash mask for indexing the table. */
static unsigned int stack_hash_mask;

/* Array of memory regions used by both stack depot backends. */
static void **stack_pools;
/* Newly allocated pool that is not yet added to stack_pools. */
static void *new_pool;
/* Number of pools in stack_pools. */
static int pools_num;
/* Offset to unused hash storage in the current pool. */
static size_t pool_offset = DEPOT_POOL_SIZE;
/* Freelist of stack records within stack_pools. */
static LIST_HEAD(free_stacks);
/* The lock must be held when performing pool or freelist modifications. */
static DEFINE_RAW_SPINLOCK(pool_lock);

/* Hash-backend statistics counters for debugfs. */
enum depot_counter_id {
	DEPOT_COUNTER_REFD_ALLOCS,
	DEPOT_COUNTER_REFD_FREES,
	DEPOT_COUNTER_REFD_INUSE,
	DEPOT_COUNTER_FREELIST_SIZE,
	DEPOT_COUNTER_PERSIST_COUNT,
	DEPOT_COUNTER_PERSIST_BYTES,
	DEPOT_COUNTER_COUNT,
};
static long counters[DEPOT_COUNTER_COUNT];
static const char *const counter_names[] = {
	[DEPOT_COUNTER_REFD_ALLOCS]	= "refcounted_allocations",
	[DEPOT_COUNTER_REFD_FREES]	= "refcounted_frees",
	[DEPOT_COUNTER_REFD_INUSE]	= "refcounted_in_use",
	[DEPOT_COUNTER_FREELIST_SIZE]	= "freelist_size",
	[DEPOT_COUNTER_PERSIST_COUNT]	= "hash_persistent_count",
	[DEPOT_COUNTER_PERSIST_BYTES]	= "hash_persistent_bytes",
};
static_assert(ARRAY_SIZE(counter_names) == DEPOT_COUNTER_COUNT);

enum stack_depot_frame_mode {
	STACK_DEPOT_FRAME_RAW,
	STACK_DEPOT_FRAME_COMPRESSED,
};

/*
 * A trie node stores one run of frames that all use the same payload format.
 * Architectures may compress some frames to 32-bit payloads; mixed raw and
 * compressed input is split across multiple trie nodes so each node has one
 * decoding mode.
 */
struct stack_depot_frame_run {
	u16 nr_entries;
	u8 mode;
};

static_assert(CONFIG_STACKDEPOT_MAX_FRAMES <= U16_MAX);

struct stack_depot_trie_child_array;

struct stack_depot_trie_node {
	/* Parent links let fetch rebuild a full stack from a leaf to the root. */
	const struct stack_depot_trie_node __rcu *parent;
	/* Child arrays are separate RCU/COW generations. */
	const struct stack_depot_trie_child_array __rcu *children;
	u32 leaf_id;
	struct stack_depot_frame_run run;
	unsigned char data[];
};

/*
 * Children are sorted by first frame and searched by insertion slot.
 * Writers may append to spare capacity at the sorted tail, but never change
 * existing child pointers. Other updates build and publish a replacement array.
 */
struct stack_depot_trie_child_array {
	unsigned int nr_children;
	unsigned int capacity;
	const struct stack_depot_trie_node __rcu *children[];
};

/* A retired child array carries an optional node through its RCU grace period. */
struct stack_depot_trie_retired_array {
	struct list_head list;
	unsigned long rcu_state;
	struct stack_depot_trie_node *pending_node;
	unsigned char data[];
};

static_assert(IS_ALIGNED(offsetof(struct stack_depot_trie_retired_array, data),
			 1UL << DEPOT_STACK_ALIGN));

#define STACK_DEPOT_TRIE_MAX_NODES (CONFIG_STACKDEPOT_MAX_FRAMES + 1)
#define STACK_DEPOT_TRIE_MAX_CHILD_ARRAYS CONFIG_STACKDEPOT_MAX_FRAMES
#define STACK_DEPOT_TRIE_SLOT_SIZE BIT(DEPOT_STACK_ALIGN)
#define STACK_DEPOT_TRIE_POOL_SLOTS \
	(DEPOT_POOL_SIZE / STACK_DEPOT_TRIE_SLOT_SIZE)

struct stack_depot_trie_pool {
	struct list_head list;
	unsigned int free_slots;
	DECLARE_BITMAP(used, STACK_DEPOT_TRIE_POOL_SLOTS);
};

#define STACK_DEPOT_TRIE_POOL_FIRST_SLOT \
	DIV_ROUND_UP(sizeof(struct stack_depot_trie_pool), \
		     STACK_DEPOT_TRIE_SLOT_SIZE)

static_assert(STACK_DEPOT_TRIE_POOL_FIRST_SLOT < STACK_DEPOT_TRIE_POOL_SLOTS);

/* Writer-owned storage for one unpublished trie insertion. */
struct stack_depot_trie_insert_alloc {
	struct stack_depot_trie_node *nodes[STACK_DEPOT_TRIE_MAX_NODES];
	size_t node_sizes[STACK_DEPOT_TRIE_MAX_NODES];
	struct stack_depot_trie_child_array *chain_arrays[STACK_DEPOT_TRIE_MAX_CHILD_ARRAYS];
	struct stack_depot_trie_child_array *prefix_children;
	struct stack_depot_trie_child_array *slot_array;
};

static DEFINE_STATIC_KEY_FALSE(stack_depot_trie_enabled);
static const struct stack_depot_trie_child_array __rcu *stack_depot_trie_root;
static struct stack_depot_trie_insert_alloc *stack_depot_trie_alloc;
static DEFINE_RAW_SPINLOCK(stack_depot_trie_writer_lock);
static bool stack_depot_trie_requested;

module_param_named(trie_enabled, stack_depot_trie_requested, bool, 0);
MODULE_PARM_DESC(trie_enabled, "Enable stack depot trie storage at boot");

#define DEPOT_POOL_INDEX_MASK ((1U << DEPOT_POOL_INDEX_BITS) - 1)
#define DEPOT_OFFSET_MASK ((1U << DEPOT_OFFSET_BITS) - 1)

/* Retired fixed-size slots remain reserved until their RCU grace period ends. */
static LIST_HEAD(stack_depot_trie_pools);
static LIST_HEAD(pending_trie_arrays);

/*
 * stack_max_pools is the split point between hash and trie handle encodings.
 * A handle with pool_index_plus_1 in 1..stack_max_pools names a hash-backed
 * stack pool. Larger pool-index values cannot refer to hash pools, so trie
 * storage uses that handle space to encode a dense leaf_id. The side table
 * maps each leaf_id to its trie leaf node.
 */
static inline u32 __stack_depot_trie_max_leaf_id(void)
{
	return (DEPOT_POOL_INDEX_MASK - stack_max_pools) <<
		DEPOT_OFFSET_BITS;
}

static depot_stack_handle_t __stack_depot_trie_handle(u32 leaf_id)
{
	union handle_parts parts = {};
	u64 pool_index_plus_1;
	u32 pool_delta;
	u32 index;

	index = leaf_id - 1;
	pool_delta = index >> DEPOT_OFFSET_BITS;
	pool_index_plus_1 = (u64)stack_max_pools + 1 + pool_delta;

	parts.pool_index_plus_1 = pool_index_plus_1;
	parts.offset = index & DEPOT_OFFSET_MASK;
	return parts.handle;
}

static inline bool stack_depot_handle_is_trie(depot_stack_handle_t handle)
{
	union handle_parts parts = { .handle = handle };

	return parts.pool_index_plus_1 > stack_max_pools;
}

static u32 __stack_depot_trie_leaf_id(depot_stack_handle_t handle)
{
	union handle_parts parts = { .handle = handle };
	u32 pool_delta;

	pool_delta = parts.pool_index_plus_1 - stack_max_pools - 1;
	return (pool_delta << DEPOT_OFFSET_BITS) + parts.offset + 1;
}

/*
 * Trie handles encode a dense leaf ID. The side table maps that ID to a leaf
 * pointer for lockless fetch/print paths, which can run from diagnostic
 * contexts where taking a lock would be unsafe. Initialization
 * installs the root; early initialization also installs the first directory and
 * chunk. Additional directories and chunks are preallocated and published
 * lazily as leaf IDs grow. RCU pointer publication makes fully initialized
 * directories, chunks, and leaves visible to those lockless readers.
 */
#define STACK_DEPOT_TRIE_SIDE_TABLE_CHUNK_SIZE \
	(PAGE_SIZE / sizeof(struct stack_depot_trie_node *))
#define STACK_DEPOT_TRIE_SIDE_TABLE_DIR_SIZE \
	(PAGE_SIZE / sizeof(struct stack_depot_trie_node **))

struct stack_depot_trie_side_dir {
	/* Both the chunk pointer and each leaf pointer in it are RCU-published. */
	const struct stack_depot_trie_node __rcu * __rcu *
		chunks[STACK_DEPOT_TRIE_SIDE_TABLE_DIR_SIZE];
};

struct stack_depot_trie_side_root {
	unsigned int dir_capacity;
	struct stack_depot_trie_side_dir __rcu *dirs[];
};

struct stack_depot_trie_side_prealloc {
	/* Preallocated side-table directory page for sparse growth. */
	struct stack_depot_trie_side_dir *dir;
	/* Preallocated side-table leaf chunk for sparse growth. */
	const struct stack_depot_trie_node __rcu **chunk;
};

static struct stack_depot_trie_side_root *trie_side_table_root;
static DEFINE_RAW_SPINLOCK(trie_side_table_cache_lock);
/* Zeroed unpublished pages; get/put transfer ownership under the cache lock. */
static struct stack_depot_trie_side_prealloc trie_side_table_cache;
static u32 trie_side_table_last_leaf_id;

/* Lock order: writer_lock -> pool_lock. The cache lock is never nested. */

static inline size_t stack_depot_frame_run_entry_bytes(enum stack_depot_frame_mode mode)
{
	if (mode == STACK_DEPOT_FRAME_COMPRESSED)
		return sizeof(u32);
	return sizeof(unsigned long);
}

static inline size_t stack_depot_frame_run_bytes(const struct stack_depot_frame_run *run)
{
	return run->nr_entries * stack_depot_frame_run_entry_bytes(run->mode);
}

static inline size_t trie_node_bytes(const struct stack_depot_frame_run *run)
{
	return ALIGN(offsetof(struct stack_depot_trie_node, data) +
		     stack_depot_frame_run_bytes(run), sizeof(unsigned long));
}

static size_t trie_child_array_bytes(unsigned int capacity)
{
	size_t size;

	size = struct_size_t(struct stack_depot_trie_child_array, children,
			     capacity);
	return ALIGN(size, sizeof(unsigned long));
}

static void trie_child_array_init(struct stack_depot_trie_child_array *array,
				  unsigned int capacity,
				  const struct stack_depot_trie_node * const *nodes,
				  unsigned int nr_children)
{
	unsigned int i;

	array->nr_children = nr_children;
	array->capacity = capacity;
	for (i = 0; i < nr_children; i++)
		RCU_INIT_POINTER(array->children[i], nodes[i]);
	for (i = nr_children; i < capacity; i++)
		RCU_INIT_POINTER(array->children[i], NULL);
}

static inline unsigned int trie_side_table_root_index(u32 id)
{
	return ((id - 1) / STACK_DEPOT_TRIE_SIDE_TABLE_CHUNK_SIZE) /
		STACK_DEPOT_TRIE_SIDE_TABLE_DIR_SIZE;
}

static inline unsigned int trie_side_table_dir_index(u32 id)
{
	return ((id - 1) / STACK_DEPOT_TRIE_SIDE_TABLE_CHUNK_SIZE) %
		STACK_DEPOT_TRIE_SIDE_TABLE_DIR_SIZE;
}

static inline unsigned int trie_side_table_slot_index(u32 id)
{
	return (id - 1) % STACK_DEPOT_TRIE_SIDE_TABLE_CHUNK_SIZE;
}

static struct stack_depot_trie_side_dir *trie_side_table_load_dir(unsigned int root)
{
	struct stack_depot_trie_side_root *root_vec;

	root_vec = trie_side_table_root;
	if (!root_vec || root >= root_vec->dir_capacity)
		return NULL;
	/* Pairs with side-table directory rcu_assign_pointer(). */
	return rcu_dereference_check(root_vec->dirs[root],
				     lockdep_is_held(&stack_depot_trie_writer_lock) ||
				     rcu_read_lock_sched_held());
}

static inline const struct stack_depot_trie_node __rcu **
trie_side_table_dir_load_chunk(struct stack_depot_trie_side_dir *dir,
			       unsigned int idx)
{
	/* Pairs with the chunk rcu_assign_pointer() in leaf ID preparation. */
	return rcu_dereference_check(dir->chunks[idx],
				     lockdep_is_held(&stack_depot_trie_writer_lock) ||
				     rcu_read_lock_sched_held());
}

static u32
trie_side_table_prepare_leaf_slot(struct stack_depot_trie_side_prealloc *prealloc)
{
	const struct stack_depot_trie_node __rcu **chunk;
	struct stack_depot_trie_side_dir *dir;
	struct stack_depot_trie_side_root *root_vec;
	unsigned int root;
	unsigned int idx;
	u32 id;

	lockdep_assert_held(&stack_depot_trie_writer_lock);

	id = trie_side_table_last_leaf_id + 1;
	if (id > __stack_depot_trie_max_leaf_id())
		return 0;

	root_vec = trie_side_table_root;
	root = trie_side_table_root_index(id);
	dir = trie_side_table_load_dir(root);
	if (!dir) {
		if (WARN_ON_ONCE(!prealloc->dir))
			return 0;
		dir = prealloc->dir;
		prealloc->dir = NULL;
		/* Publish the zeroed directory before readers can load it locklessly. */
		rcu_assign_pointer(root_vec->dirs[root], dir);
	}

	idx = trie_side_table_dir_index(id);
	chunk = trie_side_table_dir_load_chunk(dir, idx);
	if (!chunk) {
		if (WARN_ON_ONCE(!prealloc->chunk))
			return 0;
		chunk = prealloc->chunk;
		prealloc->chunk = NULL;
		rcu_assign_pointer(dir->chunks[idx], chunk);
	}

	return id;
}

static void trie_side_table_root_init(struct stack_depot_trie_side_root *root_vec,
				      unsigned int root_size)
{
	root_vec->dir_capacity = root_size;
	trie_side_table_last_leaf_id = 0;
}

static inline unsigned int trie_side_table_root_size_for_max_id(u32 max_leaf_id)
{
	unsigned int top_size;

	top_size = DIV_ROUND_UP(max_leaf_id, STACK_DEPOT_TRIE_SIDE_TABLE_CHUNK_SIZE);
	return DIV_ROUND_UP(top_size, STACK_DEPOT_TRIE_SIDE_TABLE_DIR_SIZE);
}

static int __init __stack_depot_trie_side_table_init_memblock(void)
{
	struct stack_depot_trie_side_root *root_vec;
	struct stack_depot_trie_side_dir *first_dir;
	const struct stack_depot_trie_node __rcu **first_chunk;
	size_t root_bytes;
	u32 max_leaf_id;
	unsigned int root_size;

	max_leaf_id = __stack_depot_trie_max_leaf_id();
	if (!max_leaf_id)
		return -EINVAL;
	root_size = trie_side_table_root_size_for_max_id(max_leaf_id);
	root_bytes = struct_size_t(struct stack_depot_trie_side_root, dirs, root_size);

	root_vec = memblock_alloc(root_bytes, __alignof__(*root_vec));
	if (!root_vec)
		return -ENOMEM;
	memset(root_vec, 0, root_bytes);
	first_dir = memblock_alloc(PAGE_SIZE, PAGE_SIZE);
	if (!first_dir) {
		memblock_free(root_vec, root_bytes);
		return -ENOMEM;
	}
	memset(first_dir, 0, PAGE_SIZE);
	first_chunk = memblock_alloc(PAGE_SIZE, PAGE_SIZE);
	if (!first_chunk) {
		memblock_free(first_dir, PAGE_SIZE);
		memblock_free(root_vec, root_bytes);
		return -ENOMEM;
	}
	memset(first_chunk, 0, PAGE_SIZE);

	trie_side_table_root_init(root_vec, root_size);
	RCU_INIT_POINTER(root_vec->dirs[0], first_dir);
	RCU_INIT_POINTER(first_dir->chunks[0], first_chunk);
	trie_side_table_root = root_vec;
	return 0;
}

static int __stack_depot_trie_side_table_init(gfp_t gfp_flags)
{
	struct stack_depot_trie_side_root *root_vec;
	unsigned int root_size;
	size_t root_bytes;
	u32 max_leaf_id;

	max_leaf_id = __stack_depot_trie_max_leaf_id();
	if (!max_leaf_id)
		return -EINVAL;

	root_size = trie_side_table_root_size_for_max_id(max_leaf_id);
	root_bytes = struct_size_t(struct stack_depot_trie_side_root, dirs, root_size);
	root_vec = kvzalloc(root_bytes, gfp_flags);
	if (!root_vec)
		return -ENOMEM;

	trie_side_table_root_init(root_vec, root_size);
	trie_side_table_root = root_vec;
	return 0;
}

static int __init stack_depot_trie_init_memblock(void)
{
	struct stack_depot_trie_insert_alloc *alloc;
	size_t size;
	int ret;

	size = sizeof(*stack_depot_trie_alloc);
	alloc = memblock_alloc(size, __alignof__(*alloc));
	if (!alloc)
		return -ENOMEM;

	ret = __stack_depot_trie_side_table_init_memblock();
	if (ret) {
		memblock_free(alloc, size);
		return ret;
	}
	stack_depot_trie_alloc = alloc;

	static_branch_enable(&stack_depot_trie_enabled);
	return 0;
}

static int stack_depot_trie_init(gfp_t gfp_flags)
{
	struct stack_depot_trie_insert_alloc *alloc;
	int ret;

	alloc = kvzalloc(sizeof(*stack_depot_trie_alloc), gfp_flags);
	if (!alloc)
		return -ENOMEM;

	ret = __stack_depot_trie_side_table_init(gfp_flags);
	if (ret) {
		kvfree(alloc);
		return ret;
	}
	stack_depot_trie_alloc = alloc;

	static_branch_enable(&stack_depot_trie_enabled);
	return 0;
}

static int trie_side_table_get_prealloc(gfp_t gfp_flags,
					struct stack_depot_trie_side_prealloc *prealloc)
{
	unsigned long flags;

	gfp_flags = gfp_nested_mask(gfp_flags);
	raw_spin_lock_irqsave(&trie_side_table_cache_lock, flags);
	prealloc->dir = trie_side_table_cache.dir;
	prealloc->chunk = trie_side_table_cache.chunk;
	trie_side_table_cache.dir = NULL;
	trie_side_table_cache.chunk = NULL;
	raw_spin_unlock_irqrestore(&trie_side_table_cache_lock, flags);

	if (!prealloc->dir) {
		prealloc->dir = (void *)get_zeroed_page(gfp_flags);
		if (!prealloc->dir)
			return -ENOMEM;
	}
	if (!prealloc->chunk) {
		prealloc->chunk = (void *)get_zeroed_page(gfp_flags);
		if (!prealloc->chunk)
			return -ENOMEM;
	}

	return 0;
}

static void trie_side_table_put_prealloc(struct stack_depot_trie_side_prealloc *prealloc)
{
	unsigned long flags;

	raw_spin_lock_irqsave(&trie_side_table_cache_lock, flags);
	if (!trie_side_table_cache.dir) {
		trie_side_table_cache.dir = prealloc->dir;
		prealloc->dir = NULL;
	}
	if (!trie_side_table_cache.chunk) {
		trie_side_table_cache.chunk = prealloc->chunk;
		prealloc->chunk = NULL;
	}
	raw_spin_unlock_irqrestore(&trie_side_table_cache_lock, flags);

	if (prealloc->dir)
		free_page((unsigned long)prealloc->dir);
	if (prealloc->chunk)
		free_page((unsigned long)prealloc->chunk);
}

static const struct stack_depot_trie_node __rcu **trie_side_table_leaf_slot(u32 id)
{
	const struct stack_depot_trie_node __rcu **chunk;
	struct stack_depot_trie_side_dir *dir;
	unsigned int root;

	root = trie_side_table_root_index(id);
	dir = trie_side_table_load_dir(root);
	chunk = trie_side_table_dir_load_chunk(dir, trie_side_table_dir_index(id));

	return &chunk[trie_side_table_slot_index(id)];
}

static const struct stack_depot_trie_node *__stack_depot_trie_side_table_lookup(u32 id)
{
	const struct stack_depot_trie_node __rcu **chunk;
	struct stack_depot_trie_side_dir *dir;
	unsigned int root;

	root = trie_side_table_root_index(id);
	dir = trie_side_table_load_dir(root);
	if (!dir)
		return NULL;
	chunk = trie_side_table_dir_load_chunk(dir, trie_side_table_dir_index(id));
	if (!chunk)
		return NULL;

	/* Pairs with side-table leaf rcu_assign_pointer(). */
	return rcu_dereference_check(chunk[trie_side_table_slot_index(id)],
				     lockdep_is_held(&stack_depot_trie_writer_lock) ||
				     rcu_read_lock_sched_held());
}

static inline size_t trie_array_alloc_size(size_t size)
{
	return offsetof(struct stack_depot_trie_retired_array, data) + size;
}

static inline struct stack_depot_trie_retired_array *
trie_retired_array(const void *ptr)
{
	return container_of(ptr, struct stack_depot_trie_retired_array, data);
}

static bool depot_init_pool(void **prealloc);

static unsigned int trie_pool_reserve_slots(struct stack_depot_trie_pool *pool,
					    unsigned int nr_slots)
{
	unsigned int run_start = STACK_DEPOT_TRIE_POOL_FIRST_SLOT;
	unsigned int run = 0;
	unsigned int i;
	unsigned int slot;

	if (pool->free_slots < nr_slots)
		return STACK_DEPOT_TRIE_POOL_SLOTS;

	for (slot = STACK_DEPOT_TRIE_POOL_FIRST_SLOT;
	     slot < STACK_DEPOT_TRIE_POOL_SLOTS; slot++) {
		if (pool->used[slot / BITS_PER_LONG] &
		    BIT(slot % BITS_PER_LONG)) {
			run = 0;
			continue;
		}
		if (!run)
			run_start = slot;
		if (++run != nr_slots)
			continue;

		for (i = run_start; i < run_start + nr_slots; i++)
			pool->used[i / BITS_PER_LONG] |= BIT(i % BITS_PER_LONG);
		pool->free_slots -= nr_slots;
		return run_start;
	}

	return STACK_DEPOT_TRIE_POOL_SLOTS;
}

static void *trie_pool_alloc(size_t size, void **prealloc)
{
	struct stack_depot_trie_pool *pool;
	unsigned int nr_slots;
	unsigned int slot;

	lockdep_assert_held(&pool_lock);

	nr_slots = DIV_ROUND_UP(size, STACK_DEPOT_TRIE_SLOT_SIZE);
	list_for_each_entry_reverse(pool, &stack_depot_trie_pools, list) {
		slot = trie_pool_reserve_slots(pool, nr_slots);
		if (slot != STACK_DEPOT_TRIE_POOL_SLOTS)
			return (char *)pool + slot * STACK_DEPOT_TRIE_SLOT_SIZE;
	}

	if (!depot_init_pool(prealloc))
		return NULL;
	pool = stack_pools[pools_num - 1];
	/* Keep hash records out of this bitmap-owned pool. */
	pool_offset = DEPOT_POOL_SIZE;
	memset(pool, 0, sizeof(*pool));
	pool->free_slots = STACK_DEPOT_TRIE_POOL_SLOTS -
			   STACK_DEPOT_TRIE_POOL_FIRST_SLOT;
	list_add_tail(&pool->list, &stack_depot_trie_pools);

	slot = trie_pool_reserve_slots(pool, nr_slots);
	return (char *)pool + slot * STACK_DEPOT_TRIE_SLOT_SIZE;
}

static void trie_pool_release(const void *ptr, size_t size)
{
	struct stack_depot_trie_pool *pool;
	unsigned long pfn;
	unsigned int nr_slots;
	unsigned int slot;
	unsigned int i;

	lockdep_assert_held(&pool_lock);

	pfn = page_to_pfn(virt_to_page(ptr));
	pfn &= ~(BIT(DEPOT_POOL_ORDER) - 1);
	pool = page_address(pfn_to_page(pfn));
	slot = ((unsigned long)ptr - (unsigned long)pool) >> DEPOT_STACK_ALIGN;
	nr_slots = DIV_ROUND_UP(size, STACK_DEPOT_TRIE_SLOT_SIZE);
	for (i = slot; i < slot + nr_slots; i++)
		pool->used[i / BITS_PER_LONG] &= ~BIT(i % BITS_PER_LONG);
	pool->free_slots += nr_slots;
}

static struct stack_depot_trie_child_array *
trie_pool_alloc_array(size_t size, void **prealloc)
{
	struct stack_depot_trie_retired_array *retired;

	retired = trie_pool_alloc(trie_array_alloc_size(size), prealloc);
	return retired ? (void *)retired->data : NULL;
}

static void trie_pool_release_array(const void *ptr, size_t size)
{
	trie_pool_release(trie_retired_array(ptr), trie_array_alloc_size(size));
}

static void trie_drain_pending_arrays(void)
{
	struct stack_depot_trie_retired_array *retired;
	struct stack_depot_trie_retired_array *tmp;
	struct stack_depot_trie_child_array *array;

	lockdep_assert_held(&pool_lock);

	list_for_each_entry_safe(retired, tmp, &pending_trie_arrays, list) {
		/* Pending arrays are FIFO; later entries cannot be ready yet. */
		if (!poll_state_synchronize_rcu(retired->rcu_state))
			break;
		array = (void *)retired->data;
		list_del(&retired->list);
		if (retired->pending_node)
			trie_pool_release(retired->pending_node,
					  trie_node_bytes(&retired->pending_node->run));
		trie_pool_release_array(array,
					trie_child_array_bytes(array->capacity));
	}
}

static void trie_retire_child_array_locked(const void *ptr)
{
	struct stack_depot_trie_retired_array *retired;

	lockdep_assert_held(&pool_lock);

	retired = trie_retired_array(ptr);
	retired->pending_node = NULL;
	retired->rcu_state = get_state_synchronize_rcu();
	list_add_tail(&retired->list, &pending_trie_arrays);
}

static void
trie_retire_child_array_with_node(const void *ptr,
				  const struct stack_depot_trie_node *node)
{
	struct stack_depot_trie_retired_array *retired;
	unsigned long flags;

	raw_spin_lock_irqsave(&pool_lock, flags);
	trie_retire_child_array_locked(ptr);
	retired = trie_retired_array(ptr);
	retired->pending_node = (struct stack_depot_trie_node *)node;
	raw_spin_unlock_irqrestore(&pool_lock, flags);
}

/*
 * Preallocate resources that cannot be allocated while trie writers hold raw
 * spinlocks. Side-table growth is mandatory before a new leaf ID can be
 * used, so side-table preallocation failure disables insertion for this
 * save. Pool preallocation is opportunistic: reusable trie slots may still
 * satisfy the insertion, and trie_pool_alloc_insert() reports -ENOSPC if they
 * do not.
 */
static int
__stack_depot_trie_alloc_prealloc(gfp_t alloc_flags, void **pool_prealloc,
				  struct stack_depot_trie_side_prealloc *side_prealloc)
{
	unsigned long flags;
	bool need_pool;
	int ret;

	raw_spin_lock_irqsave(&pool_lock, flags);
	need_pool = !new_pool;
	raw_spin_unlock_irqrestore(&pool_lock, flags);
	if (need_pool) {
		struct page *page;

		page = alloc_pages(gfp_nested_mask(alloc_flags), DEPOT_POOL_ORDER);
		if (page)
			*pool_prealloc = page_address(page);
	}
	ret = trie_side_table_get_prealloc(alloc_flags, side_prealloc);

	if (ret) {
		if (*pool_prealloc) {
			free_pages((unsigned long)*pool_prealloc, DEPOT_POOL_ORDER);
			*pool_prealloc = NULL;
		}
		return -ENOSPC;
	}
	return 0;
}

/*
 * Reserve fixed-size pool slots for one trie insertion. If any reservation
 * fails, all slots reserved by this attempt are released locally.
 * The caller must not publish any returned storage before side-table and trie
 * publication succeeds.
 */
static int trie_pool_alloc_insert(struct stack_depot_trie_insert_alloc *alloc,
				  void **pool_prealloc,
				  unsigned int nr_nodes,
				  unsigned int nr_chain_arrays,
				  size_t prefix_children_size,
				  size_t slot_array_size)
{
	unsigned long flags;
	size_t one_child_size;
	unsigned int i;
	int ret = -ENOSPC;

	memset(alloc->nodes, 0, nr_nodes * sizeof(*alloc->nodes));
	memset(alloc->chain_arrays, 0,
	       nr_chain_arrays * sizeof(*alloc->chain_arrays));
	alloc->prefix_children = NULL;
	alloc->slot_array = NULL;

	raw_spin_lock_irqsave(&pool_lock, flags);
	printk_deferred_enter();
	trie_drain_pending_arrays();
	one_child_size = trie_child_array_bytes(1);
	for (i = 0; i < nr_nodes; i++) {
		alloc->nodes[i] = trie_pool_alloc(alloc->node_sizes[i],
						  pool_prealloc);
		if (!alloc->nodes[i])
			goto out_discard;
	}
	for (i = 0; i < nr_chain_arrays; i++) {
		alloc->chain_arrays[i] =
			trie_pool_alloc_array(one_child_size, pool_prealloc);
		if (!alloc->chain_arrays[i])
			goto out_discard;
	}
	if (prefix_children_size) {
		alloc->prefix_children =
			trie_pool_alloc_array(prefix_children_size, pool_prealloc);
		if (!alloc->prefix_children)
			goto out_discard;
	}
	if (slot_array_size) {
		alloc->slot_array =
			trie_pool_alloc_array(slot_array_size, pool_prealloc);
		if (!alloc->slot_array)
			goto out_discard;
	}
	ret = 0;
	goto out;
out_discard:
	for (i = 0; i < nr_nodes; i++) {
		struct stack_depot_trie_node *node = alloc->nodes[i];

		if (node)
			trie_pool_release(node, alloc->node_sizes[i]);
	}
	for (i = 0; i < nr_chain_arrays; i++) {
		struct stack_depot_trie_child_array *array = alloc->chain_arrays[i];

		if (!array)
			continue;
		trie_pool_release_array(array, one_child_size);
	}
	if (alloc->prefix_children)
		trie_pool_release_array(alloc->prefix_children,
					prefix_children_size);
	if (alloc->slot_array)
		trie_pool_release_array(alloc->slot_array, slot_array_size);
out:
	printk_deferred_exit();
	raw_spin_unlock_irqrestore(&pool_lock, flags);
	return ret;
}

static const struct stack_depot_trie_node *
stack_depot_trie_lookup(const unsigned long *entries, unsigned int nr_entries);

static depot_stack_handle_t
trie_find_handle(const unsigned long *entries, unsigned int nr_entries)
{
	depot_stack_handle_t handle = 0;
	const struct stack_depot_trie_node *leaf;

	rcu_read_lock_sched_notrace();
	leaf = stack_depot_trie_lookup(entries, nr_entries);
	if (leaf)
		handle = __stack_depot_trie_handle(leaf->leaf_id);
	rcu_read_unlock_sched_notrace();

	return handle;
}

static void trie_side_table_publish_new_leaf(u32 leaf_id,
					     const struct stack_depot_trie_node *leaf)
{
	const struct stack_depot_trie_node __rcu **slot;

	lockdep_assert_held(&stack_depot_trie_writer_lock);

	slot = trie_side_table_leaf_slot(leaf_id);
	/* Pairs with __stack_depot_trie_side_table_lookup(). */
	rcu_assign_pointer(*slot, leaf);
}

static void trie_side_table_publish_split_leaves(u32 old_leaf_id,
						 const struct stack_depot_trie_node *old_leaf,
						 u32 new_leaf_id,
						 const struct stack_depot_trie_node *new_leaf)
{
	const struct stack_depot_trie_node __rcu **new_slot;
	const struct stack_depot_trie_node __rcu **old_slot = NULL;

	lockdep_assert_held(&stack_depot_trie_writer_lock);

	if (old_leaf_id)
		old_slot = trie_side_table_leaf_slot(old_leaf_id);

	new_slot = trie_side_table_leaf_slot(new_leaf_id);
	if (old_slot) {
		/* Pairs with __stack_depot_trie_side_table_lookup(). */
		rcu_assign_pointer(*old_slot, old_leaf);
	}

	/* Pairs with __stack_depot_trie_side_table_lookup(). */
	rcu_assign_pointer(*new_slot, new_leaf);
}

static int __init disable_stack_depot(char *str)
{
	return kstrtobool(str, &stack_depot_disabled);
}
early_param("stack_depot_disable", disable_stack_depot);

static int __init parse_max_pools(char *str)
{
	const long long limit = (1LL << (DEPOT_POOL_INDEX_BITS)) - 1;
	unsigned int max_pools;
	int rv;

	rv = kstrtouint(str, 0, &max_pools);
	if (rv)
		return rv;

	if (max_pools < 1024) {
		pr_err("stack_depot_max_pools below 1024, using default of %u\n",
		       stack_max_pools);
		goto out;
	}

	if (max_pools > limit) {
		pr_err("stack_depot_max_pools exceeds %lld, using default of %u\n",
		       limit, stack_max_pools);
		goto out;
	}

	stack_max_pools = max_pools;
out:
	return 0;
}
early_param("stack_depot_max_pools", parse_max_pools);

void __init stack_depot_request_early_init(void)
{
	/* Too late to request early init now. */
	WARN_ON(__stack_depot_early_init_passed);

	__stack_depot_early_init_requested = true;
}

/* Initialize list_head's within the hash table. */
static void init_stack_table(unsigned long entries)
{
	unsigned long i;

	for (i = 0; i < entries; i++)
		INIT_LIST_HEAD(&stack_table[i]);
}

/* Initializes hash and optional trie storage during early boot. */
int __init stack_depot_early_init(void)
{
	unsigned long entries = 0;

	/* This function must be called only once, from mm_init(). */
	if (WARN_ON(__stack_depot_early_init_passed))
		return 0;
	__stack_depot_early_init_passed = true;

	/*
	 * Print disabled message even if early init has not been requested:
	 * stack_depot_init() will not print one.
	 */
	if (stack_depot_disabled) {
		pr_info("disabled\n");
		return 0;
	}

	/*
	 * If KASAN is enabled, use the maximum order: KASAN is frequently used
	 * in fuzzing scenarios, which leads to a large number of different
	 * stack traces being stored in stack depot.
	 */
	if (kasan_enabled() && !stack_bucket_number_order)
		stack_bucket_number_order = STACK_BUCKET_NUMBER_ORDER_MAX;

	/*
	 * Check if early init has been requested after setting
	 * stack_bucket_number_order: stack_depot_init() uses its value.
	 */
	if (!__stack_depot_early_init_requested)
		return 0;

	/*
	 * If stack_bucket_number_order is not set, leave entries as 0 to rely
	 * on the automatic calculations performed by alloc_large_system_hash().
	 */
	if (stack_bucket_number_order)
		entries = 1UL << stack_bucket_number_order;
	pr_info("allocating hash table via alloc_large_system_hash\n");
	stack_table = alloc_large_system_hash("stackdepot",
						sizeof(struct list_head),
						entries,
						STACK_HASH_TABLE_SCALE,
						HASH_EARLY,
						NULL,
						&stack_hash_mask,
						1UL << STACK_BUCKET_NUMBER_ORDER_MIN,
						1UL << STACK_BUCKET_NUMBER_ORDER_MAX);
	if (!stack_table) {
		pr_err("hash table allocation failed, disabling\n");
		stack_depot_disabled = true;
		return -ENOMEM;
	}
	if (!entries) {
		/*
		 * Obtain the number of entries that was calculated by
		 * alloc_large_system_hash().
		 */
		entries = stack_hash_mask + 1;
	}
	init_stack_table(entries);

	pr_info("allocating space for %u stack pools via memblock\n",
		stack_max_pools);
	stack_pools =
		memblock_alloc(stack_max_pools * sizeof(void *), PAGE_SIZE);
	if (!stack_pools) {
		pr_err("stack pools allocation failed, disabling\n");
		memblock_free(stack_table, entries * sizeof(struct list_head));
		stack_depot_disabled = true;
		return -ENOMEM;
	}
	if (stack_depot_trie_requested && stack_depot_trie_init_memblock()) {
		pr_warn("trie storage initialization failed, disabling trie storage\n");
		stack_depot_trie_requested = false;
	}

	return 0;
}

/* Initializes hash and optional trie storage after boot. */
int stack_depot_init(void)
{
	static DEFINE_MUTEX(stack_depot_init_mutex);
	unsigned long entries;
	int ret = 0;

	mutex_lock(&stack_depot_init_mutex);

	if (stack_depot_disabled || stack_table)
		goto out_unlock;

	/*
	 * Similarly to stack_depot_early_init, use stack_bucket_number_order
	 * if assigned, and rely on automatic scaling otherwise.
	 */
	if (stack_bucket_number_order) {
		entries = 1UL << stack_bucket_number_order;
	} else {
		int scale = STACK_HASH_TABLE_SCALE;

		entries = nr_free_buffer_pages();
		entries = roundup_pow_of_two(entries);

		if (scale > PAGE_SHIFT)
			entries >>= (scale - PAGE_SHIFT);
		else
			entries <<= (PAGE_SHIFT - scale);
	}

	if (entries < 1UL << STACK_BUCKET_NUMBER_ORDER_MIN)
		entries = 1UL << STACK_BUCKET_NUMBER_ORDER_MIN;
	if (entries > 1UL << STACK_BUCKET_NUMBER_ORDER_MAX)
		entries = 1UL << STACK_BUCKET_NUMBER_ORDER_MAX;

	pr_info("allocating hash table of %lu entries via kvcalloc\n", entries);
	stack_table = kvcalloc(entries, sizeof(struct list_head), GFP_KERNEL);
	if (!stack_table) {
		pr_err("hash table allocation failed, disabling\n");
		stack_depot_disabled = true;
		ret = -ENOMEM;
		goto out_unlock;
	}
	stack_hash_mask = entries - 1;
	init_stack_table(entries);

	pr_info("allocating space for %u stack pools via kvcalloc\n",
		stack_max_pools);
	stack_pools = kvcalloc(stack_max_pools, sizeof(void *), GFP_KERNEL);
	if (!stack_pools) {
		pr_err("stack pools allocation failed, disabling\n");
		kvfree(stack_table);
		stack_depot_disabled = true;
		ret = -ENOMEM;
		goto out_unlock;
	}
	if (stack_depot_trie_requested) {
		ret = stack_depot_trie_init(GFP_KERNEL);
		if (ret) {
			pr_warn("trie storage initialization failed, disabling trie storage\n");
			stack_depot_trie_requested = false;
			ret = 0;
		}
	}

out_unlock:
	mutex_unlock(&stack_depot_init_mutex);

	return ret;
}
EXPORT_SYMBOL_GPL(stack_depot_init);

/*
 * Initializes new stack pool, and updates the list of pools.
 */
static bool depot_init_pool(void **prealloc)
{
	lockdep_assert_held(&pool_lock);

	if (unlikely(pools_num >= stack_max_pools)) {
		/* Bail out if we reached the pool limit. */
		WARN_ON_ONCE(pools_num > stack_max_pools); /* should never happen */
		WARN_ON_ONCE(!new_pool); /* to avoid unnecessary pre-allocation */
		WARN_ONCE(1, "Stack depot reached limit capacity");
		return false;
	}

	if (!new_pool && *prealloc) {
		/* We have preallocated memory, use it. */
		WRITE_ONCE(new_pool, *prealloc);
		*prealloc = NULL;
	}

	if (!new_pool)
		return false; /* new_pool and *prealloc are NULL */

	/* Save reference to the pool to be used by depot_fetch_stack(). */
	stack_pools[pools_num] = new_pool;

	/*
	 * Stack depot tries to keep an extra pool allocated even before it runs
	 * out of space in the currently used pool.
	 *
	 * To indicate that a new preallocation is needed new_pool is reset to
	 * NULL; do not reset to NULL if we have reached the maximum number of
	 * pools.
	 */
	if (pools_num < stack_max_pools)
		WRITE_ONCE(new_pool, NULL);
	else
		WRITE_ONCE(new_pool, STACK_DEPOT_POISON);

	/* Pairs with concurrent READ_ONCE() in depot_fetch_stack(). */
	WRITE_ONCE(pools_num, pools_num + 1);
	ASSERT_EXCLUSIVE_WRITER(pools_num);

	pool_offset = 0;

	return true;
}

/* Keeps the preallocated memory to be used for a new stack depot pool. */
static void depot_keep_new_pool(void **prealloc)
{
	lockdep_assert_held(&pool_lock);

	/*
	 * If a new pool is already saved or the maximum number of
	 * pools is reached, do not use the preallocated memory.
	 */
	if (new_pool)
		return;

	WRITE_ONCE(new_pool, *prealloc);
	*prealloc = NULL;
}

/*
 * Try to initialize a new stack record from the current pool, a cached pool, or
 * the current pre-allocation.
 */
static struct stack_record *depot_pop_free_pool(void **prealloc, size_t size)
{
	struct stack_record *stack;
	void *current_pool;
	u32 pool_index;

	lockdep_assert_held(&pool_lock);

	if (pool_offset + size > DEPOT_POOL_SIZE) {
		if (!depot_init_pool(prealloc))
			return NULL;
	}

	if (WARN_ON_ONCE(pools_num < 1))
		return NULL;
	pool_index = pools_num - 1;
	current_pool = stack_pools[pool_index];
	if (WARN_ON_ONCE(!current_pool))
		return NULL;

	stack = current_pool + pool_offset;

	/* Pre-initialize handle once. */
	stack->handle.pool_index_plus_1 = pool_index + 1;
	stack->handle.offset = pool_offset >> DEPOT_STACK_ALIGN;
	stack->handle.extra = 0;
	INIT_LIST_HEAD(&stack->hash_list);

	pool_offset += size;

	return stack;
}

/* Try to find next free usable entry from the freelist. */
static struct stack_record *depot_pop_free(void)
{
	struct stack_record *stack;

	lockdep_assert_held(&pool_lock);

	if (list_empty(&free_stacks))
		return NULL;

	/*
	 * We maintain the invariant that the elements in front are least
	 * recently used, and are therefore more likely to be associated with an
	 * RCU grace period in the past. Consequently it is sufficient to only
	 * check the first entry.
	 */
	stack = list_first_entry(&free_stacks, struct stack_record, free_list);
	if (!poll_state_synchronize_rcu(stack->rcu_state))
		return NULL;

	list_del(&stack->free_list);
	counters[DEPOT_COUNTER_FREELIST_SIZE]--;

	return stack;
}

static inline size_t depot_stack_record_size(struct stack_record *s, unsigned int nr_entries)
{
	const size_t used = flex_array_size(s, entries, nr_entries);
	const size_t unused = sizeof(s->entries) - used;

	WARN_ON_ONCE(sizeof(s->entries) < used);

	return ALIGN(sizeof(struct stack_record) - unused, 1 << DEPOT_STACK_ALIGN);
}

/* Allocates a new stack in a stack depot pool. */
static struct stack_record *
depot_alloc_stack(unsigned long *entries, unsigned int nr_entries, u32 hash, depot_flags_t flags, void **prealloc)
{
	struct stack_record *stack = NULL;
	size_t record_size;

	lockdep_assert_held(&pool_lock);

	/* This should already be checked by public API entry points. */
	if (WARN_ON_ONCE(!nr_entries))
		return NULL;

	/* Limit number of saved frames to CONFIG_STACKDEPOT_MAX_FRAMES. */
	if (nr_entries > CONFIG_STACKDEPOT_MAX_FRAMES)
		nr_entries = CONFIG_STACKDEPOT_MAX_FRAMES;

	if (flags & STACK_DEPOT_FLAG_GET) {
		/*
		 * Evictable entries have to allocate the max. size so they may
		 * safely be re-used by differently sized allocations.
		 */
		record_size = depot_stack_record_size(stack, CONFIG_STACKDEPOT_MAX_FRAMES);
		stack = depot_pop_free();
	} else {
		record_size = depot_stack_record_size(stack, nr_entries);
	}

	if (!stack) {
		stack = depot_pop_free_pool(prealloc, record_size);
		if (!stack)
			return NULL;
	}

	/* Save the stack trace. */
	stack->hash = hash;
	stack->size = nr_entries;
	stack->flags = flags & STACK_DEPOT_FLAG_COUNTABLE;
	/* stack->handle is already filled in by depot_pop_free_pool(). */
	memcpy(stack->entries, entries, flex_array_size(stack, entries, nr_entries));

	if (flags & STACK_DEPOT_FLAG_GET) {
		refcount_set(&stack->count, 1);
		counters[DEPOT_COUNTER_REFD_ALLOCS]++;
		counters[DEPOT_COUNTER_REFD_INUSE]++;
	} else {
		/* Warn on attempts to switch to refcounting this entry. */
		refcount_set(&stack->count, REFCOUNT_SATURATED);
		counters[DEPOT_COUNTER_PERSIST_COUNT]++;
		counters[DEPOT_COUNTER_PERSIST_BYTES] += record_size;
	}

	/*
	 * Let KMSAN know the stored stack record is initialized. This shall
	 * prevent false positive reports if instrumented code accesses it.
	 */
	kmsan_unpoison_memory(stack, record_size);

	return stack;
}

static struct stack_record *depot_fetch_stack(depot_stack_handle_t handle)
{
	const int pools_num_cached = READ_ONCE(pools_num);
	union handle_parts parts = { .handle = handle };
	void *pool;
	u32 pool_index = parts.pool_index_plus_1 - 1;
	size_t offset = parts.offset << DEPOT_STACK_ALIGN;
	struct stack_record *stack;

	lockdep_assert_not_held(&pool_lock);

	if (pool_index >= pools_num_cached) {
		WARN(1, "pool index %d out of bounds (%d) for stack id %08x\n",
		     pool_index, pools_num_cached, handle);
		return NULL;
	}

	pool = stack_pools[pool_index];
	if (WARN_ON(!pool))
		return NULL;

	stack = pool + offset;
	if (WARN_ON(!refcount_read(&stack->count)))
		return NULL;

	return stack;
}

/* Links stack into the freelist. */
static void depot_free_stack(struct stack_record *stack)
{
	unsigned long flags;

	lockdep_assert_not_held(&pool_lock);

	raw_spin_lock_irqsave(&pool_lock, flags);
	printk_deferred_enter();

	/*
	 * Remove the entry from the hash list. Concurrent list traversal may
	 * still observe the entry, but since the refcount is zero, this entry
	 * will no longer be considered as valid.
	 */
	list_del_rcu(&stack->hash_list);

	/*
	 * Due to being used from constrained contexts such as the allocators,
	 * NMI, or even RCU itself, stack depot cannot rely on primitives that
	 * would sleep (such as synchronize_rcu()) or recursively call into
	 * stack depot again (such as call_rcu()).
	 *
	 * Instead, get an RCU cookie, so that we can ensure this entry isn't
	 * moved onto another list until the next grace period, and concurrent
	 * RCU list traversal remains safe.
	 */
	stack->rcu_state = get_state_synchronize_rcu();

	/*
	 * Add the entry to the freelist tail, so that older entries are
	 * considered first - their RCU cookie is more likely to no longer be
	 * associated with the current grace period.
	 */
	list_add_tail(&stack->free_list, &free_stacks);

	counters[DEPOT_COUNTER_FREELIST_SIZE]++;
	counters[DEPOT_COUNTER_REFD_FREES]++;
	counters[DEPOT_COUNTER_REFD_INUSE]--;

	printk_deferred_exit();
	raw_spin_unlock_irqrestore(&pool_lock, flags);
}

/* Calculates the hash for a stack. */
static inline u32 hash_stack(unsigned long *entries, unsigned int size)
{
	return jhash2((u32 *)entries,
		      array_size(size,  sizeof(*entries)) / sizeof(u32),
		      STACK_HASH_SEED);
}

/*
 * Non-instrumented version of memcmp().
 * Does not check the lexicographical order, only the equality.
 */
static inline
int stackdepot_memcmp(const unsigned long *u1, const unsigned long *u2,
			unsigned int n)
{
	for ( ; n-- ; u1++, u2++) {
		if (*u1 != *u2)
			return 1;
	}
	return 0;
}

/* Finds a stack in a bucket of the hash table. */
static inline struct stack_record *find_stack(struct list_head *bucket,
					      unsigned long *entries, int size,
					      u32 hash, depot_flags_t flags)
{
	struct stack_record *stack, *ret = NULL;

	/*
	 * Stack depot may be used from instrumentation that instruments RCU or
	 * tracing itself; use variant that does not call into RCU and cannot be
	 * traced.
	 *
	 * Note: Such use cases must take care when using refcounting to evict
	 * unused entries, because the stack record free-then-reuse code paths
	 * do call into RCU.
	 */
	rcu_read_lock_sched_notrace();

	list_for_each_entry_rcu(stack, bucket, hash_list) {
		if (stack->hash != hash || stack->size != size)
			continue;
		/* Page owner countable records have a distinct count lifetime. */
		if ((stack->flags ^ flags) & STACK_DEPOT_FLAG_COUNTABLE)
			continue;

		/*
		 * This may race with depot_free_stack() accessing the freelist
		 * management state unioned with @entries. The refcount is zero
		 * in that case and the below refcount_inc_not_zero() will fail.
		 */
		if (data_race(stackdepot_memcmp(entries, stack->entries, size)))
			continue;

		/*
		 * Try to increment refcount. If this succeeds, the stack record
		 * is valid and has not yet been freed.
		 *
		 * If STACK_DEPOT_FLAG_GET is not used, it is undefined behavior
		 * to then call stack_depot_put() later, and we can assume that
		 * a stack record is never placed back on the freelist.
		 */
		if ((flags & STACK_DEPOT_FLAG_GET) && !refcount_inc_not_zero(&stack->count))
			continue;

		ret = stack;
		break;
	}

	rcu_read_unlock_sched_notrace();

	return ret;
}

static u32
stack_depot_trie_insert_locked(const unsigned long *entries,
			       unsigned int nr_entries,
			       void **pool_prealloc,
			       struct stack_depot_trie_side_prealloc *side_prealloc);

static depot_stack_handle_t
stack_depot_trie_save(unsigned long *entries, unsigned int nr_entries,
		      gfp_t alloc_flags)
{
	struct stack_depot_trie_side_prealloc side_prealloc = {};
	void *pool_prealloc = NULL;
	depot_stack_handle_t handle;
	unsigned long flags;
	bool retried = false;
	u32 leaf_id;
	int ret;

retry:
	handle = trie_find_handle(entries, nr_entries);
	if (handle)
		return handle;

	ret = __stack_depot_trie_alloc_prealloc(alloc_flags, &pool_prealloc,
						&side_prealloc);
	if (ret)
		goto out_free;

	raw_spin_lock_irqsave(&stack_depot_trie_writer_lock, flags);
	leaf_id = stack_depot_trie_insert_locked(entries, nr_entries,
						 &pool_prealloc, &side_prealloc);
	if (leaf_id)
		handle = __stack_depot_trie_handle(leaf_id);
	raw_spin_unlock_irqrestore(&stack_depot_trie_writer_lock, flags);
	if (!handle && !retried) {
		retried = true;
		if (pool_prealloc) {
			raw_spin_lock_irqsave(&pool_lock, flags);
			depot_keep_new_pool(&pool_prealloc);
			raw_spin_unlock_irqrestore(&pool_lock, flags);
		}
		if (pool_prealloc) {
			free_pages((unsigned long)pool_prealloc, DEPOT_POOL_ORDER);
			pool_prealloc = NULL;
		}
		trie_side_table_put_prealloc(&side_prealloc);
		goto retry;
	}

out_free:
	if (pool_prealloc) {
		raw_spin_lock_irqsave(&pool_lock, flags);
		depot_keep_new_pool(&pool_prealloc);
		raw_spin_unlock_irqrestore(&pool_lock, flags);
	}
	if (pool_prealloc)
		free_pages((unsigned long)pool_prealloc, DEPOT_POOL_ORDER);
	trie_side_table_put_prealloc(&side_prealloc);
	return handle;
}

depot_stack_handle_t stack_depot_save_flags(unsigned long *entries,
					    unsigned int nr_entries,
					    gfp_t alloc_flags,
					    depot_flags_t depot_flags)
{
	struct list_head *bucket;
	struct stack_record *found = NULL;
	depot_stack_handle_t handle = 0;
	struct page *page = NULL;
	void *prealloc = NULL;
	bool allow_spin = gfpflags_allow_spinning(alloc_flags);
	bool can_alloc = (depot_flags & STACK_DEPOT_FLAG_CAN_ALLOC) && allow_spin;
	unsigned long flags;
	u32 hash;

	if (WARN_ON(depot_flags & ~STACK_DEPOT_FLAGS_MASK))
		return 0;
	if (WARN_ON_ONCE((depot_flags & STACK_DEPOT_FLAG_GET) &&
			 (depot_flags & STACK_DEPOT_FLAG_COUNTABLE)))
		return 0;

	/*
	 * If this stack trace is from an interrupt, including anything before
	 * interrupt entry usually leads to unbounded stack depot growth.
	 *
	 * Since use of filter_irq_stacks() is a requirement to ensure stack
	 * depot can efficiently deduplicate interrupt stacks, always
	 * filter_irq_stacks() to simplify all callers' use of stack depot.
	 */
	nr_entries = filter_irq_stacks(entries, nr_entries);

	if (unlikely(nr_entries == 0) || stack_depot_disabled)
		return 0;

	if (!(depot_flags & (STACK_DEPOT_FLAG_GET | STACK_DEPOT_FLAG_COUNTABLE)) &&
	    static_branch_unlikely(&stack_depot_trie_enabled)) {
		if (nr_entries > CONFIG_STACKDEPOT_MAX_FRAMES)
			nr_entries = CONFIG_STACKDEPOT_MAX_FRAMES;
		if (in_nmi() || !can_alloc) {
			WARN_ON_ONCE(can_alloc);
			return trie_find_handle(entries, nr_entries);
		}
		return stack_depot_trie_save(entries, nr_entries, alloc_flags);
	}

	hash = hash_stack(entries, nr_entries);
	bucket = &stack_table[hash & stack_hash_mask];

	/* Fast path: look the stack trace up without locking. */
	found = find_stack(bucket, entries, nr_entries, hash, depot_flags);
	if (found)
		goto exit;

	/*
	 * Allocate memory for a new pool if required now:
	 * we won't be able to do that under the lock.
	 */
	if (unlikely(can_alloc && !READ_ONCE(new_pool))) {
		page = alloc_pages(gfp_nested_mask(alloc_flags),
				   DEPOT_POOL_ORDER);
		if (page)
			prealloc = page_address(page);
	}

	if (in_nmi() || !allow_spin) {
		/* We can never allocate in NMI context. */
		WARN_ON_ONCE(can_alloc);
		/* Best effort; bail if we fail to take the lock. */
		if (!raw_spin_trylock_irqsave(&pool_lock, flags))
			goto exit;
	} else {
		raw_spin_lock_irqsave(&pool_lock, flags);
	}
	printk_deferred_enter();

	/* Try to find again, to avoid concurrently inserting duplicates. */
	found = find_stack(bucket, entries, nr_entries, hash, depot_flags);
	if (!found) {
		struct stack_record *new =
			depot_alloc_stack(entries, nr_entries, hash, depot_flags, &prealloc);

		if (new) {
			/*
			 * This releases the stack record into the bucket and
			 * makes it visible to readers in find_stack().
			 */
			list_add_rcu(&new->hash_list, bucket);
			found = new;
		}
	}

	if (prealloc) {
		/*
		 * Either stack depot already contains this stack trace, or
		 * depot_alloc_stack() did not consume the preallocated memory.
		 * Try to keep the preallocated memory for future.
		 */
		depot_keep_new_pool(&prealloc);
	}

	printk_deferred_exit();
	raw_spin_unlock_irqrestore(&pool_lock, flags);
exit:
	if (prealloc) {
		/* Stack depot didn't use this memory, free it. */
		if (!allow_spin)
			free_pages_nolock(virt_to_page(prealloc), DEPOT_POOL_ORDER);
		else
			free_pages((unsigned long)prealloc, DEPOT_POOL_ORDER);
	}
	if (found)
		handle = found->handle.handle;
	return handle;
}
EXPORT_SYMBOL_GPL(stack_depot_save_flags);

depot_stack_handle_t stack_depot_save(unsigned long *entries,
				      unsigned int nr_entries,
				      gfp_t alloc_flags)
{
	return stack_depot_save_flags(entries, nr_entries, alloc_flags,
				      STACK_DEPOT_FLAG_CAN_ALLOC);
}
EXPORT_SYMBOL_GPL(stack_depot_save);

struct stack_record *__stack_depot_get_stack_record(depot_stack_handle_t handle)
{
	struct stack_record *stack;

	if (!handle)
		return NULL;
	if (WARN_ON_ONCE(stack_depot_handle_is_trie(handle)))
		return NULL;

	stack = depot_fetch_stack(handle);
	if (!stack)
		return NULL;
	if (WARN_ON_ONCE(!(stack->flags & STACK_DEPOT_FLAG_COUNTABLE)))
		return NULL;

	return stack;
}

static void frame_run_init(const unsigned long *entries,
			   unsigned int nr_entries,
			   struct stack_depot_frame_run *run)
{
	u32 payload;
	unsigned int i;
	bool compressed;

	compressed = arch_stack_depot_frame_try_compress(entries[0], &payload);
	for (i = 1; i < nr_entries; i++) {
		bool next;

		next = arch_stack_depot_frame_try_compress(entries[i], &payload);
		if (next != compressed)
			break;
	}

	/* @i is the first non-matching frame, or @nr_entries if all matched. */
	run->mode = compressed ? STACK_DEPOT_FRAME_COMPRESSED : STACK_DEPOT_FRAME_RAW;
	run->nr_entries = i;
}

static void
stack_depot_trie_node_frame(const struct stack_depot_trie_node *node,
			    unsigned int index, unsigned long *frame)
{
	u32 payload;

	if (node->run.mode == STACK_DEPOT_FRAME_RAW) {
		memcpy(frame, node->data + index * sizeof(*frame),
		       sizeof(*frame));
		return;
	}

	memcpy(&payload, node->data + index * sizeof(payload), sizeof(payload));
	arch_stack_depot_frame_decompress(payload, frame);
}

static void trie_node_init(struct stack_depot_trie_node *node,
			   const struct stack_depot_trie_node *parent, u32 leaf_id,
			   const unsigned long *entries,
			   const struct stack_depot_frame_run *run)
{
	if (run->mode == STACK_DEPOT_FRAME_COMPRESSED) {
		unsigned int i;

		for (i = 0; i < run->nr_entries; i++) {
			u32 payload;

			arch_stack_depot_frame_try_compress(entries[i], &payload);
			memcpy(node->data + i * sizeof(payload), &payload,
			       sizeof(payload));
		}
	} else {
		memcpy(node->data, entries, stack_depot_frame_run_bytes(run));
	}

	RCU_INIT_POINTER(node->parent, parent);
	RCU_INIT_POINTER(node->children, NULL);
	node->leaf_id = leaf_id;
	node->run = *run;
}

static void trie_node_init_slice(struct stack_depot_trie_node *node,
				 const struct stack_depot_trie_node *parent, u32 leaf_id,
				 const struct stack_depot_trie_node *src_node,
				 unsigned int start, unsigned int nr_entries)
{
	struct stack_depot_frame_run run;
	size_t entry_bytes;

	run = src_node->run;
	run.nr_entries = nr_entries;

	entry_bytes = stack_depot_frame_run_entry_bytes(src_node->run.mode);
	memcpy(node->data, src_node->data + start * entry_bytes,
	       stack_depot_frame_run_bytes(&run));
	RCU_INIT_POINTER(node->parent, parent);
	RCU_INIT_POINTER(node->children, NULL);
	node->leaf_id = leaf_id;
	node->run = run;
}

static unsigned int
__stack_depot_trie_node_match(const struct stack_depot_trie_node *node,
			      const unsigned long *entries,
			      unsigned int nr_entries)
{
	unsigned int limit;
	unsigned int i;

	limit = min(node->run.nr_entries, nr_entries);
	if (node->run.mode == STACK_DEPOT_FRAME_RAW) {
		for (i = 0; i < limit; i++) {
			unsigned long frame;

			memcpy(&frame, node->data + i * sizeof(frame), sizeof(frame));
			if (frame != entries[i])
				break;
		}

		return i;
	}

	for (i = 0; i < limit; i++) {
		unsigned long frame;

		stack_depot_trie_node_frame(node, i, &frame);
		if (frame != entries[i])
			break;
	}

	return i;
}

static inline const struct stack_depot_trie_node *
trie_load_parent(const struct stack_depot_trie_node *node)
{
	return rcu_dereference_check(node->parent,
				     lockdep_is_held(&stack_depot_trie_writer_lock) ||
				     rcu_read_lock_sched_held());
}

static inline const struct stack_depot_trie_child_array *
trie_load_children_slot(const struct stack_depot_trie_child_array __rcu * const *slot)
{
	return rcu_dereference_check(*slot,
				     lockdep_is_held(&stack_depot_trie_writer_lock) ||
				     rcu_read_lock_sched_held());
}

static inline const struct stack_depot_trie_node *
trie_child_array_load_child(const struct stack_depot_trie_child_array *array,
			    unsigned int pos)
{
	return rcu_dereference_check(array->children[pos],
				     lockdep_is_held(&stack_depot_trie_writer_lock) ||
				     rcu_read_lock_sched_held());
}

/*
 * Find the child slot for @frame.
 *
 * Return true and set @pos to the matching child index when @array contains a
 * matching child. Return false and set @pos to the insertion index otherwise.
 * A NULL @array is treated as empty and returns @pos = 0.
 */
static bool
trie_child_array_find_slot(const struct stack_depot_trie_child_array *array,
			   unsigned long frame, unsigned int *pos)
{
	unsigned int left = 0;
	unsigned int right;

	if (!array) {
		*pos = 0;
		return false;
	}

	right = READ_ONCE(array->nr_children);
	while (left < right) {
		unsigned int mid = left + (right - left) / 2;
		const struct stack_depot_trie_node *node;
		unsigned long mid_frame;

		node = trie_child_array_load_child(array, mid);
		if (!node) {
			/* Tail append may produce a transient lockless lookup miss. */
			right = mid;
			continue;
		}
		stack_depot_trie_node_frame(node, 0, &mid_frame);
		if (mid_frame < frame) {
			left = mid + 1;
		} else if (mid_frame > frame) {
			right = mid;
		} else {
			*pos = mid;
			return true;
		}
	}

	*pos = left;
	return false;
}

static void
trie_child_array_insert_at(const struct stack_depot_trie_child_array *old,
			   unsigned int pos,
			   const struct stack_depot_trie_node *node,
			   struct stack_depot_trie_child_array *new_array,
			   unsigned int new_capacity)
{
	unsigned int nr_old;
	unsigned int i;

	nr_old = old->nr_children;

	new_array->nr_children = nr_old + 1;
	new_array->capacity = new_capacity;
	for (i = 0; i < pos; i++)
		RCU_INIT_POINTER(new_array->children[i],
				 trie_child_array_load_child(old, i));
	RCU_INIT_POINTER(new_array->children[pos], node);
	for (i = pos; i < nr_old; i++)
		RCU_INIT_POINTER(new_array->children[i + 1],
				 trie_child_array_load_child(old, i));
	for (i = nr_old + 1; i < new_array->capacity; i++)
		RCU_INIT_POINTER(new_array->children[i], NULL);
}

static void
trie_child_array_replace_at(const struct stack_depot_trie_child_array *old_array,
			    const struct stack_depot_trie_node *new_child,
			    struct stack_depot_trie_child_array *new_array,
			    unsigned int pos)
{
	unsigned int i;

	new_array->nr_children = old_array->nr_children;
	new_array->capacity = old_array->capacity;
	for (i = 0; i < old_array->nr_children; i++)
		RCU_INIT_POINTER(new_array->children[i],
				 trie_child_array_load_child(old_array, i));
	RCU_INIT_POINTER(new_array->children[pos], new_child);
	for (i = old_array->nr_children; i < new_array->capacity; i++)
		RCU_INIT_POINTER(new_array->children[i], NULL);
}

static void trie_reparent_children(struct stack_depot_trie_node *parent)
{
	const struct stack_depot_trie_child_array *children;
	unsigned int i;

	children = trie_load_children_slot(&parent->children);
	if (!children)
		return;
	/*
	 * COW updates reuse unchanged descendant subtrees. Repoint their parent
	 * links before retiring the old parent so fetch never follows a freed node.
	 * Lockless fetches may see the new parent before structural publication, but
	 * the old and new parent chains contain the same frames and remain RCU-live.
	 */
	for (i = 0; i < children->nr_children; i++) {
		struct stack_depot_trie_node *child;

		child = (struct stack_depot_trie_node *)trie_child_array_load_child(children, i);
		rcu_assign_pointer(child->parent, parent);
	}
}

static void
trie_build_append_chain(const struct stack_depot_trie_node *parent, u32 leaf_id,
			const unsigned long *entries, unsigned int nr_entries,
			struct stack_depot_trie_node * const *nodes,
			struct stack_depot_trie_child_array * const *chain_arrays,
			const struct stack_depot_trie_node **head,
			const struct stack_depot_trie_node **tail)
{
	const struct stack_depot_trie_node *prev = parent;
	unsigned int pos = 0;
	unsigned int used = 0;
	unsigned int i;

	while (pos < nr_entries) {
		struct stack_depot_frame_run run;
		struct stack_depot_trie_node *node;
		u32 id;

		node = nodes[used];
		frame_run_init(&entries[pos], nr_entries - pos, &run);

		id = pos + run.nr_entries == nr_entries ? leaf_id : 0;
		trie_node_init(node, prev, id, &entries[pos], &run);

		prev = node;
		pos += run.nr_entries;
		used++;
	}

	for (i = 0; i + 1 < used; i++) {
		const struct stack_depot_trie_node *next = nodes[i + 1];
		struct stack_depot_trie_node *node = nodes[i];
		struct stack_depot_trie_child_array *array = chain_arrays[i];

		trie_child_array_init(array, 1, &next, 1);
		RCU_INIT_POINTER(node->children, array);
	}

	*head = nodes[0];
	*tail = nodes[used - 1];
}

static void trie_publish_tail_append(struct stack_depot_trie_child_array *array,
				     unsigned int pos,
				     const struct stack_depot_trie_node *head)
{
	/*
	 * Writers hold stack_depot_trie_writer_lock. Existing children are
	 * immutable, so tail append publishes the new child before increasing the
	 * visible count. Lockless readers that see the old count miss; readers that
	 * see the new count either load the initialized child or treat a transient
	 * NULL as a miss. The writer-lock recheck prevents permanent duplicates.
	 */
	rcu_assign_pointer(array->children[pos], head);
	WRITE_ONCE(array->nr_children, pos + 1);
}

static const struct stack_depot_trie_node *
stack_depot_trie_lookup(const unsigned long *entries, unsigned int nr_entries)
{
	const struct stack_depot_trie_child_array *children;
	unsigned int pos = 0;

	children = trie_load_children_slot(&stack_depot_trie_root);

	while (pos < nr_entries) {
		const struct stack_depot_trie_node *node;
		unsigned int remaining = nr_entries - pos;
		unsigned int matched;
		unsigned int slot;

		if (!trie_child_array_find_slot(children, entries[pos], &slot))
			return NULL;

		node = trie_child_array_load_child(children, slot);
		matched = __stack_depot_trie_node_match(node, &entries[pos], remaining);
		if (matched < node->run.nr_entries)
			return NULL;
		pos += matched;
		if (pos == nr_entries)
			return node->leaf_id ? node : NULL;

		children = trie_load_children_slot(&node->children);
	}

	return NULL;
}

static void trie_build_split(const struct stack_depot_trie_node *child,
			     unsigned int matched, u32 leaf_id,
			     const unsigned long *entries, unsigned int nr_entries,
			     struct stack_depot_trie_node *prefix,
			     struct stack_depot_trie_node *old_tail,
			     struct stack_depot_trie_node * const *new_nodes,
			     struct stack_depot_trie_child_array * const *chain_arrays,
			     struct stack_depot_trie_child_array *prefix_children,
			     const struct stack_depot_trie_node **new_leaf)
{
	const struct stack_depot_trie_child_array *child_children;
	const struct stack_depot_trie_node *child_parent;
	const unsigned long *tail_entries;
	const struct stack_depot_trie_node *split_children[2];
	const struct stack_depot_trie_node *new_head = NULL;
	const struct stack_depot_trie_node *new_tail = NULL;
	unsigned long new_frame;
	unsigned long old_frame;
	u32 prefix_leaf_id;
	unsigned int tail_len;
	bool has_new_tail;

	child_children = trie_load_children_slot(&child->children);
	child_parent = trie_load_parent(child);

	has_new_tail = matched < nr_entries;
	prefix_leaf_id = has_new_tail ? 0 : leaf_id;
	trie_node_init_slice(prefix, child_parent, prefix_leaf_id, child, 0,
			     matched);
	tail_len = child->run.nr_entries - matched;
	trie_node_init_slice(old_tail, prefix, child->leaf_id, child, matched,
			     tail_len);

	if (has_new_tail) {
		tail_entries = &entries[matched];
		tail_len = nr_entries - matched;
		trie_build_append_chain(prefix, leaf_id, tail_entries, tail_len,
					new_nodes, chain_arrays,
					&new_head, &new_tail);
		stack_depot_trie_node_frame(old_tail, 0, &old_frame);
		stack_depot_trie_node_frame(new_head, 0, &new_frame);
		if (old_frame < new_frame) {
			split_children[0] = old_tail;
			split_children[1] = new_head;
		} else {
			split_children[0] = new_head;
			split_children[1] = old_tail;
		}
		trie_child_array_init(prefix_children, ARRAY_SIZE(split_children),
				      split_children, ARRAY_SIZE(split_children));
	} else {
		split_children[0] = old_tail;
		trie_child_array_init(prefix_children, 1, split_children, 1);
	}
	RCU_INIT_POINTER(old_tail->children, child_children);
	RCU_INIT_POINTER(prefix->children, prefix_children);
	*new_leaf = has_new_tail ? new_tail : prefix;
}

static unsigned int trie_size_append_chain(const unsigned long *entries,
					   unsigned int nr_entries,
					   size_t *node_sizes)
{
	unsigned int pos = 0;
	unsigned int nr_nodes = 0;

	while (pos < nr_entries) {
		struct stack_depot_frame_run run;

		frame_run_init(&entries[pos], nr_entries - pos, &run);
		node_sizes[nr_nodes] = trie_node_bytes(&run);
		pos += run.nr_entries;
		nr_nodes++;
	}

	return nr_nodes;
}

static u32
trie_insert_path_locked(const struct stack_depot_trie_child_array __rcu **slot,
			 struct stack_depot_trie_node *parent,
			 const struct stack_depot_trie_child_array *children,
			 unsigned int pos, const unsigned long *entries,
			 unsigned int nr_entries, void **pool_prealloc,
			 struct stack_depot_trie_side_prealloc *side_prealloc)
{
	struct stack_depot_trie_insert_alloc *alloc = stack_depot_trie_alloc;
	struct stack_depot_trie_child_array *slot_array;
	struct stack_depot_trie_child_array *tail_array;
	const struct stack_depot_trie_node *chain_head;
	const struct stack_depot_trie_node *chain_leaf;
	unsigned int capacity = 1;
	unsigned int nr_nodes;
	unsigned long flags;
	u32 new_leaf_id;
	size_t slot_array_size;
	bool tail_append = false;

	if (children) {
		capacity = roundup_pow_of_two(children->nr_children + 1);
		tail_append = pos == children->nr_children &&
			children->nr_children < children->capacity;
		slot_array_size = tail_append ? 0 : trie_child_array_bytes(capacity);
	} else {
		slot_array_size = trie_child_array_bytes(capacity);
	}
	if (slot_array_size > DEPOT_POOL_SIZE)
		return 0;

	new_leaf_id = trie_side_table_prepare_leaf_slot(side_prealloc);
	if (!new_leaf_id)
		return 0;
	nr_nodes = trie_size_append_chain(entries, nr_entries, alloc->node_sizes);
	if (trie_pool_alloc_insert(alloc, pool_prealloc, nr_nodes, nr_nodes - 1,
					0, slot_array_size))
		return 0;
	trie_build_append_chain(parent, new_leaf_id, entries, nr_entries,
				      alloc->nodes, alloc->chain_arrays,
				      &chain_head, &chain_leaf);
	trie_side_table_publish_new_leaf(new_leaf_id, chain_leaf);

	if (!children) {
		slot_array = alloc->slot_array;
		slot_array->nr_children = 1;
		slot_array->capacity = 1;
		RCU_INIT_POINTER(slot_array->children[0], chain_head);
		rcu_assign_pointer(*slot, slot_array);
		return new_leaf_id;
	}

	if (tail_append) {
		tail_array = (struct stack_depot_trie_child_array *)children;
		trie_publish_tail_append(tail_array, pos, chain_head);
	} else {
		slot_array = alloc->slot_array;
		trie_child_array_insert_at(children, pos, chain_head,
					   slot_array, capacity);
		rcu_assign_pointer(*slot, slot_array);
		raw_spin_lock_irqsave(&pool_lock, flags);
		trie_retire_child_array_locked(children);
		raw_spin_unlock_irqrestore(&pool_lock, flags);
	}

	return new_leaf_id;
}

static u32
trie_split_child_locked(const struct stack_depot_trie_child_array __rcu **slot,
			const struct stack_depot_trie_child_array *children,
			const struct stack_depot_trie_node *child,
			unsigned int pos, unsigned int matched,
			const unsigned long *entries, unsigned int nr_entries,
			void **pool_prealloc,
			struct stack_depot_trie_side_prealloc *side_prealloc)
{
	struct stack_depot_trie_insert_alloc *alloc = stack_depot_trie_alloc;
	struct stack_depot_trie_child_array *prefix_children;
	struct stack_depot_trie_child_array *slot_array;
	const struct stack_depot_trie_node *split_leaf;
	struct stack_depot_frame_run run;
	struct stack_depot_trie_node *split_prefix;
	struct stack_depot_trie_node *old_tail;
	unsigned int new_nodes = 0;
	u32 new_leaf_id;
	bool has_new_tail;

	new_leaf_id = trie_side_table_prepare_leaf_slot(side_prealloc);
	if (!new_leaf_id)
		return 0;

	run = child->run;
	run.nr_entries = matched;
	alloc->node_sizes[0] = trie_node_bytes(&run);
	run.nr_entries = child->run.nr_entries - matched;
	alloc->node_sizes[1] = trie_node_bytes(&run);
	has_new_tail = matched < nr_entries;
	if (has_new_tail)
		new_nodes = trie_size_append_chain(&entries[matched],
						   nr_entries - matched,
						   &alloc->node_sizes[2]);
	if (trie_pool_alloc_insert(alloc, pool_prealloc, 2 + new_nodes,
					 new_nodes ? new_nodes - 1 : 0,
					 trie_child_array_bytes(has_new_tail ? 2 : 1),
					 trie_child_array_bytes(children->capacity)))
		return 0;

	slot_array = alloc->slot_array;
	prefix_children = alloc->prefix_children;
	split_prefix = alloc->nodes[0];
	old_tail = alloc->nodes[1];
	trie_build_split(child, matched, new_leaf_id, entries, nr_entries,
			 split_prefix, old_tail, &alloc->nodes[2],
			 alloc->chain_arrays, prefix_children, &split_leaf);
	trie_side_table_publish_split_leaves(child->leaf_id, old_tail,
					     new_leaf_id, split_leaf);
	trie_child_array_replace_at(children, split_prefix, slot_array, pos);
	trie_reparent_children(old_tail);
	rcu_assign_pointer(*slot, slot_array);
	trie_retire_child_array_with_node(children, child);

	return new_leaf_id;
}

static u32
trie_promote_child_locked(const struct stack_depot_trie_child_array __rcu **slot,
			  const struct stack_depot_trie_child_array *children,
			  const struct stack_depot_trie_node *child,
			  unsigned int pos, void **pool_prealloc,
			  struct stack_depot_trie_side_prealloc *side_prealloc)
{
	struct stack_depot_trie_insert_alloc *alloc = stack_depot_trie_alloc;
	struct stack_depot_trie_child_array *slot_array;
	struct stack_depot_trie_node *promoted_node;
	u32 new_leaf_id;
	size_t slot_array_size;

	new_leaf_id = trie_side_table_prepare_leaf_slot(side_prealloc);
	if (!new_leaf_id)
		return 0;
	alloc->node_sizes[0] = trie_node_bytes(&child->run);
	slot_array_size = trie_child_array_bytes(children->capacity);
	if (trie_pool_alloc_insert(alloc, pool_prealloc, 1, 0, 0,
				       slot_array_size))
		return 0;

	promoted_node = alloc->nodes[0];
	slot_array = alloc->slot_array;
	memcpy(promoted_node, child, alloc->node_sizes[0]);
	promoted_node->leaf_id = new_leaf_id;
	trie_side_table_publish_new_leaf(new_leaf_id, promoted_node);
	trie_child_array_replace_at(children, promoted_node, slot_array, pos);
	trie_reparent_children(promoted_node);
	rcu_assign_pointer(*slot, slot_array);
	trie_retire_child_array_with_node(children, child);

	return new_leaf_id;
}

static u32 trie_finish_insert(u32 leaf_id)
{
	if (leaf_id)
		trie_side_table_last_leaf_id = leaf_id;
	return leaf_id;
}

static u32
stack_depot_trie_insert_locked(const unsigned long *entries,
			       unsigned int nr_entries,
			       void **pool_prealloc,
			       struct stack_depot_trie_side_prealloc *side_prealloc)
{
	const struct stack_depot_trie_child_array *children;
	const struct stack_depot_trie_child_array __rcu **slot =
		&stack_depot_trie_root;
	const struct stack_depot_trie_node *child;
	struct stack_depot_trie_node *parent = NULL;
	unsigned int matched;
	unsigned int pos;
	u32 leaf_id = 0;

	for (;;) {
		children = trie_load_children_slot(slot);
		/* Case 1: no matching child, so append the remaining stack path. */
		if (!trie_child_array_find_slot(children, entries[0], &pos)) {
			leaf_id = trie_insert_path_locked(slot, parent, children, pos,
							   entries, nr_entries,
							   pool_prealloc,
							   side_prealloc);
			break;
		}

		child = trie_child_array_load_child(children, pos);
		matched = __stack_depot_trie_node_match(child, entries, nr_entries);

		/* Case 2: the match ends inside the child run, so split it. */
		if (matched < child->run.nr_entries) {
			leaf_id = trie_split_child_locked(slot, children, child, pos,
							       matched, entries, nr_entries,
							       pool_prealloc, side_prealloc);
			break;
		}

		/* Case 3: the input ends here, so reuse or promote this child. */
		if (matched == nr_entries) {
			if (child->leaf_id)
				return child->leaf_id;
			leaf_id = trie_promote_child_locked(slot, children, child, pos,
							       pool_prealloc,
							       side_prealloc);
			break;
		}

		/* Case 4: the child matched fully; descend with remaining frames. */
		parent = (struct stack_depot_trie_node *)child;
		slot = &parent->children;
		entries += matched;
		nr_entries -= matched;
	}

	return trie_finish_insert(leaf_id);
}

static unsigned int
__stack_depot_trie_fetch_into(const struct stack_depot_trie_node *leaf,
			      unsigned long *entries,
			      unsigned int max_entries)
{
	const struct stack_depot_trie_node *node;
	unsigned int total;
	unsigned int pos;
	unsigned int i;

	total = 0;
	for (node = leaf; node; node = trie_load_parent(node))
		total += node->run.nr_entries;
	if (WARN_ON_ONCE(!total))
		return 0;
	if (max_entries < total)
		return 0;

	pos = total;
	for (node = leaf; node; node = trie_load_parent(node)) {
		pos -= node->run.nr_entries;
		for (i = 0; i < node->run.nr_entries; i++)
			stack_depot_trie_node_frame(node, i, &entries[pos + i]);
	}

	return total;
}

static unsigned int
__stack_depot_trie_fetch_handle_into(depot_stack_handle_t handle,
				     unsigned long *entries,
				     unsigned int max_entries)
{
	const struct stack_depot_trie_node *leaf;
	u32 leaf_id;
	unsigned int nr_entries;

	leaf_id = __stack_depot_trie_leaf_id(handle);
	rcu_read_lock_sched_notrace();
	leaf = __stack_depot_trie_side_table_lookup(leaf_id);
	if (WARN_ONCE(!leaf, "corrupt trie handle %08x\n", handle)) {
		rcu_read_unlock_sched_notrace();
		return 0;
	}
	nr_entries = __stack_depot_trie_fetch_into(leaf, entries, max_entries);
	rcu_read_unlock_sched_notrace();
	if (nr_entries)
		kmsan_unpoison_memory(entries, nr_entries * sizeof(*entries));

	return nr_entries;
}

unsigned int stack_depot_fetch(depot_stack_handle_t handle,
			       unsigned long **entries)
{
	struct stack_record *stack;

	*entries = NULL;
	/*
	 * Let KMSAN know *entries is initialized. This shall prevent false
	 * positive reports if instrumented code accesses it.
	 */
	kmsan_unpoison_memory(entries, sizeof(*entries));

	if (!handle || stack_depot_disabled)
		return 0;
	if (WARN_ON_ONCE(stack_depot_handle_is_trie(handle)))
		return 0;

	stack = depot_fetch_stack(handle);
	/*
	 * Should never be NULL, otherwise this is a use-after-put (or just a
	 * corrupt handle).
	 */
	if (WARN(!stack, "corrupt handle or use after stack_depot_put()"))
		return 0;

	*entries = stack->entries;
	return stack->size;
}
EXPORT_SYMBOL_GPL(stack_depot_fetch);

unsigned int stack_depot_fetch_into(depot_stack_handle_t handle,
				    unsigned long *entries,
				    unsigned int max_entries)
{
	struct stack_record *stack;
	unsigned int nr_entries;

	if (!handle)
		return 0;
	if (stack_depot_disabled)
		return 0;
	WARN_ON_ONCE(!entries || !max_entries);
	if (stack_depot_handle_is_trie(handle))
		return __stack_depot_trie_fetch_handle_into(handle, entries,
						      max_entries);

	stack = depot_fetch_stack(handle);
	if (!stack)
		return 0;
	nr_entries = stack->size;
	if (WARN_ON_ONCE(!nr_entries))
		return 0;
	if (nr_entries > max_entries)
		return 0;

	memcpy(entries, stack->entries, nr_entries * sizeof(*entries));
	kmsan_unpoison_memory(entries, nr_entries * sizeof(*entries));
	return nr_entries;
}
EXPORT_SYMBOL_GPL(stack_depot_fetch_into);

void stack_depot_put(depot_stack_handle_t handle)
{
	struct stack_record *stack;

	if (!handle || stack_depot_disabled)
		return;
	if (WARN_ON_ONCE(stack_depot_handle_is_trie(handle)))
		return;

	stack = depot_fetch_stack(handle);
	/*
	 * Should always be able to find the stack record, otherwise this is an
	 * unbalanced put attempt (or corrupt handle).
	 */
	if (WARN(!stack, "corrupt handle or unbalanced stack_depot_put()"))
		return;

	if (WARN_ON_ONCE(stack->flags & STACK_DEPOT_FLAG_COUNTABLE))
		return;
	if (refcount_dec_and_test(&stack->count))
		depot_free_stack(stack);
}
EXPORT_SYMBOL_GPL(stack_depot_put);

void stack_depot_print(depot_stack_handle_t stack)
{
	unsigned long *entries;
	unsigned int nr_entries;

	if (stack_depot_handle_is_trie(stack)) {
		unsigned long trie_entries[CONFIG_STACKDEPOT_MAX_FRAMES];

		nr_entries = __stack_depot_trie_fetch_handle_into(stack,
								  trie_entries,
								  ARRAY_SIZE(trie_entries));
		stack_trace_print(trie_entries, nr_entries, 0);
		return;
	}

	nr_entries = stack_depot_fetch(stack, &entries);
	if (nr_entries > 0)
		stack_trace_print(entries, nr_entries, 0);
}
EXPORT_SYMBOL_GPL(stack_depot_print);

int stack_depot_snprint(depot_stack_handle_t handle, char *buf, size_t size,
		       int spaces)
{
	unsigned long *entries;
	unsigned int nr_entries;

	if (stack_depot_handle_is_trie(handle)) {
		unsigned long trie_entries[CONFIG_STACKDEPOT_MAX_FRAMES];

		nr_entries = __stack_depot_trie_fetch_handle_into(handle,
								  trie_entries,
								  ARRAY_SIZE(trie_entries));
		return stack_trace_snprint(buf, size, trie_entries, nr_entries,
					   spaces);
	}

	nr_entries = stack_depot_fetch(handle, &entries);
	return nr_entries ? stack_trace_snprint(buf, size, entries, nr_entries,
						spaces) : 0;
}
EXPORT_SYMBOL_GPL(stack_depot_snprint);

depot_stack_handle_t __must_check stack_depot_set_extra_bits(
			depot_stack_handle_t handle, unsigned int extra_bits)
{
	union handle_parts parts = { .handle = handle };

	/* Don't set extra bits on empty handles. */
	if (!handle)
		return 0;

	parts.extra = extra_bits;
	return parts.handle;
}
EXPORT_SYMBOL(stack_depot_set_extra_bits);

unsigned int stack_depot_get_extra_bits(depot_stack_handle_t handle)
{
	union handle_parts parts = { .handle = handle };

	return parts.extra;
}
EXPORT_SYMBOL(stack_depot_get_extra_bits);

static int stats_show(struct seq_file *seq, void *v)
{
	/*
	 * data race ok: These are just statistics counters, and approximate
	 * statistics are ok for debugging.
	 */
	seq_printf(seq, "pools: %d\n", data_race(pools_num));
	for (int i = 0; i < DEPOT_COUNTER_COUNT; i++)
		seq_printf(seq, "%s: %ld\n", counter_names[i], data_race(counters[i]));

	return 0;
}
DEFINE_SHOW_ATTRIBUTE(stats);

static int depot_debugfs_init(void)
{
	struct dentry *dir;

	if (stack_depot_disabled)
		return 0;

	dir = debugfs_create_dir("stackdepot", NULL);
	debugfs_create_file("stats", 0444, dir, NULL, &stats_fops);
	return 0;
}
late_initcall(depot_debugfs_init);
