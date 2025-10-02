// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2025 Cloudflare */

#include <vmlinux.h>
#include <errno.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_core_read.h>
#include "bpf_misc.h"

#define NR_LOOPS 10000
#define DATA_SIZE 64
#define MAX_ENTRIES 1024

char _license[] SEC("license") = "GPL";

struct buffer {
	__u8 data[DATA_SIZE];
};

struct {
	//__uint(type, BPF_MAP_TYPE_LRU_PERCPU_HASH);
	__uint(type, BPF_MAP_TYPE_LRU_HASH);
	__type(key, int);
	__type(value, struct buffer);
	__uint(max_entries, MAX_ENTRIES);
	//__uint(map_flags, BPF_F_NO_COMMON_LRU);
} lru_hash SEC(".maps");

long duration_ns;
long hits;

static int update(__u32 index, int *retval)
{
	struct buffer val = {};
	int key = index % MAX_ENTRIES;
	int err;

	err = bpf_map_update_elem(&lru_hash, &key, &val, BPF_ANY);
	if (err) {
		*retval = err;
		return 1;
	}

	return 0;
}

SEC("xdp")
int BPF_PROG(run_bench)
{
	u64 start, delta;
	int err, loops;

	start = bpf_ktime_get_ns();
	loops = bpf_loop(NR_LOOPS, update, &err, 0);
	delta = bpf_ktime_get_ns() - start;

	__sync_add_and_fetch(&duration_ns, delta);
	__sync_add_and_fetch(&hits, loops);

	return err;
}
