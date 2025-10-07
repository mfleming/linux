// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2025 Cloudflare */

#include <argp.h>
#include <linux/time64.h>
#include <linux/if_ether.h>
#include "lru_hash_bench.skel.h"
#include "bench.h"
#include "testing_helpers.h"

static struct ctx {
	struct lru_hash_bench *bench;
} ctx;

static struct {
	__u32 nr_entries;
} args = {
	.nr_entries = 1024,
};

enum {
	ARG_NR_ENTRIES = 10000,
};

static const struct argp_option opts[] = {
	{"nr_entries", ARG_NR_ENTRIES, "NR_ENTRIES", 0,
	  "Maximum number of entries in the LRU hash map" },
	{},
};

static error_t lru_hash_parse_arg(int key, char *arg, struct argp_state *state)
{
	long ret;

	switch (key) {
	case ARG_NR_ENTRIES:
		ret = strtoul(arg, NULL, 10);
		if (ret < 1 || ret > UINT_MAX) {
			fprintf(stderr, "Invalid nr_entries count");
			argp_usage(state);
		}
		args.nr_entries = ret;
		break;
	default:
		return ARGP_ERR_UNKNOWN;
	}
	return 0;
}

const struct argp bench_lru_hash_argp = {
	.options = opts,
	.parser = lru_hash_parse_arg,
};

static void lru_hash_validate(void)
{
}

static void lru_hash_setup(void)
{
	int ret;

	ctx.bench = lru_hash_bench__open();
	if (!ctx.bench) {
		fprintf(stderr, "failed to open skeleton\n");
		exit(1);
	}

	ctx.bench->bss->max_entries = args.nr_entries;
	bpf_map__set_max_entries(ctx.bench->maps.lru_hash, args.nr_entries);

	ret = lru_hash_bench__load(ctx.bench);
	if (ret) {
		fprintf(stderr, "failed to load skeleton\n");
		exit(1);
	}

	if (lru_hash_bench__attach(ctx.bench)) {
		fprintf(stderr, "failed to attach skeleton\n");
		exit(1);
	}

}

static void lru_hash_measure(struct bench_res *res)
{
	res->hits = atomic_swap(&ctx.bench->bss->hits, 0);
	res->duration_ns = atomic_swap(&ctx.bench->bss->duration_ns, 0);
}

static void *lru_hash_producer(void *unused __always_unused)
{
	int err;
	char in[ETH_HLEN]; /* unused */

	LIBBPF_OPTS(bpf_test_run_opts, opts, .data_in = in,
			.data_size_in = sizeof(in), .repeat = 1, );

	while (true) {
		int fd = bpf_program__fd(ctx.bench->progs.run_bench);
		err = bpf_prog_test_run_opts(fd, &opts);
		if (err) {
			fprintf(stderr, "failed to run BPF prog: %d\n", err);
			exit(1);
		}
	}
	return NULL;
}

/*
 * The standard bench op_report_*() functions assume measurements are
 * taken over a 1-second interval but operations that modify the map
 * (INSERT, DELETE, and FREE) cannot run indefinitely without
 * "resetting" the map to the initial state. Depending on the size of
 * the map, this likely needs to happen before the 1-second timer fires.
 *
 * Calculate the fraction of a second over which the op measurement was
 * taken (to ignore any time spent doing the reset) and report the
 * throughput results per second.
 */
static void frac_second_report_progress(int iter, struct bench_res *res,
					long delta_ns, double rate_divisor,
					char rate)
{
	double hits_per_sec, hits_per_prod;

	hits_per_sec = res->hits / rate_divisor /
		(res->duration_ns / (double)NSEC_PER_SEC);
	hits_per_prod = hits_per_sec / env.producer_cnt;

	printf("Iter %3d (%7.3lfus): ", iter,
			(delta_ns - NSEC_PER_SEC) / 1000.0);
	printf("hits %8.3lf%c/s (%7.3lf%c/prod)\n", hits_per_sec, rate,
			hits_per_prod, rate);
}

static void frac_second_report_final(struct bench_res res[], int res_cnt,
				     double lat_divisor, double rate_divisor,
				     char rate, const char *unit)
{
	double hits_mean = 0.0, hits_stddev = 0.0;
	double latency = 0.0;
	int i;

	for (i = 0; i < res_cnt; i++) {
		double val = res[i].hits / rate_divisor /
			(res[i].duration_ns / (double)NSEC_PER_SEC);
		hits_mean += val / (0.0 + res_cnt);
		latency += res[i].duration_ns / res[i].hits / (0.0 + res_cnt);
	}

	if (res_cnt > 1) {
		for (i = 0; i < res_cnt; i++) {
			double val =
				res[i].hits / rate_divisor /
				(res[i].duration_ns / (double)NSEC_PER_SEC);
			hits_stddev += (hits_mean - val) * (hits_mean - val) /
				(res_cnt - 1.0);
		}

		hits_stddev = sqrt(hits_stddev);
	}
	printf("Summary: throughput %8.3lf \u00B1 %5.3lf %c ops/s (%7.3lf%c ops/prod), ",
			hits_mean, hits_stddev, rate, hits_mean / env.producer_cnt, rate);
	printf("latency %8.3lf %s/op\n",
			latency / lat_divisor / env.producer_cnt, unit);
}

static void lru_hash_report_progress(int iter, struct bench_res *res,
				     long delta_ns)
{
	double rate_divisor = 1000000.0;
	char rate = 'M';

	frac_second_report_progress(iter, res, delta_ns, rate_divisor, rate);
}

static void lru_hash_report_final(struct bench_res res[], int res_cnt)
{
	double lat_divisor = 1.0;
	double rate_divisor = 1000000.0;
	const char *unit = "ns";
	char rate = 'M';

       	frac_second_report_final(res, res_cnt, lat_divisor, rate_divisor, rate, unit);
}

const struct bench bench_lru_hash = {
	.name = "lru-hash",
	.argp = &bench_lru_hash_argp,
	.validate = lru_hash_validate,
	.setup = lru_hash_setup,
	.producer_thread = lru_hash_producer,
	.measure = lru_hash_measure,
	.report_progress = lru_hash_report_progress,
	.report_final = lru_hash_report_final,
};
