#!/usr/bin/env python3
"""Table of flash-iobench JSON lines (tools/iobench.sh / iobench_hwchar.sh output).
layers arms: st/tok = IO stall ms per token (mean, p50, p90); lyr.* = stall of one layer with >= 1 miss, ms.
closed arms: lyr.* = latency of one whole-record read, ms (mean, p50, p90, p99).
lat = mean push->READY per read (ms); MiB/s = bytes read / wall; parks = IO worker sleeps;
inv/s = involuntary context switches of the bench threads per second (high = another process on its cores);
softirq = kernel softirq time per CPU over the arm, ms (dm-crypt decrypt runs there)."""
import json,sys
print("%-24s %7s %7s %7s | %6s %6s %6s %6s | %6s %7s %6s %5s | softirq ms by cpu (>40)" % ("label","st/tok","p50","p90","lyr.mn","p50","p90","p99","lat","MiB/s","parks","inv/s"))
for l in open(sys.argv[1]):
    d=json.loads(l)
    io=d['io']; sq=" ".join("%d:%d"%(c[0],c[4]) for c in d['cpu_ms_user_sys_irq_softirq'] if c[4]>40)
    inv=d.get('invol_per_s',-1)
    if d['mode']=='layers':
        t=d['stall_ms_per_token']; y=d['layer_stall_ms']
        extra=""
        if d.get('lookahead'): extra=" spec=%s"%d['spec']
        print("%-24s %7.2f %7.2f %7.2f | %6.3f %6.3f %6.3f %6.3f | %6.3f %7.0f %6d %5.0f | %s%s" % (d['label'],t['mean'],t['p50'],t['p90'],y['mean'],y['p50'],y['p90'],y['p99'],io['mean_latency_ms'],io['mib_per_s_wall'],io['parks'],inv,sq,extra))
    else:
        y=d['latency_ms']
        print("%-24s %7s %7s %7s | %6.3f %6.3f %6.3f %6.3f | %6.3f %7.0f %6d %5.0f | %s  [closed depth %d len %d n %d]" % (d['label'],"","","",y['mean'],y['p50'],y['p90'],y['p99'],io['mean_latency_ms'],io['mib_per_s_wall'],io['parks'],inv,sq,d['depth'],d['len'],y['n']))
