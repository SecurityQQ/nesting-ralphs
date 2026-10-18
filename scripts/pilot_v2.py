import asyncio, json, time
from trace_critic import nemo
from trace_critic.nemo import generate_dod, initial_state, ralph, USAGE
from trace_critic.tasks_v2 import FTASKS
async def main():
    t=time.time()
    dods = await asyncio.gather(*[generate_dod(t) for t in FTASKS.values()])
    D={d["task"]:d for d in dods if d}
    nemo.save_json("runs/v2/dod.json", D)
    print("dod ok", len(D), "/", len(FTASKS), round(time.time()-t,1),"s")
    for d in D.values(): print(f"  {d['task']:16s} tests={d['n_tests']} fail_buggy={d['fail_on_buggy']} pass_ref={d['pass_on_reference']}/{d['n_tests']} dropped={len(d['dropped_by_auditor'])} att={d['attempt']}")
    for model in ["nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B","nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"]:
        nemo.EXEC_MODEL=model; t=time.time()
        jobs=[(n,s) for n in D for s in range(3)]
        rs=await asyncio.gather(*[ralph(FTASKS[n],D[n],initial_state(FTASKS[n],D[n]),6) for n,s in jobs])
        succ=sum(r[0] for r in rs); fd=sum(r[2]-r[0] for r in rs)
        per={}
        for (n,s),r in zip(jobs,rs): per.setdefault(n,[]).append((int(r[0]),r[1]))
        print(model.split('/')[-1], "success", succ,"/",len(rs), "false-done verifs", fd, "mean calls", sum(r[1] for r in rs)/len(rs), round(time.time()-t,1),"s")
        print("  ", {k:v for k,v in per.items()})
    print(USAGE)
asyncio.run(main())
