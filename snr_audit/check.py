"""Check the archived gain expressions, replay ledger, and Table 5 responses."""
import argparse
import ast
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.signal import firwin, freqz

TARGETS = (-20, -10, -5, 0, 5, 10, 20)


def binding(node, namespace):
    module = ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    exec(compile(module,"<archived source>","exec"),namespace)


def function(tree, name):
    return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)


def test_gain(archive):
    rng=np.random.default_rng(216)
    clean=rng.normal(size=(32,8064)).astype("float32")
    direct=rng.normal(size=8064).astype("float32")
    aa=(.57*rng.normal(size=8064)+.05).astype("float32")
    profile=np.exp(-.2*np.abs(np.arange(32)-16)).astype("float32")
    cases=[]
    exponents=[]
    for name in archive.namelist():
        if not name.endswith(".py") or name=="snr_audit/check.py":
            continue
        tree=ast.parse(archive.read(name),filename=name)
        power_terms=[n for n in ast.walk(tree) if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Pow)
                     and isinstance(n.left,ast.Constant) and n.left.value==10]
        for n in power_terms:
            exponents.append({"file":name,"line":n.lineno,"exponent":ast.unparse(n.right)})
        if name.endswith("followup.py"):
            root={"np":np}
            binding(function(tree,"power"),root)
            impl=function(tree,"amount")
            binding(impl,root)
            achieved=[]
            for db in TARGETS:
                n=profile[:,None]*aa
                alpha=root["amount"](clean,n,db)
                achieved.append(10*np.log10(root["power"](clean)/root["power"](alpha*n)))
            cases.append((name,impl.lineno,"global scalar",TARGETS,achieved))
        elif name.endswith("paired_audit.py"):
            root={"np":np}
            active=next(n for n in tree.body if isinstance(n,ast.Assign)
                        and any(isinstance(t,ast.Name) and t.id=="ACTIVE" for t in n.targets))
            binding(active,root)
            impl=function(tree,"scales")
            binding(impl,root)
            achieved=[]
            for db in TARGETS:
                root["SNR"]=db
                gains=root["scales"](clean,direct,aa,profile)
                for i,(wave,scope) in enumerate(((direct,slice(None)),(aa,slice(None)),
                                                 (direct,root["ACTIVE"]),(direct,slice(None)))):
                    product=gains[i,:,None]*wave
                    achieved.append(10*np.log10(np.mean(clean[scope]**2)/np.mean(product[scope]**2)))
            cases.append((name,impl.lineno,"separable spatial gains",np.repeat(TARGETS,4),achieved))
        elif name.endswith(("calibration_replay.py","calibration_boundary.py")):
            for n in ast.walk(tree):
                if not (isinstance(n,ast.Call) and ast.unparse(n.func)=="np.sqrt"
                        and any(a in power_terms for a in ast.walk(n))):
                    continue
                expr=compile(ast.Expression(n),name,"eval")
                targets=(-5,) if name.endswith("calibration_boundary.py") else TARGETS
                cp=np.array([[2.0,1.0],[1.8,.7]])
                npow=np.array([[1.0,.1],[.3,.02]])
                achieved=[]
                for value in targets:
                    # Test every operator index in the vector path and the chosen index
                    # in each of the two fixed-target expressions.
                    gain=eval(expr,{"np":np,"cp":cp,"npow":npow,"TARGET":value,"iop":1})
                    op=0 if "cp[0, 0]" in ast.unparse(n) else 1
                    ratio=cp[:,0]/npow[:,0] if np.ndim(gain) else cp[op,0]/npow[op,0]
                    achieved.extend(np.atleast_1d(10*np.log10(ratio/gain**2)))
                targets=np.repeat(targets,2) if len(achieved)==2*len(targets) else targets
                cases.append((name,n.lineno,"power-array gain",targets,achieved))
    results=[]
    for name,line,method,wanted,actual in cases:
        error=float(np.max(np.abs(np.array(actual)-wanted)))
        assert error<1e-5,(name,line,error)
        results.append({"file":name,"line":line,"method":method,"max_abs_snr_error_db":error})
    assert len(results)==sum(x["exponent"]!="mean_log / 10" for x in exponents)
    return {"all_python_files_scanned":sum(n.endswith(".py") and n!="snr_audit/check.py" for n in archive.namelist()),
            "sample_cases_tested":sum(len(x[3]) for x in cases),
            "gain_locations":results,"base_10_power_inventory":exponents,
            "demonstration_noise_to_signal_power":{"-5":float(10**.5),"-20":100.0}}


def ledger(archive):
    names=("calibration_order/results/calibration_trials.csv",
           "spatial_calibration/calibration_trials.csv",
           "emg_release/results/aligned/calibration_trials.csv")
    output={}
    for name in names:
        rows=list(csv.DictReader(io.StringIO(archive.read(name).decode("utf-8"))))
        def field(key):
            return np.array([r[key] for r in rows],dtype=float)
        shift=10*np.log10(field("noise_retained")/field("clean_retained"))
        np.testing.assert_allclose(field("beta_post_minus_pre_db"),shift,atol=1e-9,rtol=0)
        np.testing.assert_allclose(field("broadband_pre_db"),-5-shift,atol=1e-9,rtol=0)
        hp=[r for r in rows if r["operator"]=="hp"]
        output[name]={"rows":len(rows),"hp":{
            m:{k:float(np.mean([float(r[k]) for r in hp if r["mixture"]==m]))
               for k in ("noise_retained","beta_pre_db","broadband_pre_db")}
            for m in ("line","synthetic","joint")}}
    return output


def responses(archive):
    frequency=np.array([50,100,150])
    rc=1/(1+1j*frequency/83)
    # The independent CIC calculation uses convolution of finite boxcar FIRs.
    box=np.ones(16)/16
    def cic(n):
        kernel=box
        for _ in range(n-1):
            kernel=np.convolve(kernel,box)
        return freqz(kernel,fs=2048,worN=frequency)[1]
    fir=firwin(2501,64,fs=16000,window=("kaiser",5))
    paths={"direct":np.ones(3),"unfiltered_decimation":np.ones(3),
           "rc1":rc,"rc2":rc**2,"cic3":cic(3),"cic5":cic(5),
           "rc1_cic5":rc*cic(5),"polyphase":freqz(fir,fs=16000,worN=frequency)[1]}
    old=json.loads(archive.read("front_end/front_end_summary.json"))["tone_gain_db"]
    ref=json.loads(archive.read("methods_audit/methods_audit_summary.json"))["polyphase_fir"]["gain_db"]
    checked={k:{str(int(f)):float(a) for f,a in zip(frequency,20*np.log10(abs(h)))}
             for k,h in paths.items()}
    for key,values in checked.items():
        if key=="unfiltered_decimation":
            continue
        expected=ref if key=="polyphase" else old[key]
        np.testing.assert_allclose(list(values.values()),[expected[f] for f in values],rtol=0,atol=1e-8)
    return checked


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive",type=Path)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    with ZipFile(args.archive) as z:
        data={"gain_audit":test_gain(z),"calibration_ledger":ledger(z),"gains_db":responses(z)}
    args.out.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
