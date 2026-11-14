"""汇总第二个 LLM（Llama）与 Qwen 的对比，输出 results/llama_summary/。用法（在仓库根目录）：python cpu/15b_summarize_llama.py .
复用 cpu/15_summarize_v3.py 中的读取与检验函数。"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
V2=Path(sys.argv[1]); sys.path.insert(0,str(V2)); sys.argv=['x']
src=open(V2/'cpu/15_summarize_v3.py',encoding='utf-8').read().split('\nif __name__')[0].replace('RESULTS / "v3_summary"','RESULTS / "llama_summary"')
g={'__file__':str(V2/'cpu/15_summarize_v3.py'),'__name__':'x'}; exec(src,g)
load,wil,OUT=g['load_proba'],g['wil'],g['OUT']
DS=['cse2018','nf_unsw_v3','cic2017']; DET=['SAORF','RF-grid','LightGBM','XGBoost']
Q=lambda ds:('v2_' if ds=='cse2018' else '')
def qs(ds): return {'Q-SFT':Q(ds)+'llm_sft_relov_filt','Q-base':Q(ds)+'llm_base_filt'}
L={'L-SFT':'llm_sft_llama_relov_filt','L-SFT-tau':'llm_sft_llama_filt','L-SFT-noov':'llm_sft_llama_noneov_filt','L-base':'llm_base_llama_filt'}
md=[]
# ---- A original split
rows=[]
for ds in DS:
    d=load(ds); m=dict(none='none',cw='cw',ros='ros',smote='smote',**qs(ds),**L)
    for det in DET:
        x=d[d.detector==det]; piv=x.pivot_table(index='seed',columns='source',values='rare_f1'); ap=x.groupby('source').rare_ap.mean(); fp=x.groupby('source').fp10k.mean()
        for k,s in m.items():
            if s not in piv: continue
            r=dict(dataset=ds,detector=det,src=k,rare_f1=piv[s].mean(),rare_ap=ap.get(s,np.nan),fp10k=fp.get(s,np.nan))
            for ref in ['none','cw','ros','Q-SFT']:
                rs=m[ref]
                if rs in piv and rs!=s:
                    b=piv[[s,rs]].dropna(); r['d_'+ref]=(b[s]-b[rs]).mean(); r['p_'+ref]=wil(b[s].values,b[rs].values)
            rows.append(r)
A=pd.DataFrame(rows); A.to_csv(OUT/'orig_llama_vs_qwen.csv',index=False)
# ---- B resplit r1-r3
rows=[]
for ds in DS:
    parts=[]
    for r in (1,2,3):
        p=load(f'{ds}_r{r}'); p['rep']=r; parts.append(p)
    d=pd.concat(parts); m={'none':'none','cw':'cw','ros':'ros','smote':'smote','Q-SFT':'llm_sft_relov_filt','L-SFT':'llm_sft_llama_relov_filt','L-SFT-tau':'llm_sft_llama_filt','L-SFT-noov':'llm_sft_llama_noneov_filt'}
    for det in DET:
        x=d[d.detector==det].groupby(['rep','source'])[['rare_f1','rare_ap','fp10k']].mean().reset_index()
        piv=x.pivot(index='rep',columns='source',values='rare_f1')
        for k,s in m.items():
            if s not in piv: continue
            h=x[x.source==s]; r=dict(dataset=ds,detector=det,src=k,rare_f1=h.rare_f1.mean(),sd=h.rare_f1.std(ddof=0),rare_ap=h.rare_ap.mean(),fp10k=h.fp10k.mean())
            for ref in ['none','cw','ros','Q-SFT']:
                rs=m[ref]
                if rs in piv and rs!=s:
                    b=piv[[s,rs]].dropna(); r['d_'+ref]=(b[s]-b[rs]).mean(); r['w_'+ref]=f"{int((b[s]>b[rs]).sum())}/{len(b)}"
            rows.append(r)
B=pd.DataFrame(rows); B.to_csv(OUT/'resplit_llama_vs_qwen.csv',index=False)
# ---- C fidelity
F=[]
for ds in DS:
    f=pd.read_csv(V2/f'results/{ds}/fidelity_llama/fidelity_summary.csv'); f.insert(0,'dataset',ds); F.append(f)
F=pd.concat(F); F.to_csv(OUT/'fidelity_llama_vs_qwen.csv',index=False)
# ---- D gen stats
G=[]
for ds in DS+[f'{b}_r{r}' for b in DS for r in (1,2,3)]:
    for n,qn in [('llm_sft_llama','llm_sft'),('llm_base_llama','llm_base')]:
        for tag,nm in [('Llama',n),('Qwen',(('v2_' if ds=='cse2018' else '')+qn))]:
            fg=V2/f'data/{ds}/llm/gen_{nm}_stats.json'; fr=V2/f'data/{ds}/llm/reward_{nm}_stats.json'
            if not fg.exists(): continue
            a=json.load(open(fg)); b=json.load(open(fr)) if fr.exists() else {}
            G.append(dict(dataset=ds,model=tag,stage=qn,parse=a.get('parse_rate'),unique=a.get('unique_rate'),accept=b.get('accept_rate'),mem=b.get('mem_rate_of_valid'),acc_per_class=b.get('accepted_per_class')))
G=pd.DataFrame(G); G.to_csv(OUT/'gen_stats_llama_vs_qwen.csv',index=False)
pd.set_option('display.width',250)
print('== A'); print(A.round(4).to_string(index=False))
print('== B'); print(B.round(4).to_string(index=False))
print('== C'); print(F.round(3).to_string(index=False))
print('== D'); print(G.drop(columns='acc_per_class').round(3).to_string(index=False))
