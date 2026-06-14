"""
# Figure 5 RP2 model comparison analysis

This code compares computational models predicting water-excretion onset from RP2 spike timing.

Input:
A tab-delimited file containing RP2 spike times and water-excretion onset times for multiple imaging recordings.

What the code does:

1. Builds a spike-level dataset from RP2 spike timing.
2. Defines water-excretion-associated spikes as the last RP2 spike occurring within 20 seconds before water-excretion onset.
3. Tests spike-count models using RP2 spike count in the previous 30, 60, or 120 seconds.
4. Tests a short-interspike-interval model.
5. Tests leaky evidence-accumulation models across candidate τ values from 5 to 300 seconds.
6. Fits logistic regression models predicting water-excretion onset.
7. Compares model performance using AICc and ROC AUC.
8. Saves model comparison tables and plots.

Output:
CSV files containing model comparison results and spike-level model data, plus figure files showing leaky evidence traces, excretion-aligned evidence, AICc model comparison, and the leaky-model probability curve.

Notes:
This script was used for the model comparison shown in Figure 5C–D. The primary analysis excludes visually ambiguous water-excretion events, while a sensitivity analysis includes them. The leaky evidence-accumulation model selected τ by maximum likelihood across candidate values.

"""



import pandas as pd, numpy as np
from math import log
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

path='/mnt/data/Pasted text(17).txt'
df=pd.read_csv(path, sep='\t', header=None, engine='python')
groups=[(0,1,2),(5,6,7),(10,11,12),(15,16,17),(21,22,23)]
records=[]; ex_records=[]
for gi,(a,b,c) in enumerate(groups,1):
    spikes=pd.to_numeric(df.iloc[2:,b], errors='coerce').dropna().values.astype(float)
    ex=pd.to_numeric(df.iloc[2:,c], errors='coerce').dropna().values.astype(float)
    for t in spikes: records.append({'imaging':gi,'spike_time':t})
    for t in ex: ex_records.append({'imaging':gi,'excretion_time':t,'ambiguous':False})
# add ambiguous for sensitivity
ambiguous=[(5,3649.0),(5,9742.0)]
for gi,t in ambiguous: ex_records.append({'imaging':gi,'excretion_time':t,'ambiguous':True})
spikes_df=pd.DataFrame(records).sort_values(['imaging','spike_time']).reset_index(drop=True)
ex_df=pd.DataFrame(ex_records).sort_values(['imaging','excretion_time']).reset_index(drop=True)


def build_dataset(include_ambiguous=False, max_lag=20.0):
    rows=[]
    for gi,g in spikes_df.groupby('imaging'):
        st=np.array(g['spike_time'])
        ex=np.array(ex_df[(ex_df.imaging==gi) & ((~ex_df.ambiguous) | include_ambiguous)]['excretion_time'])
        # last preceding spike within max_lag for each excretion is positive
        positive_idx=set()
        lags=[]
        for e in ex:
            inds=np.where(st<=e)[0]
            if len(inds):
                idx=inds[-1]
                if 0 <= e-st[idx] <= max_lag:
                    positive_idx.add(idx)
        for j,t in enumerate(st):
            prev=st[:j+1]
            rows.append({
                'imaging':gi,'time':t,'y':1 if j in positive_idx else 0,
                'count30':np.sum(prev>=t-30),
                'count60':np.sum(prev>=t-60),
                'count120':np.sum(prev>=t-120),
                'isi': (t-st[j-1]) if j>0 else np.nan,
            })
    return pd.DataFrame(rows)

def logit_fit(x,y):
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    X=np.column_stack([np.ones_like(x), x])
    def nll(b):
        z=X@b
        # stable log likelihood
        return -np.sum(y*z - np.logaddexp(0,z))
    res=minimize(nll, np.array([np.log((y.mean()+1e-6)/(1-y.mean()+1e-6)),0.0]), method='BFGS')
    b=res.x; ll=-nll(b); p=expit(X@b)
    return b,ll,p,res.success

def aicc(ll,k,n):
    aic=2*k-2*ll
    return aic + (2*k*(k+1))/(n-k-1)

def fit_all(include_ambiguous=False):
    data=build_dataset(include_ambiguous)
    y=data.y.values
    results=[]; preds={}
    # null
    p0=np.repeat(y.mean(), len(y))
    ll0=np.sum(y*np.log(p0+1e-12)+(1-y)*np.log(1-p0+1e-12))
    results.append({'model':'Null','AICc':aicc(ll0,1,len(y)),'AUC':0.5,'ll':ll0,'k':1})
    preds['Null']=p0
    for feat in ['count30','count60','count120']:
        b,ll,p,ok=logit_fit(data[feat].values,y)
        results.append({'model':feat,'AICc':aicc(ll,2,len(y)),'AUC':roc_auc_score(y,p),'ll':ll,'k':2,'coef':b[1],'intercept':b[0]})
        preds[feat]=p
    # isi: smaller isi should predict; use log isi or negative, fill large for first
    isi=data['isi'].fillna(1e6).clip(lower=0.5)
    x=-np.log(isi.values)
    b,ll,p,ok=logit_fit(x,y)
    results.append({'model':'short_isi','AICc':aicc(ll,2,len(y)),'AUC':roc_auc_score(y,p),'ll':ll,'k':2,'coef':b[1],'intercept':b[0]})
    preds['short_isi']=p
    best=None
    for tau in np.linspace(5,300,296):
        E=[]
        for gi,g in data.groupby('imaging'):
            times=g.time.values
            for j,t in enumerate(times):
                prev=times[:j+1]
                E.append(np.exp(-(t-prev)/tau).sum())
        b,ll,p,ok=logit_fit(np.array(E),y)
        if best is None or ll>best['ll']:
            best={'tau':tau,'ll':ll,'p':p,'b':b,'E':np.array(E)}
    results.append({'model':f'leaky_tau_{best["tau"]:.0f}s','AICc':aicc(best['ll'],3,len(y)),'AUC':roc_auc_score(y,best['p']),'ll':best['ll'],'k':3,'coef':best['b'][1],'intercept':best['b'][0],'tau':best['tau'], 'boundary': -best['b'][0]/best['b'][1]})
    preds['leaky']=best['p']
    data['leaky_E']=best['E']; data['leaky_pred']=best['p']
    resdf=pd.DataFrame(results).sort_values('AICc').reset_index(drop=True)
    return data,resdf,best,preds

primary,res_primary,best_primary,preds_primary=fit_all(False)
sens,res_sens,best_sens,preds_sens=fit_all(True)
print('Primary')
print(res_primary.to_string(index=False))
print('positive',primary.y.sum(),'n',len(primary))
print('best tau',best_primary['tau'],'boundary',-best_primary['b'][0]/best_primary['b'][1])
print('Sensitivity')
print(res_sens.to_string(index=False))
print('positive',sens.y.sum(),'n',len(sens))

# Save results tables
res_primary.to_csv('/mnt/data/hydra_rp2_model_comparison_primary.csv', index=False)
res_sens.to_csv('/mnt/data/hydra_rp2_model_comparison_sensitivity.csv', index=False)
primary.to_csv('/mnt/data/hydra_rp2_spike_level_model_data_primary.csv', index=False)

# Plot raster/evidence for primary
# Compute continuous evidence at 1 s grid per imaging using best tau
tau=best_primary['tau']
fig,axes=plt.subplots(5,1,figsize=(10,8),sharex=False)
# to respect no specific colors? This is analysis not final? If user-visible? But using matplotlib defaults colors ok.
for ax,(gi,g) in zip(axes, spikes_df.groupby('imaging')):
    st=g.spike_time.values
    ex=ex_df[(ex_df.imaging==gi)&(~ex_df.ambiguous)].excretion_time.values
    amb=ex_df[(ex_df.imaging==gi)&(ex_df.ambiguous)].excretion_time.values
    tgrid=np.arange(0, max(st.max(), ex.max() if len(ex) else 0, amb.max() if len(amb) else 0)+1, 2.0)
    E=np.zeros_like(tgrid,dtype=float)
    for s in st:
        idx=tgrid>=s
        E[idx]+=np.exp(-(tgrid[idx]-s)/tau)
    ax.plot(tgrid,E,label='RP2 evidence')
    ax.vlines(st, 0, max(E.max(),1)*0.08, linewidth=0.6, alpha=0.5)
    for e in ex: ax.axvline(e, linestyle='--', linewidth=1.2)
    for e in amb: ax.axvline(e, linestyle=':', linewidth=1.2)
    ax.set_ylabel(f'Img {gi}')
axes[-1].set_xlabel('Time (s)')
fig.suptitle(f'RP2 spike-triggered leaky evidence (tau = {tau:.0f} s)')
fig.text(0.02,0.5,'Accumulated RP2 evidence',rotation=90,va='center')
plt.tight_layout(rect=[0.04,0.02,1,0.96])
fig.savefig('/mnt/data/hydra_rp2_leaky_evidence_traces.png',dpi=300)
plt.close(fig)

# Onset aligned evidence primary, controls random times maybe all excretion. Plot individual and mean around excretion +/-200s
window=(-200,100); step=2; rel=np.arange(window[0],window[1]+step,step)
curves=[]
for _,r in ex_df[~ex_df.ambiguous].iterrows():
    gi=r.imaging; e=r.excretion_time
    st=spikes_df[spikes_df.imaging==gi].spike_time.values
    vals=[]
    for rr in rel:
        t=e+rr
        prev=st[st<=t]
        vals.append(np.exp(-(t-prev)/tau).sum() if len(prev) else 0)
    curves.append(vals)
curves=np.array(curves)
fig,ax=plt.subplots(figsize=(6,4))
for c in curves: ax.plot(rel,c,alpha=0.25,linewidth=1)
ax.plot(rel,curves.mean(axis=0),linewidth=2.5,label='Mean')
ax.axvline(0,linestyle='--',linewidth=1)
ax.set_xlabel('Time from water excretion initiation (s)')
ax.set_ylabel('Accumulated RP2 evidence')
ax.set_title('RP2 evidence aligned to water excretion')
plt.tight_layout()
fig.savefig('/mnt/data/hydra_rp2_excretion_aligned_evidence.png',dpi=300)
plt.close(fig)

# Model comparison bar AICc delta
res=res_primary.copy(); res['delta_AICc']=res.AICc-res.AICc.min()
fig,ax=plt.subplots(figsize=(6.5,4))
ax.barh(res['model'], res['delta_AICc'])
ax.invert_yaxis()
ax.set_xlabel('ΔAICc')
ax.set_title('Model comparison, primary analysis')
plt.tight_layout()
fig.savefig('/mnt/data/hydra_rp2_model_comparison_aicc_primary.png',dpi=300)
plt.close(fig)

# Probability curve leaky
E=primary.leaky_E.values; y=primary.y.values
b=best_primary['b']
xx=np.linspace(0,max(E)*1.1,200); pp=expit(b[0]+b[1]*xx)
fig,ax=plt.subplots(figsize=(5,4))
# jitter y
rng=np.random.default_rng(1)
ax.scatter(E, y+rng.normal(0,0.03,len(y)), alpha=0.5, s=18)
ax.plot(xx,pp,linewidth=2)
ax.axvline(-b[0]/b[1],linestyle='--',linewidth=1)
ax.set_xlabel('Accumulated RP2 evidence')
ax.set_ylabel('Probability of water excretion')
ax.set_title('Leaky accumulator model')
plt.tight_layout()
fig.savefig('/mnt/data/hydra_rp2_leaky_probability_curve_primary.png',dpi=300)
plt.close(fig)
