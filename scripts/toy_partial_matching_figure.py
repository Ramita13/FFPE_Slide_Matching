"""Figure for docs/research/03: plain CPD absorbs a missing fragment; piecewise-rigid coverage reports it.
Usage (repo root): python scripts/toy_partial_matching_figure.py 3"""
import sys, numpy as np, cv2
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toy_partial_matching as t
from pycpd import DeformableRegistration
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 3)
while True:
    frags, masks = t.make_block(rng)
    if len(frags) >= 4: break
bm = np.clip(sum(masks),0,1).astype(np.uint8); h=np.sqrt(bm.sum()/300)
B = t.grid_points(bm,h,rng); fid=np.full(len(B),-1)
for i,m in enumerate(masks): fid[m[B[:,1].astype(int),B[:,0].astype(int)]>0]=i
drop=int(np.argmin([m.sum() for m in masks]))
smask, sc = t.make_slide(rng, frags, drop, True, False)
S = t.grid_points(smask, h*sc, rng)/sc
_, lab = cv2.connectedComponents(smask); sl = lab[(S[:,1]*sc).astype(int),(S[:,0]*sc).astype(int)]
tf, dt = t.global_rigid(S, bm); Sa=tf(S); Sp=t.piecewise_refine(Sa, sl, dt)
mu=B.mean(0); sd=np.sqrt(((B-mu)**2).sum(1).mean())
reg=DeformableRegistration(X=(B-mu)/sd, Y=(Sa-mu)/sd, w=0.0, max_iterations=150); TY,_=reg.register(); TY=TY*sd+mu
uc=t.coverage_u(B,TY,2*h); up=t.coverage_u(B,Sp,2*h)
fig,ax=plt.subplots(1,3,figsize=(13,4.5))
for a in ax: a.set_aspect('equal'); a.invert_yaxis(); a.set_xticks([]); a.set_yticks([])
ax[0].scatter(*B.T,s=6,c='0.7',label='block'); ax[0].scatter(*B[fid==drop].T,s=8,c='C3',label='removed fragment'); ax[0].scatter(*Sa.T,s=5,c='C0',label='slide (rigid init)'); ax[0].legend(fontsize=7); ax[0].set_title('input after global rigid+flip')
ax[1].scatter(*B.T,s=6,c='0.7'); ax[1].scatter(*TY.T,s=5,c='C0'); ax[1].scatter(*B[uc>0.5].T,s=14,marker='x',c='C3'); ax[1].set_title(f'plain CPD (w=0): flagged {uc.mean():.1%} of block')
ax[2].scatter(*B.T,s=6,c='0.7'); ax[2].scatter(*Sp.T,s=5,c='C0'); ax[2].scatter(*B[up>0.5].T,s=14,marker='x',c='C3'); ax[2].set_title(f'piecewise rigid coverage: flagged {up.mean():.1%}')
fig.suptitle(f'removed fragment = {(fid==drop).mean():.1%} of block tissue (red x = reported unmatched)')
fig.tight_layout(); fig.savefig("results/toy_partial_matching/example_cpd_vs_coverage.png",dpi=110)
print((fid==drop).mean(), uc.mean(), up.mean())
