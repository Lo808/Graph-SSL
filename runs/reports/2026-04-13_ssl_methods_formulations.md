# Self-Supervised Graph Methods: Theory and Implemented Formulations

This note summarizes the methods used in this project and gives the exact training objectives implemented in `wl_gcl/src/trainers/train_wl_dino.py`.

## 1) Shared Setup

We work on one graph \(G=(V,E,X)\) with:

- \(N = |V|\) nodes
- node features \(X \in \mathbb{R}^{N \times F}\)
- encoder \(f_\theta\) (GIN/GCN/GAT/WLHN), producing node embeddings \(h_v\)

Student/teacher framework:

- student encoder parameters: \(\theta_s\)
- teacher encoder parameters: \(\theta_t\)
- EMA update each epoch:

\[
\theta_t \leftarrow m\theta_t + (1-m)\theta_s
\]

Projection head (DINO/BYOL branch):

- \(g(\cdot)\): MLP \(d \rightarrow 2d \rightarrow d\), then \(\ell_2\)-normalization.
- \(z_v = g(h_v)\)

Predictor head (BYOL/BGRL branch):

- \(q(\cdot)\): MLP \(d \rightarrow 2d \rightarrow d\)

Optional augmentations (enabled with `use_augmentations=True`):

- edge dropout
- feature masking

If disabled, both student/teacher use the same graph view.

## 2) DINO (Prototype Distillation)

### 2.1 Prototypes

Prototype matrix \(W \in \mathbb{R}^{M \times d}\), \(M=\) `num_prototypes`, with row-wise normalized prototypes.

Student logits for node \(v\):

\[
\ell_s(v) = \frac{W z_s(v)}{\tau_s}
\]

Teacher raw logits:

\[
\ell_t^{raw}(v) = W z_t(v)
\]

### 2.2 Teacher centering + temperature schedule

Center \(c \in \mathbb{R}^{1 \times M}\) updated by EMA:

\[
c \leftarrow \mu c + (1-\mu)\,\mathrm{mean}_v\left[\ell_t^{raw}(v)\right]
\]

Teacher temperature schedule:

\[
\tau_t(e)=
\begin{cases}
\tau_{start} + \frac{e}{E_w}(\tau_t-\tau_{start}), & e < E_w \\
\tau_t, & e \ge E_w
\end{cases}
\]

Teacher target distribution:

\[
q_t(v)=\mathrm{softmax}\left(\frac{\ell_t^{raw}(v)-c}{\tau_t(e)}\right)
\]

Student distribution:

\[
q_s(v)=\mathrm{softmax}\left(\ell_s(v)\right)
\]

### 2.3 DINO loss

\[
\mathcal{L}_{\text{DINO}}
=
-\frac{1}{|B|}\sum_{v\in B}\sum_{m=1}^{M} q_t(v,m)\log q_s(v,m)
\]

Teacher branch is stop-grad.

## 3) WL-Guided DINO (DINO+WL)

WL hierarchy depth \(T\) with WL colors \(C^{(t)}(v)\).

WL distance:

\[
d_{WL}(u,v)=T-\max\{t: C^{(t)}(u)=C^{(t)}(v)\}
\]

Candidate set \(C(v)\) is built from:

- top WL neighbors (`k_wl`)
- top feature neighbors from cosine kNN (`k_feat`, recomputed each epoch)
- optional random negatives
- truncated to `max_candidates`

### 3.1 KL-style WL loss (current DINO+WL default)

WL prior over candidates:

\[
p_{WL}(u|v)
=
\frac{\exp\left(-d_{WL}(u,v)/\tau_{WL}\right)}
{\sum_{u' \in C(v)} \exp\left(-d_{WL}(u',v)/\tau_{WL}\right)}
\]

Student candidate distribution:

\[
q_s(u|v)
=
\mathrm{softmax}\left(\frac{z_s(v)^\top z_s(u)}{\tau_s}\right)_{u\in C(v)}
\]

WL KL loss:

\[
\mathcal{L}_{WL}^{KL}
=
\mathrm{KL}\!\left(p_{WL}(\cdot|v)\;\|\;q_s(\cdot|v)\right)
\]

### 3.2 Alignment-style WL loss (implemented option)

\[
w_{vu}
=
\frac{\exp\left(-d_{WL}(u,v)/\tau_{WL}\right)}
{\sum_{u'\in C(v)}\exp\left(-d_{WL}(u',v)/\tau_{WL}\right)}
\]

\[
\mathcal{L}_{WL}^{align}
=
\frac{1}{|B|}\sum_{v\in B}\sum_{u\in C(v)}
w_{vu}\,\|z_s(v)-z_s(u)\|_2^2
\]

Optional weak repulsion for far nodes \(d_{WL}(u,v)>\delta\):

\[
\mathcal{L}_{rep}
=
\mathrm{mean}\left[\max(0, z_s(v)^\top z_s(u))\right]
\]

\[
\mathcal{L}_{WL}^{align+rep}
=
\mathcal{L}_{WL}^{align} + \beta \mathcal{L}_{rep}
\]

### 3.3 Combined DINO+WL objective

For full objective:

\[
\mathcal{L}_{full}
=
\mathcal{L}_{DINO} + \lambda_{WL}\,\mathcal{L}_{WL}
\]

where \(\mathcal{L}_{WL}\) is either KL or alignment, depending on `wl_loss_type`.

The code also supports adaptive scaling:

\[
\lambda_{WL}^{eff}
=
\lambda_{WL}\cdot
\mathrm{clip}\!\left(
\frac{\mathrm{EMA}(\mathcal{L}_{DINO})}{\mathrm{EMA}(\mathcal{L}_{WL})+\epsilon},
s_{min}, s_{max}
\right)
\]

## 4) BYOL (Graph BYOL variant)

Two augmented views \(x^1, x^2\).

Student:

\[
h_s^1=f_{\theta_s}(x^1), \quad h_s^2=f_{\theta_s}(x^2)
\]
\[
z_s^1=g(h_s^1), \quad z_s^2=g(h_s^2)
\]
\[
p_s^1=q(z_s^1), \quad p_s^2=q(z_s^2)
\]

Teacher (EMA, stop-grad):

\[
z_t^1=g_t(f_{\theta_t}(x^1)), \quad z_t^2=g_t(f_{\theta_t}(x^2))
\]

Regression loss used in code:

\[
\ell(p,z)=2-2\cdot\cos(p,z)
\]

Symmetric BYOL objective:

\[
\mathcal{L}_{BYOL}
=
\frac{1}{2}\ell(p_s^1, z_t^2)
+
\frac{1}{2}\ell(p_s^2, z_t^1)
\]

No negatives and no WL term in pure BYOL mode.

## 5) BGRL (Graph Bootstrap variant in this code)

Same two-view bootstrap template, but distillation is done in encoder space:

- use \(h\) directly instead of projection \(z\) before predictor/target matching.

Student:

\[
h_s^1=f_{\theta_s}(x^1), \quad h_s^2=f_{\theta_s}(x^2)
\]
\[
p_s^1=q(h_s^1), \quad p_s^2=q(h_s^2)
\]

Teacher targets:

\[
h_t^1=f_{\theta_t}(x^1), \quad h_t^2=f_{\theta_t}(x^2)
\]

Loss:

\[
\mathcal{L}_{BGRL}
=
\frac{1}{2}\ell(p_s^1, h_t^2)
+
\frac{1}{2}\ell(p_s^2, h_t^1)
\]

Again, stop-grad on teacher and EMA updates are used.

## 6) Objective Modes in Code

`objective` switch maps to:

- `dino`: \(\mathcal{L}=\mathcal{L}_{DINO}\)
- `wl`: \(\mathcal{L}=\mathcal{L}_{WL}\)
- `full` (`dino_wl`): \(\mathcal{L}=\mathcal{L}_{DINO}+\lambda_{WL}\mathcal{L}_{WL}\)
- `byol`: \(\mathcal{L}=\mathcal{L}_{BYOL}\)
- `bgrl`: \(\mathcal{L}=\mathcal{L}_{BGRL}\)

## 7) Practical Notes for This Repo

- Reported `Acc` during training is linear-probe node classification accuracy on student encoder features.
- DINO-family runs can use prototype distillation (`distill_space=prototype`) or candidate distillation (`candidate`) but recent runs used `prototype`.
- For DINO+WL KL experiments, a large `lambda_wl` was often necessary because \(\mathcal{L}_{WL}\) magnitude is much smaller than \(\mathcal{L}_{DINO}\).
