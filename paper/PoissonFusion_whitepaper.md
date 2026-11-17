# PoissonFusion

## Boundary-driven screened harmonic transfer for identity-preserving image insertion

**Technical white paper — experimental Rust prototype v0.1**  
**18 November 2026**

### Abstract

Poisson cloning removes cutout seams by preserving source gradients while
forcing destination values at the pasted region’s boundary. Its strength is
also its central failure mode: the boundary discrepancy becomes a harmonic
correction that can propagate throughout the object, tinting brand colours,
flattening material appearance, or destroying small text and texture.

This paper proposes a compact alternative, PoissonFusion. Rather than solving
for every output pixel, it solves only for a smooth, low-frequency correction
field \(c\) added to the original source. The field is the unique minimizer of a
screened Dirichlet energy. Its single principal parameter, the integration
length \(\ell\), has a direct spatial meaning: in a locally flat boundary model,
the correction decays as \(e^{-d/\ell}\). Therefore \(\ell\to0\) approaches an
identity-preserving alpha paste, while \(\ell\to\infty\) approaches the harmonic
correction of classic Poisson cloning.

The boundary data is not taken from raw pixels. It is estimated from
linear-light, low-frequency target colour and a mask-normalized source blur,
preventing transparent pixels and high-frequency object detail from entering
the illumination correction. The full-resolution source is then passed through
exactly and only \(c\) is added before soft-alpha compositing.

The method is implemented as a reusable, safe-Rust crate and CLI with a
matrix-free, Jacobi-preconditioned conjugate-gradient solver. In a controlled
760×480 regression, the faithful setting retained 0.979 high-frequency detail
correlation and 18.16 dB interior PSNR, versus approximately 0.12 and 11.5 dB
for OpenCV normal and mixed Poisson cloning. The experiment is diagnostic, not
evidence of universal state-of-the-art performance.

## 1. Research question

The practical goal is not merely “remove a seam.” It is:

> Integrate a cutout into a new image while giving the artist an explicit,
> mathematically predictable control over how far the destination may alter the
> source object.

That distinction matters in product, luxury, fashion, portrait, and editorial
work. A visually smooth output is a failure if a logo is softened, a material
finish is recoloured, or facial identity is changed.

The image-composition literature separates several problems: placement,
boundary blending, foreground harmonization, shadow generation, reflection
generation, geometry consistency, and semantic consistency [1]. No scalar PDE
can solve all of them. PoissonFusion deliberately addresses only the boundary
and low-frequency appearance component. Matte estimation, physical relighting,
contact shadows, reflections, and perspective remain independent stages.

## 2. Scientific prior art

### 2.1 Poisson cloning

Pérez, Gangnet, and Blake formulate seamless cloning as [2]

\[
f^*=\arg\min_f\int_\Omega\|\nabla f-\nabla S\|^2\,dx,
\qquad f|_{\partial\Omega}=T|_{\partial\Omega}.
\]

The Euler equation is \(\Delta f=\Delta S\). Defining the correction
\(h=f-S\) reveals the mechanism more clearly:

\[
\Delta h=0,\qquad h|_{\partial\Omega}=T-S.
\]

Classic Poisson cloning is therefore the source plus a harmonic extension of
the source/target boundary mismatch. A harmonic field has no intrinsic decay
length; a boundary offset can influence the entire connected region. OpenCV’s
documentation explicitly notes the resulting tint when source and destination
colours do not already match [3].

### 2.2 Colour preservation and screened Poisson

The idea of adding source-colour fidelity to Poisson editing is established
prior art, not a new claim. Bhat et al. derive the screened Poisson objective
combining a data term and a gradient term, and provide a Fourier-domain solution
[4]. Bie, Huang, and Wang add appearance-control energies to reduce colour
bleeding [5]. Later work adds a fidelity term and a boundary band for
photomontage [6]. Modified Poisson Blending uses information from both source
and destination boundaries plus an alpha step [7]. Image Melding combines a
screened Poisson solver with patch synthesis [8].

Screened Poisson is also used outside cloning. Mechrez, Shechtman, and
Zelnik-Manor retain stylized colour while recovering gradients from the original
photograph [9]. Deep Image Blending combines Poisson, content, and style losses
but requires iterative neural optimization [10]. These papers establish the
scientific validity of balancing gradient agreement and appearance fidelity.

### 2.3 Solvers and modern alternatives

Green Function Convolution and discrete sine transforms provide fast routes to
Laplacian reconstruction [11]. Recent neural Poisson solvers move the
variational problem into continuous implicit representations [12]. For the
broader appearance problem, PCT-Net performs full-resolution pixel-wise colour
transforms [13], while Harmonizer reports a compact model for high-resolution
image and video harmonization [14]. Generative insertion systems can alter
lighting and geometry more radically, but may redraw the object and thus cannot
guarantee asset identity [15, 16].

### 2.4 The remaining design gap

Existing screened formulations usually express a weight between a reconstructed
image and a data image. The numeric weight is mathematically valid but not
necessarily intuitive to an artist. PoissonFusion recasts the problem around
three design requirements:

1. solve a correction field, not a replacement image;
2. make the control a spatial decay length measured in pixels;
3. isolate the correction to low-frequency boundary mismatch while passing the
   full-resolution source through exactly.

This combination is the proposed contribution. The screened operator itself,
fidelity regularization, normalized convolution, Gaussian scale separation, and
conjugate gradients are all established techniques. A formal novelty or patent
claim would require a broader prior-art search.

## 3. The formulation

Let:

- \(S:\Omega\to\mathbb{R}^3\) be the placed source in linear RGB;
- \(T:D\to\mathbb{R}^3\) be the destination in linear RGB;
- \(m:D\to[0,1]\) be the source alpha;
- \(\Omega=\{x:m(x)\ge\tau\}\) be the PDE domain;
- \(G_\sigma\) be a normalized Gaussian kernel;
- \(\ell>0\) be the integration length.

All equations act independently on the three linear-light colour channels; the
vector notation is only for compactness.

### 3.1 Mask-normalized low-frequency source

Transparent pixels have undefined colour. A conventional blur would mix those
undefined samples into the boundary estimate. PoissonFusion uses normalized
convolution:

\[
\bar S_\sigma(x)=
\frac{G_\sigma*(mS)(x)}{G_\sigma*m(x)+\varepsilon}.
\]

The target base is

\[
\bar T_\sigma(x)=G_\sigma*T(x).
\]

The boundary mismatch is

\[
\delta_\sigma(x)=\bar T_\sigma(x)-\bar S_\sigma(x),
\qquad x\in\partial\Omega.
\]

Only broad exposure and colour differences enter \(\delta_\sigma\). Text,
pores, fabric weave, label edges, and other high-frequency source structures do
not become boundary conditions.

### 3.2 Screened harmonic correction

The correction field is

\[
\boxed{
c^*=\arg\min_{c|_{\partial\Omega}=\delta_\sigma}
\left[
\frac12\int_\Omega\|\nabla c\|^2dx+
\frac{1}{2\ell^2}\int_\Omega\|c\|^2dx
\right].
}
\]

Its Euler–Lagrange equation is the modified Helmholtz, or homogeneous screened
Poisson, equation:

\[
\boxed{
(-\Delta+\ell^{-2})c=0\quad\text{in }\Omega,
\qquad c=\delta_\sigma\quad\text{on }\partial\Omega.
}
\]

The final candidate foreground is simply

\[
F=S+c^*.
\]

The composite is evaluated in linear light:

\[
\boxed{C=(1-m)T+mF.}
\]

No source detail is reconstructed. The only modification is the smooth field
\(c^*\).

## 4. Why \(\ell\) is the right control

### 4.1 Local closed-form interpretation

Consider a half-plane with perpendicular distance \(d\ge0\) from a locally
straight boundary and constant mismatch \(\delta\). The PDE reduces to

\[
-c''(d)+\ell^{-2}c(d)=0,
\quad c(0)=\delta,
\quad c(\infty)=0.
\]

The solution is

\[
\boxed{c(d)=\delta e^{-d/\ell}.}
\]

Thus \(\ell\) is the e-folding distance of appearance transfer. At one \(\ell\),
37% of the boundary correction remains; at three \(\ell\), only 5% remains.
This is more interpretable than an arbitrary regularization weight.

### 4.2 Limiting behaviour

Because \(\mu=\ell^{-2}\):

- \(\ell\to0\): \(\mu\to\infty\), so \(c\to0\) away from an infinitesimal
  boundary layer. The method approaches identity-preserving alpha paste.
- \(\ell\to\infty\): \(\mu\to0\), so \(\Delta c=0\). The method approaches the
  harmonic correction underlying classic Poisson cloning.

PoissonFusion is therefore not a binary alternative to Poisson. It is a
continuous family connecting direct paste to classic Poisson with one spatially
meaningful control.

## 5. Well-posedness and uniqueness

On a four-neighbour pixel graph, let \(x\) collect correction values at interior
pixels after Dirichlet boundary values have been moved to the right-hand side.
The discrete system is

\[
(L_D+\mu I)x=b,\qquad\mu=\ell^{-2}>0,
\]

where \(L_D\) is the Dirichlet graph Laplacian.

**Proposition.** The matrix \(A=L_D+\mu I\) is symmetric positive definite for
every finite \(\ell>0\). Therefore the discrete energy has exactly one minimizer.

**Proof.** For any non-zero vector \(x\),

\[
x^TAx=
\sum_{(p,q)\in E_{II}}(x_p-x_q)^2+
\sum_{(p,b)\in E_{I\partial}}x_p^2+
\mu\sum_{p\in I}x_p^2.
\]

Every term is non-negative and the final term is strictly positive because
\(\mu>0\) and \(x\ne0\). Hence \(x^TAx>0\), so \(A\) is SPD. A finite-dimensional
strictly convex quadratic has one unique minimizer. ∎

This is the precise sense in which the core is mathematically “perfect”: it is
well-posed, stable under finite perturbations, deterministic, and has a unique
solution. It does not mean that one PDE can make every composite perceptually
perfect.

## 6. Rust implementation

The reference crate implements the following pipeline:

```text
RGBA source + RGB destination
      │
      ├─ decode sRGB → linear RGB
      ├─ crop an active ROI with Gaussian padding
      ├─ normalized Gaussian source base
      ├─ Gaussian destination base
      ├─ boundary mismatch δσ
      ├─ matrix-free PCG solve of (LD + ℓ⁻²I)c = b
      ├─ full-resolution source + c
      └─ linear-light soft-alpha composite → sRGB
```

The solver stores one integer mapping, four neighbour indices, the diagonal,
and a small number of vectors over the unknown pixels. It never constructs a
general sparse matrix.

For \(N\) ROI pixels, Gaussian radius \(r\), \(M\) PDE unknowns, and \(k\) PCG
iterations, the current complexity is

\[
O(Nr+kM)\text{ time},\qquad O(N+M)\text{ memory}.
\]

The implementation uses safe Rust, linear-light arithmetic, deterministic
iteration order, a Jacobi preconditioner, configurable residual tolerance, and
byte-for-byte preservation of destination pixels outside the source overlap.

Public API:

```rust
let result = poissonfusion::blend(
    &background_rgb,
    &foreground_rgba,
    (x, y),
    poissonfusion::BlendConfig::faithful(),
)?;
```

The CLI exposes `faithful` (\(\ell=2\) px), `balanced` (\(\ell=6\) px), and
`seamless` (\(\ell=18\) px) presets, plus direct controls for \(\ell\), \(\sigma\),
solver tolerance, placement, and diagnostics.

## 7. Controlled regression

The included fixture constructs a bottle with a crisp label, creates a known
target-scene version, and applies a broad exposure/colour cast to produce the
copied source. It compares alpha paste, OpenCV normal clone, OpenCV mixed clone,
and the Rust faithful setting. The image is 760×480. Runtime is deliberately
excluded from the scientific table because the independent OpenCV and Rust
paths showed large scheduling variance on the shared development machine; the
machine-readable benchmark retains the raw indicative measurements.

| Method | Interior PSNR dB ↑ | Boundary PSNR dB ↑ | Detail corr. ↑ | OKLab error ↓ |
|---|---:|---:|---:|---:|
| Alpha paste | 18.440 | 25.263 | 0.99822 | 0.073219 |
| OpenCV Normal Clone | 11.640 | 17.342 | 0.12802 | 0.175209 |
| OpenCV Mixed Clone | 11.380 | 17.271 | 0.12317 | 0.183318 |
| PoissonFusion Rust, \(\ell=2\) | 18.156 | 17.690 | 0.97920 | 0.075278 |

The Rust solve contains 43,112 unknowns, converges all three channels in 32 PCG
iterations, and reaches relative residuals below \(10^{-5}\).

The result should be read as a Pareto test. Alpha paste is fastest and closest
to the synthetic ground truth in this deliberately simple case. PoissonFusion
stays close to that identity baseline while outperforming both classic Poisson
modes on interior fidelity, detail correlation, and colour error. It does not
prove superiority on arbitrary photographs.

## 8. Recommended production architecture

For a faithful “copy, paste, integrate” tool:

1. **Matte:** estimate/refine alpha, including hair, defocus, translucency, and
   motion blur.
2. **Placement:** set scale, perspective, and occlusion.
3. **PoissonFusion:** integrate the boundary with an artist-controlled \(\ell\).
4. **Optional harmonization:** apply a bounded low-frequency colour transform
   only when the scene mismatch is too large for boundary transfer.
5. **Shadow/reflection:** generate or paint them as independent layers.
6. **Quality control:** measure source identity, text/logo legibility, seam
   energy, and perceptual realism separately.

Diffusion-based insertion should be a separate creative mode. It can improve
global realism but cannot guarantee exact product or facial identity.

## 9. Limitations and next science

- The boundary alpha is thresholded to define a hard PDE domain; a future soft
  graph formulation should incorporate alpha uncertainty directly.
- The method corrects additive linear-light colour. Strong contrast inversion,
  directional relighting, and specular changes require a richer model.
- Gaussian normalized convolution is robust but not edge-aware. A bilateral or
  guided certainty-weighted base may improve mixed foreground/background edges.
- PCG is CPU-only and single-threaded. Multigrid, FFT/DST, Metal, CUDA, or WebGPU
  should make high-resolution interaction practical.
- There is no temporal term. Video requires flow-aware boundary data and
  temporal regularization.
- One synthetic case is insufficient. Evaluation should include iHarmony4,
  real product cutouts, portraits, hair, transparent glass, HDR/16-bit data,
  adversarial masks, and blinded user preference.

The most important next experiment is a full ablation over:

1. raw versus low-frequency boundary mismatch;
2. ordinary versus mask-normalized source blur;
3. sRGB versus linear-light computation;
4. \(\ell\) relative to object scale;
5. hard versus soft domains;
6. PCG, multigrid, FFT/DST, and GPU backends.

Metrics should keep seam quality and identity fidelity separate. Collapsing them
into one score would hide the exact trade-off this formulation is intended to
make controllable.

## 10. References

1. Niu et al., “Making Images Real Again: A Comprehensive Survey on Deep Image
   Composition,” arXiv:2106.14490, revised 2026.
   https://arxiv.org/abs/2106.14490
2. Pérez, Gangnet, and Blake, “Poisson Image Editing,” ACM TOG / SIGGRAPH 2003.
   https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf
3. OpenCV, “Seamless Cloning” documentation.
   https://docs.opencv.org/4.13.0/df/da0/group__photo__clone.html
4. Bhat, Curless, Cohen, and Zitnick, “Fourier Analysis of the 2D Screened
   Poisson Equation for Gradient Domain Problems,” ECCV 2008.
   https://grail.cs.washington.edu/wp-content/uploads/2015/08/bhat08spe.pdf
5. Bie, Huang, and Wang, “Free Appearance-Editing with Improved Poisson Image
   Cloning,” Journal of Computer Science and Technology 26(6), 2011.
   https://doi.org/10.1007/s11390-011-1197-5
6. Dizdaroğlu, “Gradient Domain Photomontage via Interactive Object Selection,”
   Turkish Journal of Electrical Engineering and Computer Sciences 26(3), 2018.
   https://doi.org/10.3906/elk-1412-4
7. Afifi and Hussain, “MPB: A Modified Poisson Blending Technique,”
   Computational Visual Media 1(4), 2015.
   https://github.com/mahmoudnafifi/modified-Poisson-image-editing
8. Darabi et al., “Image Melding: Combining Inconsistent Images using
   Patch-based Synthesis,” ACM TOG / SIGGRAPH 2012.
   https://web.ece.ucsb.edu/~psen/melding
9. Mechrez, Shechtman, and Zelnik-Manor, “Photorealistic Style Transfer with
   Screened Poisson Equation,” 2017. https://arxiv.org/abs/1709.09828
10. Zhang, Wen, and Shi, “Deep Image Blending,” WACV 2020.
    https://arxiv.org/abs/1910.11495
11. Beaini et al., “Fast and Optimal Laplacian Solver for Gradient-Domain Image
    Editing using Green Function Convolution,” 2019.
    https://arxiv.org/abs/1902.00176
12. Neural Poisson Solver: A Universal and Continuous Framework for Natural
    Signal Blending, 2024. https://arxiv.org/abs/2407.08457
13. Guerreiro, Nakazawa, and Stenger, “PCT-Net: Full Resolution Image
    Harmonization Using Pixel-Wise Color Transformations,” CVPR 2023.
    https://github.com/rakutentech/PCT-Net-Image-Harmonization
14. Ke et al., “Harmonizer: Learning to Perform White-Box Image and Video
    Harmonization,” ECCV 2022. https://github.com/ZHKKKe/Harmonizer
15. Song et al., “Insert Anything: Image Insertion via In-Context Editing in
    DiT,” 2025. https://arxiv.org/abs/2504.15009
16. “OSInsert: Towards High-authenticity and High-fidelity Image Composition,”
    2026. https://arxiv.org/abs/2602.19523

---

**Status:** experimental, deterministic research prototype. The mathematical
well-posedness claim applies to the stated discrete energy. No academic novelty,
patentability, or universal perceptual-superiority claim is made.
