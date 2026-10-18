## What's Changed in v1.12.0 (STRUCT-6)

`codon-infer v1.12.0` enhances the all-atom protein kinematic reconstruction engine (`STRUCT-4` NeRF + DEE, `STRUCT-5` Ramachandran backbone conditioning) with directional hydrogen bonding potentials, screened electrostatic salt bridges, and an analytical MolProbity steric barrier (`STRUCT-6`), while preserving zero dynamic heap allocations in packing hot-paths.

### 🌟 Key Highlights & Biophysical Architecture

* **Directional 10–12 Hydrogen Bonding Potential (Rosetta / Kortemme-Baker style):**
  * Normalized radial Lennard-Jones attractive well ($r_0 = 2.80\text{ \AA}$, active in $[2.30, 3.40\text{ \AA}]$) with $C^1$-continuous smoothstep tapering between $3.0$ and $3.4\text{ \AA}$.
  * Strictly bounded to negative values to serve as a stabilizing bonus ($w_{\text{hb}} = 2.50\text{ kcal/mol}$).
  * $\cos^2(\theta)$ directional angular modulation for backbone carbonyl oxygen acceptors ($C=O \cdots D$) and backbone amide nitrogen donors ($N-H \cdots A$).
* **Screened Electrostatic Salt Bridges & Bidentate Normalization:**
  * Screened Lorentzian / Debye-Hückel potential ($r_{\text{opt}} = 2.90\text{ \AA}$, $\sigma = 0.60\text{ \AA}$, window $[2.30, 4.20\text{ \AA}]$) modeling aqueous counterion screening ($w_{\text{salt}} = 3.00\text{ kcal/mol}$).
  * Fractional charge weighting (`charge_weight`) normalizing resonant guanidinium (`ARG`) and carboxylate (`ASP`/`GLU`) groups to prevent unphysical multi-atom over-attraction, matching experimental $\Delta\Delta G$ ranges ($\approx -3.0\text{ kcal/mol}$).
  * +50% like-charge electrostatic repulsion preventing unphysical cation-cation or anion-anion clustering.
* **Strict MolProbity Steric Clash Barrier ($d < 2.0\text{ \AA}$):**
  * Steep linear repulsion ramp ($+50.0 \times (1.0 - d / 2.0)$) ensuring attractive electrostatic and H-bond bonuses never override physical electron shell exclusion below $2.0\text{ \AA}$.
* **Zero-Allocation Chemical Bitmasking (`SideChainAtom`):**
  * Chemical classification flags (`ATOM_POLAR_DONOR`, `ATOM_POLAR_ACCEPTOR`, `ATOM_CATION`, `ATOM_ANION`) packed into alignment padding bytes, maintaining the exact 32-byte struct size and zero-heap `Copy` semantics.
* **Pixel-Perfect CLI Output:**
  * Synchronized `--help` and `-V` banners dynamically bound to `CARGO_PKG_VERSION`.

### 📊 Validation & Benchmark Metrics

| Dataset / Target | Total Clashes | Clashscore | sc-bb (sc) | sc-sc (sc) | Covalent-like (<1.65 Å) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Control A (Native Crystal Ground Truth)** | **2** | **0.8** | 0 (0.0) | 2 (0.8) | 0 |
| **Control A (v1.12.0 Reconstructed)** | **32** | **12.3** | 13 (5.0) | 19 (7.3) | 14 |
| *└─ 1AOY (ARGR E. coli)* | **1** | **1.8** | 1 (1.8) | 0 (0.0) | 0 |
| *└─ 2GP8 (VG08 BPP22)* | **0** | **0.0** | 0 (0.0) | 0 (0.0) | 0 |
| **Control B (ESM-2 650M Unoptimized)** | 566 | 214.3 | 345 (130.6) | 195 (73.8) | 283 |
| **Control B (v1.12.0 Empaquetado)** | **194** | **73.5** | **120 (45.4)** | **48 (18.2)** | **61 (-78.4% vs init)** |

### 🛠️ Verification & Test Suite
* **Unit Tests:** 70 tests (69 passed, 1 ignored, 0 failures).
* **Dependencies:** 0 third-party crates (`std` only).
* **Binary Footprint:** 1.5 MB (`codon-infer`) / 1.1 MB (`codon-infer-mcp`).
