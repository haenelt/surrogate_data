# Surrogate Data

[![Test and formatting](https://github.com/haenelt/surrogate_data/actions/workflows/test.yml/badge.svg)](https://github.com/haenelt/surrogate_data/actions/workflows/test.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3127/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Generate null distribution with preserved autocorrelation.

## Procedure

- let $x\in\mathbb R^2$

### Step 1: random permutation

- generate $x_0^{\prime}$ (random permutation of $x$)

### Step 2: smoothing of permuted map

- perform a local kernel-weighted sum of values in $x_0^{\prime}$ to construct a smoothed map $x_k^{\prime}$
- single elements $x_i$ are computed by

$$
x_{k,i}=\frac{\sum_{j=1}^{k}K(d_{ij})x_{0,j}^{\prime}}{\sum_{j=1}^{k}K(d_{ij})}
$$

- $k$: number of nearest-neighboring regions used to perform the smoothing
- $K$: distance-dependent smoothing kernel (Burt et al., 2020) uses an exponentially decaying smoothing (truncated) kernel)
- $d_{ij}$: distance separating region $i$ and $j$
- $k$ is chosen from a set of pre-defined values

### Step 3: rescale permuted map

- rescale $x_k^{\prime}$ such that its spatial autocorrelation approximately matches the spatial autocorrelation in the target map
- this is achieved by maximizing the fit between $\gamma(x)$ and $\gamma(x_k^{\prime})$ by linear regression
- for each $k$ (sweep through $k$'s)
  - $\gamma(x)=\beta\gamma(x_k^{\prime})+\alpha+\epsilon$
  - compute sum of squared errors (SSE)
- select $k$ with minimum SSE denoted as $k^*$

### Step 4: construct surrogate map

- $z$: normally distributed random variable with zero mean and unit variance

$$
\hat{x}=|\beta_{k^{*}}|^{1/2} x_{{k^1}}
$$

### Step 5: generate null distribution

- generate null distirbution by repeating step 1 and step 2 multiple times

## Variogram $\gamma$

- a variogram is a summary measure of the autocorrelation in spatial data
- measure of pairwise variation as a function of distance
- typically computed within finite width distance intervals

$$
\gamma(h\pm\delta)=\frac{1}{2N(h\pm\delta)}\sum_{i=1}^{N(h\pm\delta)}\sum_{i\neq j}^{N(h\pm\delta)}(x_i-x_j)^2
$$

- $h$: length scale
- $2\delta$: width around $h$
- $N(h\pm\delta)$: number of sample pairs separated by a distance $d_{ij}$, which lie in the interval $h-\delta\leq d_{ij}\leq h+\delta$

## Smoothed version of $\gamma$

- following Villadomat et al., 2014
- reduce noise in $\gamma$ by smoothing

$$
\gamma(h)=\frac{\sum_{i=1}^{N}\sum_{j=i+1}^{N}w_{ij}v_{ij}}{\sum_{i=1}^{N}\sum_{j=i+1}^{N}w_{ij}}
$$

- $v_{ij}=\frac12(x_i-x_j)^2$
- $w_{ij}=\exp(-(2.68s)^2/(2b)^2)$
- $w_{ij}$: Gaussian kernel which falls off smoothly
- $s=|h-d_{ij}|$
- $b$: bandwidth
- $b$ controls the smoothness
- constants are chosen in the kernel such that the quartiles of the kernel are at $\pm0.25b$
- spatial autocorrelation is primarily a local effect:
  - only consider $\{d_{ij}\}$ in the bottom 25th percentile of the distribution
- evaluate $\gamma(h)$ at 25 uniformly spaced distance intervals found in $\{d_{ij}\}$
  - $\{h_n=h_0+2\delta n;\,n=0,1,\dots,24\}$
  - $h_0=\min(\{d_{ij}\})$
  - $h_{24}=\text{25th percentile of values in }\{d_{ij}\}$
- $b$ for dense variograms was chosen to be three times the distance interval spacing, i.e., $b=3\Delta h=6\delta$
