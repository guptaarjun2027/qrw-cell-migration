# QRW Cell Migration

Quantum random walk model of cancer cell migration through extracellular matrix networks.

## What this project is

Classical diffusion models predict that cancer cells spread proportionally to the 
square root of time. Experimental tracking data for several cancer cell lines 
(glioma, MDA-MB-231, HT1080) shows spreading that is consistently faster than 
this — with non-Gaussian displacement distributions that classical models cannot 
reproduce.

This project applies continuous-time quantum random walks (QRW) to the same 
network topologies that cells physically migrate through, and asks whether the 
QRW's super-diffusive spreading better fits the experimental mean-square 
displacement (MSD) curves. A decoherence parameter interpolates between the 
fully quantum and fully classical regimes, allowing the model to be fit to data.

## Why this is different from existing work

IBM Research (Dubovitskii et al., 2025) demonstrated that QRWs outperform 
classical walks on abstract gene-gene and cell-cell interaction networks for 
disease gene ranking. This project addresses a distinct question: not which genes 
matter, but how fast and how far cancer cells physically move through tissue — 
a question no prior QRW study has targeted.

## Methods

- Continuous-time QRW via graph Laplacian exponentiation: exp(-iLt)
- Network topologies: 2D lattice, Voronoi tessellation, Barabási-Albert scale-free
- Decoherence modeled via Lindblad master equation (QuTiP)
- Validation against published MSD data from Khain et al. (Phys. Rev. E, 2012)
  and experimental cell tracking datasets (cellmigration.org)

## Status

Work in progress, June–July 2026.