# Third-Party Notices

These are third-party sources, not declarations of submission authorship.
Retain required copyright/license texts when preparing an anonymous artifact.

| Component | Upstream | License / status |
| --- | --- | --- |
| KTH VQ-VAE interface and spatial predictor adaptation | [RIVER](https://github.com/Araachie/river) | GPL-3.0; text in licenses/RIVER-GPL-3.0.txt |
| Encoder, decoder and vector quantizer | [Taming Transformers](https://github.com/CompVis/taming-transformers) | MIT; text in licenses/TAMING-TRANSFORMERS-MIT.txt |
| NSE U-Net components | [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) | MIT; text in licenses/DENOISING-DIFFUSION-MIT.txt |
| External Acrobot GPE codec (not bundled) | [GPE](https://github.com/wonjunee/GPE_codes) | Reader obtains source separately under upstream terms; copied encoder/decoder definitions removed from the current tree; historical copies still require cleanup |
| Optional FVD preprocessing/statistics references | [fvd-comparison](https://github.com/universome/fvd-comparison), [StyleGAN-V](https://github.com/cvpr2022-stylegan-v/stylegan-v) | Provenance retained; applicable redistribution terms need confirmation before publication |
| KTH human action videos | [Official dataset](https://www.csc.kth.se/cvap/actions/) | Non-commercial use and requested publication reference; not bundled |
| KTH pretrained image codec | [Public model](https://huggingface.co/cvg-unibe/river_kth_64) | Model card GPL-3.0; not bundled; pinned SHA-256 in assets.json |
| Stochastic NSE data | [Zenodo record 10939479](https://zenodo.org/records/10939479) | Record states CC BY 4.0; data not bundled; exact file/schema verification pending |

The paper's FVD references and the currently implemented TorchScript backend
must be distinguished; see FVD.md. Reader-managed downloads do not by themselves
settle redistribution rights for copied evaluation source.

Local adaptations include explicit time embeddings, unfused attention for
second derivatives, strict inference-weight validation and a compatibility
loader for unused Lightning checkpoint metadata. The VQ network weight layout
is unchanged.

The NSE objective and Acrobot/GPE contributions also require the submitting
rights holder to confirm their provenance and right to redistribute; do not
infer permission solely because source files were accessible.

No project-wide license has been selected. See LICENSE_STATUS.md. Full upstream
license texts are preserved without anonymizing third-party copyright holders.
