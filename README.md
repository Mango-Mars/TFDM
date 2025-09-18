# Void Filling of Digital Elevation Models Based on Terrain Feature-Guided Diffusion Model

[![DOI](https://img.shields.io/badge/DOI-10.1016/j.rse.2024.114432-blue)](https://doi.org/10.1016/j.rse.2024.114432)

This repository contains the official implementation of the paper:

**Void filling of digital elevation models based on terrain feature-guided diffusion model**
*Remote Sensing of Environment (RSE), 2024*

---

## Introduction



Digital Elevation Models (DEMs) are essential for topographic and geomorphological studies. However, voids in DEMs cause loss of terrain details and reduce their applicability.

Traditional interpolation methods degrade in accuracy and struggle with complex terrain. GAN-based methods improve performance but are prone to artifacts and elevation jumps near voids.

We propose a **Terrain Feature-Guided Diffusion Model (TFDM)** for DEM void filling. By constraining the diffusion process with terrain feature lines, TFDM generates seamless DEM surfaces and preserves terrain contours.

---



## Training

Train the diffusion model with:

```bash
python train.py --conf_path confs/train.yml
```

---

## Testing & Evaluation

Run evaluation on test DEMs:

```bash
python test.py --conf_path confs/dem_eval.yml
```


---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{zhao2024void,
  title={Void filling of digital elevation models based on terrain feature-guided diffusion model},
  author={Zhao, Ji and Yuan, Yingying and Dong, Yuting and Li, Yaozu and Shao, Changliang and Yang, Haixia},
  journal={Remote Sensing of Environment},
  volume={309},
  pages={114432},
  year={2024},
  publisher={Elsevier}
}

```

---

## Acknowledgements

The authors would like to thank the editor, associate editor, and anonymous reviewers for their helpful comments and advice.

This work was supported by:

* National Natural Science Foundation of China (Grant No. 42171384)
* Key Laboratory of Urban Meteorology, China Meteorological Administration, China (Grant No. LUM-2024-10)
* Opening Fund of Key Laboratory of Geological Survey and Evaluation of Ministry of Education (Grant No. GLAB 2023ZR06)
* Fundamental Research Funds for the Central Universities, China University of Geosciences (Grant No. 2024XLB34)

Yuting Dong gratefully acknowledges the support of the Alexander von Humboldt Foundation.

---

