# Dimensional Reduction Strike: Robust Dynamic MRI Reconstruction with Spatiotemporal Diffusion Model
Accelerated dynamic magnetic resonance imaging (MRI) is highly expected in clinical applications. However, its reconstruction remains challenging due to the inherently high dimensionality and complexity of spatiotemporal information. Recent diffusion model has emerged as a robust reconstruction approach for spatial imaging, but their applications to spatiotemporal data remain underexplored. Therefore, it is essential to develop a new diffusion model tailored for dynamic MRI reconstruction, to better visualize the changes in organ physiological functions. 

![Method_dSTDM](https://github.com/wangziblake/STDM/blob/main/Figure/Method_dSTDM.png)

Here, we propose **a novel spatiotemporal diffusion model specifically designed for accelerated dynamic MRI**. Our main contributions are summarized as follows: 

1) A **flexible and robust** spatiotemporal diffusion model is proposed for dynamic MRI reconstruction. The learned spatiotemporal prior is agnostic to undersampling scenarios, allowing our method to **flexibly adapt to changes in reconstructions without re-training**.
2) By **analyzing the latent feature distributions** of different datasets, we find that the spatiotemporal diffusion shows greater potential than the existing spatial diffusion for **achieving robust reconstruction across diverse data**.
3) A **simple and effective** diffusion enhancement framework is proposed. It employs dual-directional orthogonal 2D models as a 3D spatiotemporal prior to **avoid visual misalignment and improve reconstruction performance**.
4) For both healthy cases and patients, the proposed method consistently provides state-of-the-art performance in **highly accelerated reconstruction** and shows remarkable robustness across **various undersampling scenarios and unseen data**, including patient data, real-time MRI data, non-Cartesian radial sampling, and different anatomies.

Given the **amazing adaptability and generalization capability**, we believe that our spatiotemporal diffusion model offers a promising direction for achieving more reliable dynamic MRI in diverse scenarios. Furthermore, this innovative approach holds great potential to be extended to the inverse problems in other medical modalities involving dynamic acquisitions.

![Analysis_dSTDM](https://github.com/wangziblake/STDM/blob/main/Figure/Analysis_dSTDM.png)

**This paper has been accepted by IEEE Transactions on Computational Imaging (2025) at https://doi.org/10.1109/TCI.2025.3598421**

**Email: Dr. Zi Wang (zi.wang@imperial.ac.uk); Dr. Xiaobo Qu (quxiaobo@xmu.edu.cn)**


## dSTDM (Dual-directional spatiotemporal diffusion model) framework (under construction)
The training and testing codes of dSTDM framework are released here.

Python environment should be: python=3.6.13, pytorch=1.10.1

**Note: The software is used for academic only, and cannot be used commercially.**


## Citation
If you want to use the code, please cite the following paper:

Zi Wang et al., Robust cardiac cine MRI reconstruction with spatiotemporal diffusion model, ***IEEE Transactions on Computational Imaging***, 11: 1258-1270, 2025.


## Acknowledgement
The authors thank Drs. Michael Lustig, Ricardo Otazo, Jo Schlemper, Dong Liang, Hyungjin Chung, Jong Chul Ye, and Yang Song for sharing their codes online. 


![Visitors](https://visitor-badge.laobi.icu/badge?page_id=<wangziblake>.<STDM>)
