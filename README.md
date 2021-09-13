# Hyperspectral Image Classification

⭐ [Welcome to my HomePage](https://lironui.github.io/) ⭐ 

This repository implementates 6 frameworks for hyperspectral image classification based on PyTorch and sklearn.

The detailed results can be seen in the [Classification of Hyperspectral Image Based on 
Double-Branch Dual-Attention Mechanism Network](https://www.mdpi.com/2072-4292/12/3/582).

Some of our code references the projects
* [Dual-Attention-Network](https://github.com/SH8899/Dual-Attention-Network.git)
* [Remote sensing image classification](https://github.com/stop68/Remote-Sensing-Image-Classification.git)
* [A Fast Dense Spectral-Spatial Convolution Network Framework for Hyperspectral Images Classification](https://github.com/shuguang-52/FDSSC.git) 

If our code is helpful to you, please cite
`Li R, Zheng S, Duan C, et al. Classification of Hyperspectral Image Based on Double-Branch Dual-Attention Mechanism Network[J]. Remote Sensing, 2020, 12(3): 582.`


Requirements：
------- 
```
numpy >= 1.16.5
PyTorch >= 1.3.1
sklearn >= 0.20.4
```

Datasets:
------- 
You can download the hyperspectral datasets in mat format at: http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes, and move the files to `./datasets` folder.

Usage:
------- 
1. Set the percentage of training and validation samples by the `load_dataset` function in the file `./global_module/generate_pic.py`.
2. Taking the DBDA framework as an example, run `./DBDA/main.py` and type the name of dataset. 
3. The classfication maps are obtained in `./DBDA/classification_maps` folder, and accuracy result is generated in `./DBDA/records` folder.

Network:
------- 
* [DBDA](https://www.mdpi.com/2072-4292/12/3/582)
* [DBMA](https://www.mdpi.com/2072-4292/11/11/1307/xml)
* [FDSSC](https://www.mdpi.com/2072-4292/10/7/1068/htm)
* [SSRN](https://ieeexplore.ieee.org/document/8061020)
* [CDCNN](https://ieeexplore.ieee.org/document/7973178)
* [SVM](https://ieeexplore.ieee.org/document/1323134/)

![network](https://github.com/lironui/Double-Branch-Dual-Attention-Mechanism-Network/blob/master/figures/Figure%206.%20The%20structure%20of%20the%20DBDA%20network.png)  
Figure 1. The structure of the DBDA network. The upper Spectral Branch composed of the dense 
spectral block and channel attention block is designed to capture spectral features. The lower Spatial 
Branch constituted by dense spatial block, and spatial attention block is designed to exploit spatial 
features. 

Results:
------- 
![IP](https://github.com/lironui/Double-Branch-Dual-Attention-Mechanism-Network/blob/master/figures/Figure%209.%20Classi%EF%AC%81cation%20maps%20for%20the%20IP%20dataset%20using%203%25%20training%20samples.png)
Figure 2. Classiﬁcation maps for the IP dataset using 3% training samples. (a) False-color image. (b) 
Ground-truth (GT). (c)–(h) The classiﬁcation maps with disparate algorithms. 
![UP](https://github.com/lironui/Double-Branch-Dual-Attention-Mechanism-Network/blob/master/figures/Figure%2010.%20Classi%EF%AC%81cation%20maps%20for%20the%20UP%20dataset%20using%200.5%25%20training%20samples.png)
Figure 3. Classiﬁcation maps for the UP dataset using 0.5% training samples. (a) False-color image. 
(b) Ground-truth (GT). (c)–(h) The classiﬁcation maps with disparate algorithms. 
![SV](https://github.com/lironui/Double-Branch-Dual-Attention-Mechanism-Network/blob/master/figures/Figure%2011.%20Classi%EF%AC%81cation%20maps%20for%20the%20SV%20dataset%20using%200.5%25%20training%20samples.png)
Figure 4. Classiﬁcation maps for the UP dataset using 0.5% training samples. (a) False-color image. 
(b) Ground-truth (GT). (c)–(h) The classiﬁcation maps with disparate algorithms. 
![BS](https://github.com/lironui/Double-Branch-Dual-Attention-Mechanism-Network/blob/master/figures/Figure%2012.%20Classi%EF%AC%81cation%20maps%20for%20the%20BS%20dataset%20using%201.2%25%20training%20samples.png)
Figure 5. Classiﬁcation maps for the BS dataset using 1.2% training samples. (a) False-color image. 
(b) Ground-truth (GT). (c)–(h) The classiﬁcation maps with disparate algorithms. 
