run run.sh for training and testing.

## Environment Setup

To set up the enviroment you can easily run the following command:
```
pip install -r requirements.txt
```

package requirements: selectors, aug_lib
The environment setting and parameter follows baseline method and PES method[1].

[1]Yingbin Bai, Erkun Yang, Bo Han, Yanhua Yang, Jiatong Li, Yinian Mao, Gang Niu, and Tongliang
Liu. Understanding and improving early stopping for learning with noisy labels. Advances in Neural Information Processing Systems, 2021

## Results

| Task      | #1      | #2      | #3      | #4      | #5      | Mean±std       |
|-----------|---------|---------|---------|---------|---------|----------------|
| c10_aggre | 95.6445 | 95.7910 | 95.7715 | 95.6738 | 95.5957 | **95.70±0.07** |
| c10_rand1 | 96.1524 | 95.7422 | 95.9668 | 95.9766 | 95.9961 | **95.97±0.13** |
| c10_rand2 | 95.7715 | 96.1621 | 96.0840 | 96.2012 | 96.1231 | **96.07±0.15** |
| c10_rand3 | 96.2305 | 96.2012 | 96.1328 | 96.2207 | 96.1328 | **96.18±0.04** |
| c10_worse | 94.4727 | 94.4238 | 94.5508 | 94.5800 | 94.3848 | **94.48±0.07** |
| c100      | 73.2813 | 73.2128 | 73.3008 | 73.3399 | 72.6563 | **73.16±0.25** |

If you failed run this code, please be free to email us(weichen.yu@cripac.ia.ac.cn, hongyuan.yu@cripac.ia.ac.cn).
