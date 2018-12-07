This directory contains scripts used for experiments on the theoretical model presented in
the following paper:

```
Abdullah Al-Dujaili, Tom Schmiedlechner, Erik Hemberg, Una-May O'Reilly, “Towards distributed coevolutionary GANs,” AAAI 2018 Fall Symposium, 2018.
```

To re-run the experiments: 

```
$ source activate lipizzaner
$ python gaussian_gan.py # this will generate csv files in `experiment_results`
$ python gaussian_gan_plot.py # this will generate pdf plots in `plots`
```

To visualize the interval generator and the discriminator solutions with respect to the objective function (i.e., Fig. 3 (b) & (c)), you can refer to the demo in `opt_disc.py`

```
$ python opt_disc.py
```


