[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# A Polyhedral Study on Fuel-Constrained Unit Commitment

This repository includes the data set and code used in the computational experiments of the paper: [A Polyhedral Study on Fuel-Constrained Unit Commitment](https://doi.org/10.1287/ijoc.2022.1235), which is co-authored by Kai Pan, Ming Zhao, Chung-Lun Li, and Feng Qiu.

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

<!---
# CacheTest

The software and data in this repository are a snapshot of the software and data
that were used in the research reported on in the paper 
[This is a Template](https://doi.org/10.1287/ijoc.2021.0934) by T. Ralphs. 
The snapshot is based on 
[this SHA](https://github.com/tkralphs/JoCTemplate/commit/f7f30c63adbcb0811e5a133e1def696b74f3ba15) 
in the development repository. 

**Important: This code is being developed on an on-going basis at 
https://github.com/tkralphs/JoCTemplate. Please go there if you would like to
get a more recent version or would like support**
--->

## Cite

To cite this software, please cite the [paper](https://doi.org/10.1287/ijoc.2022.1235) using its DOI and the software itself, using the following DOI: [10.1287/ijoc.2022.1235.cd](https://doi.org/10.1287/ijoc.2022.1235.cd)  

Below is the BibTex for citing this version of the code.

```
@article{kaipanuc2022,
  author =        {Pan, Kai and Zhao, Ming and Li, Chung-Lun and Qiu, Feng},
  publisher =     {INFORMS Journal on Computing},
  title =         {A Polyhedral Study on Fuel-Constrained Unit Commitment},
  year =          {2022},
  doi =           {10.1287/ijoc.2022.1235.cd},
  url =           {https://github.com/INFORMSJoC/2021.0219},
}  
```

<!---
## Description

The goal of this software is to demonstrate the effect of cache optimization.

## Building

In Linux, to build the version that multiplies all elements of a vector by a
constant (used to obtain the results in [Figure 1](results/mult-test.png) in the
paper), stepping K elements at a time, execute the following commands.

```
make mult
```

Alternatively, to build the version that sums the elements of a vector (used
to obtain the results [Figure 2](results/sum-test.png) in the paper), stepping K
elements at a time, do the following.

```
make clean
make sum
```

Be sure to make clean before building a different version of the code.

## Results

Figure 1 in the paper shows the results of the multiplication test with different
values of K using `gcc` 7.5 on an Ubuntu Linux box.

![Figure 1](results/mult-test.png)

Figure 2 in the paper shows the results of the sum test with different
values of K using `gcc` 7.5 on an Ubuntu Linux box.

![Figure 1](results/sum-test.png)

## Replicating

To replicate the results in [Figure 1](results/mult-test), do either

```
make mult-test
```
or
```
python test.py mult
```
To replicate the results in [Figure 2](results/sum-test), do either

```
make sum-test
```
or
```
python test.py sum
```

## Ongoing Development

This code is being developed on an on-going basis at the author's
[Github site](https://github.com/tkralphs/JoCTemplate).

## Support

For support in using this software, submit an
[issue](https://github.com/tkralphs/JoCTemplate/issues/new).
--->
