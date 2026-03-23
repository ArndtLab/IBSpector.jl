```@meta
CurrentModule = IBSpector
```

# IBSpector

Documentation for [IBSpector](https://github.com/ArndtLab/IBSpector.jl).

Module to run demographic inference on diploid genomes, under the assumption of panmixia.

## Data form of input and output

The genome needs to be SNP-called and the genomic distance between consecutive heterozygous positions needs to be computed. Heterozygous positions are the ones with genotype 0/1 or 1/0 (Note that the phase is not important). The input is then a vector containind such distances. Additionally, mutation and recombination rates need to be chosen and passed as input as well. See [Tutorial](@ref) for more details on preparing input data.

The demographic model underlying the inference is composed of a variable number of epochs and the population size is constant along each epoch.

The output is a vector of parameters in the form `[L, N0, T1, N1, T2, N2, ...]` where `L` is the total sequence length,
`N0` is the ancestral population size in the furthermost epoch and extending to the infinite past, the subsequent pairs $(T_i, N_i)$ are the duration and size of following epochs going from past to present. This format is referred to as `TN` vector throughout. The length `L` should match the input sequence length and is floating to improve the fit.

```@index
```

```@autodocs
Modules = [IBSpector, IBSpector.Spectra]
```
