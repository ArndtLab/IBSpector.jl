# Tutorial

To run the package, first install julia ([here](https://julialang.org/downloads/)).
To create a local environment with the package `cd` into your work directory and 
launch julia, then install the package (and other useful packages to handle the
data):
```julia
using Pkg; Pkg.activate(".")
```
```julia
Pkg.Registry.add(RegistrySpec(url = "https://github.com/ArndtLab/JuliaRegistry.git"))
```
```julia
Pkg.add("IBSpector","HistogramBinnings","CSV","DataFrames","DataFramesMeta")
```
You can now load the installed packages:
```julia
using IBSpector, HistogramBinnings, CSV, DataFrames, DataFramesMeta
```

## Preparing input data
For example, suppose you have a `.vcf` file with called variants you want to analyze. Then, in the most basic case, you may compute distances between heterozygous SNPs as follows:
```julia
f = "/myproject/myfavouritespecies.vcf"
df = CSV.read(f, DataFrame, 
    delim='\t', 
    comment="##",
    missingstring=[".", "NaN"],
    normalizenames=true,
    ntasks = 1,
    drop = [:INFO, :ID, :FILTER],
)

# remove homozygous variants
@chain df begin
    @rsubset! (:SampleName[1] == '1' && :SampleName[3] == '0') || (:SampleName[1] == '0' && :SampleName[3] == '1')
end

ils = df.POS[2:end] .- df.POS[1:end-1]
@assert all(ils .> 0)
```
Now we have a vector of intervals `ils`.

## Fitting and comparing demographic models
The tool require three inputs: a vector of IBS segments lengths, a mutation rate and 
a recombination rate (both per bp per generation). Additionally, we need to choose
a range of demographic models with epochs of piecewise constant effective size.

In the simplest use case we can just call
```julia
mu = 1e-8
rho = 1e-8
results = demoinfer(ils, 1:8, mu, rho)
```
and the 8 models will be saved in `results.fits`.
Then we can obtain the most probable model in the set with
```julia
best = compare_models(results.fits)
```

### More advanced options
First IBS spectrum is obtained as an histogram and the binning can be 
controlled by the function `adapt_histogram` and it keyword arguments
```julia
h = adapt_histogram(ils)
mu = 1e-8
rho = 1e-8
```
Then we set up the `FitOptions` object that contains several parameters for the optimization.
We stick with default values and only initialize with the three required inputs:
```julia
fop = FitOptions(sum(ils), length(ils), mu, rho)
```
And we fit 8 different model, with a number of epochs in the range 1 to 8:
```julia
results = demoinfer(h_obs, 1:8, fop)
```
The fitted models can be accessed with `results.fits`.

See [Diagnostics](@ref) to inspect the result and assess goodness of fit and optimization convergence.