# Diagnostics

Let's say that we inferred some demographic models and we have 
obtained the most probable as explained in the [Tutorial](@ref) 
```julia
results = demoinfer(segments, 1:8, mu, rho)
best = compare_models(results.fits)
```
First we can print
```julia
best.converged
```
which indicates whether the maximum likelihood optimization
converged. If that is not the case, we can inspect further
```julia
best.opt.optim_result.original
``` 
to get more details and decide whether this flag is correct.
In case the non convergence is true, a more greedy search might
be needed (see [`FitOptions`](@ref)) and larger number of iterations
and/or optimization time allowed.

We can also compute z-score residuals to assess goodness of fit
```
wth = wth = yth .* diff(h.edges[1])
resid = (h.weights .- wth) ./ sqrt.(wth)
```
Because the probabilistic model is Poisson, it might be that bins
in the tail have a skewed distribution of residuals, but the others
should closely follow a standard normal.
We can also assess the correlation structure of neighboring residuals
```julia
ps = IBSpector.residstructure(resid)
```
This function return a vector of right tail p values from t-tests for
correlation of neighbouring residuals (see the function doc).