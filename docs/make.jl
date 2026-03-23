using IBSpector
using Documenter

DocMeta.setdocmeta!(IBSpector, :DocTestSetup, :(using IBSpector); recursive=true)

makedocs(;
    modules=[IBSpector],
    authors="Tommaso Stentella <stentell@molgen.mpg.de> and contributors",
    sitename="IBSpector.jl",
    format=Documenter.HTML(;
        canonical="https://ArndtLab.github.io/IBSpector.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Diagnostics" => "diagnostics.md"
    ],
    warnonly=[:missing_docs],
)

deploydocs(;
    repo="github.com/ArndtLab/IBSpector.jl",
    devbranch="main",
    versions=["stable" => "v^", "v#.#"],
)
