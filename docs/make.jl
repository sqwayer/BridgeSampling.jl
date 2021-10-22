using Documenter
using BridgeSampling

push!(LOAD_PATH,"../src/")
makedocs(
    sitename = "BridgeSampling.jl Documentation",
    pages = [
        "Index" => "index.md",
        "Basic usage" => "basic.md",
    ],
    format = Documenter.HTML(prettyurls = false),
    modules = [BridgeSampling]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/sqwayer/BridgeSampling.jl.git"
)
