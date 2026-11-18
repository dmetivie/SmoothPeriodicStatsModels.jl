using SmoothPeriodicStatsModels
# The fitting routines live in a package extension, loaded by these four packages.
using Optimization, OptimizationOptimJL, SimpleNonlinearSolve, ForwardDiff
using Documenter
using DocumenterLandingPage   # renders the hero and feature tiles of index.md
using DocumenterCodeBlocks
# DocumenterCodeBlocks 1.5.1 keys pages with `relpath`, whose `\` on Windows misses every page in a
# subfolder (no reference links, no continued line numbers there); use `/` as on other systems.
Sys.iswindows() && @eval DocumenterCodeBlocks page_key(doc, page) =
    replace(relpath(page.build, doc.user.build), Base.Filesystem.path_separator => "/")
using Literate
import Pkg

PROJECT_TOML = Pkg.TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))
PkgVERSION = PROJECT_TOML["version"]
NAME = PROJECT_TOML["name"]
AUTHORS = join(PROJECT_TOML["authors"], ", ") * " and contributors"
GITHUB = "https://github.com/dmetivie/SmoothPeriodicStatsModels.jl"

DocMeta.setdocmeta!(SmoothPeriodicStatsModels, :DocTestSetup, :(using SmoothPeriodicStatsModels); recursive = true)

const OptimizationExt = Base.get_extension(SmoothPeriodicStatsModels, :SmoothPeriodicStatsModelsOptimizationExt)

# Tutorials: the Literate scripts in `examples/`, rendered as pages of `docs/src/10-tutorials/`
const tutorials_jl = joinpath(dirname(@__DIR__), "examples")
const tutorials_md = joinpath(@__DIR__, "src", "10-tutorials")
rm(tutorials_md; force = true, recursive = true)
# Each tutorial numbers its code lines on from one block to the next (DocumenterCodeBlocks)
continued_line_numbers(md) = "```@codeblocks\nline_counter = :continue\n```\n\n" * md
for (i, file) in enumerate(["tuto_ar1.jl", "tuto_mixture.jl", "tuto_hmm.jl", "tuto_hmm_dists.jl", "tuto_genomics.jl"])
    Literate.markdown(joinpath(tutorials_jl, file), tutorials_md; mdstrings = true, name = "$i-$(splitext(file)[1])",
        postprocess = continued_line_numbers)
end

# Add titles of sections and overrides page titles
const titles = Dict(
    "10-tutorials" => "Tutorials",
    "20-guides" => "Guides",
)

function recursively_list_pages(folder; path_prefix="")
    pages_list = Any[]
    for file in readdir(folder)
        if file == "index.md"
            # We add index.md separately to make sure it is the first in the list
            continue
        end
        # this is the relative path according to our prefix, not @__DIR__, i.e., relative to `src`
        relpath = joinpath(path_prefix, file)
        # full path of the file
        fullpath = joinpath(folder, relpath)

        if isdir(fullpath)
            # If this is a folder, enter the recursion case
            subsection = recursively_list_pages(fullpath; path_prefix=relpath)

            # Ignore empty folders
            if length(subsection) > 0
                title = if haskey(titles, relpath)
                titles[relpath]
                else
                @error "Bad usage: '$relpath' does not have a title set. Fix in 'docs/make.jl'"
                relpath
                end
                push!(pages_list, title => subsection)
            end

            continue
        end

        if splitext(file)[2] != ".md" # non .md files are ignored
            continue
        elseif haskey(titles, relpath) # case 'title => path'
            push!(pages_list, titles[relpath] => relpath)
        else # case 'title'
            push!(pages_list, relpath)
        end
    end

    return pages_list
end

function list_pages()
    root_dir = joinpath(@__DIR__, "src")
    pages_list = recursively_list_pages(root_dir)

    return ["index.md"; pages_list]
end

makedocs(;
    modules = [SmoothPeriodicStatsModels, OptimizationExt],
    authors = "David Métivier <david.metivier@inrae.fr> and Caroline Cognot",
    repo = "https://github.com/dmetivie/SmoothPeriodicStatsModels.jl/blob/{commit}{path}#{line}",
    sitename = "SmoothPeriodicStatsModels.jl",
    format = 
        Documenter.HTML(; 
            canonical = "https://dmetivie.github.io/SmoothPeriodicStatsModels.jl",
            edit_link = "main",
            repolink = GITHUB,
            # the reference pages carry the docstring tooltips of DocumenterCodeBlocks
            size_threshold_warn = 200 * 2^10,
            size_threshold = 400 * 2^10,
            footer="[$NAME.jl]($GITHUB) v$PkgVERSION docs powered by [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl)."
        ),
    pages = list_pages(),
    plugins = [LandingPage(), CodeBlocks()],
)

deploydocs(; repo = "github.com/dmetivie/SmoothPeriodicStatsModels.jl", devbranch = "main")
