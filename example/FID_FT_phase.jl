## test opt

using BenchmarkTools

import Random
import PyPlot
import Printf


import Statistics
import SpecialFunctions

import Calculus
import ForwardDiff

import Optim

using LinearAlgebra

import BSON

include("../src/problems/FID_helpers.jl")
include("../src/problems/FID_persist.jl")

include("../src/problems/FID_phase_helpers.jl")

include("../src/declarations.jl")

include("../src/retractions/Rp.jl")
include("../src/retractions/circular.jl")
include("../src/retractions/compositions.jl")
include("../src/retractions/interval.jl")

include("../src/manifold/vector_transport.jl")

include("../src/optimization/CG.jl")
include("../src/optimization/vectorspace/engine_array.jl")
include("../src/optimization/TRS/trustregion.jl")
include("../src/optimization/TRS/trhelpers.jl")

include("../src/problems/RKHS_positive_coefficients.jl")

include("../src/frontends/RKHS.jl")
include("../src/frontends/FID_FT.jl")
include("../src/frontends/FID_DTFT.jl")

include("../src/frontends/RKHS.jl")
include("../src/frontends/FID_FT.jl")
include("../src/frontends/FID_DTFT.jl")

PyPlot.close("all")

Random.seed!(25)

fig_num = 1

L = 11

ξs = rand(L) .* -10000.0
λs = rand(L) .* 14.0
αs = rand(L) .* 21000.0
κs = rand(L) .* 3100.0

β_oracle = projectcircle.( rand(L) .* (2*π) )
p_oracle = β_oracle

## visualize.
S = pp->evalFIDFT(pp, αs, β_oracle, λs, κs, ξs)

# eval.
N_viz = 1000
P = LinRange(0, 10.0, N_viz)

S_P = S.(P)

PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(P, abs.(S_P), label = "abs.(S_P)")

PyPlot.legend()
PyPlot.xlabel("Hz")
PyPlot.ylabel("linear amplitude unit")
PyPlot.title("DTFT of refined candidates")

##### solve.

# set up data.
N_𝓟 = 200
𝓟 = LinRange(0, 10, N_𝓟)
S_𝓟 = S.(𝓟)

verbose_flag = false

# parameters.
max_iter = 500
β_initial = ones(L)
minimum_TR_radius = 1e-3

# solve.
@time p_star, f_p_array, norm_df_array,
        num_iters = solveFIDFTβproblem( ξs,
                        λs,
                        αs,
                        κs,
                        S_𝓟,
                        𝓟,
                        β_initial;
                        verbose_flag = verbose_flag,
                        max_iter = max_iter,
                        minimum_TR_radius = minimum_TR_radius)

#





max_iter = 500
β_initial = copy(p_star)
minimum_TR_radius = 1e-4

# solve.
@time p_star, f_p_array, norm_df_array,
        num_iters = solveFIDFTβproblem( ξs,
                        λs,
                        αs,
                        κs,
                        S_𝓟,
                        𝓟,
                        β_initial;
                        verbose_flag = verbose_flag,
                        max_iter = max_iter,
                        minimum_TR_radius = minimum_TR_radius)

#




max_iter = 500
β_initial = copy(p_star)
minimum_TR_radius = 1e-6

# solve.
@time p_star, f_p_array, norm_df_array,
        num_iters = solveFIDFTβproblem( ξs,
                        λs,
                        αs,
                        κs,
                        S_𝓟,
                        𝓟,
                        β_initial;
                        verbose_flag = verbose_flag,
                        max_iter = max_iter,
                        minimum_TR_radius = minimum_TR_radius)

#

discrepancy = norm(p_oracle-p_star)
println("discrepancy between oracle and the solution: ", discrepancy)
println("[p_oracle p_star]:")
display([p_oracle p_star])
println()


# Visualize regression result.
PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(collect(1:num_iters), log.(f_p_array))

title_string = "log f(p) history vs. iterations"
PyPlot.title(title_string)

# TODO full space search.
