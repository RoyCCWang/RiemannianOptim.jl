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

#include("../src/problems/FID_helpers.jl")
#include("../src/problems/FID_persist.jl")



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
include("../src/frontends/complex_lorentzian.jl")
include("../src/problems/complex_lorentzian_phase_helpers.jl")

PyPlot.close("all")

Random.seed!(25)

fig_num = 1

L = 11

Ωs = randn(L) .* 10000.0
λ = 7.32
αs = rand(L) .* 21000.0

β_oracle = projectcircle.( rand(L) .* (2*π) )
p_oracle = β_oracle

## visualize.
S = pp->evalcomplexLorentzian(pp, αs, β_oracle, λ, Ωs)

# eval.
N_viz = 1000
u_lower = minimum(Ωs./(2*π))*1.1
u_upper = maximum(Ωs./(2*π))*1.1
U = LinRange(u_lower, u_upper, N_viz)

S_U = S.(U)

PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(U, abs.(S_U), label = "abs.(S_U)")

PyPlot.legend()
PyPlot.xlabel("Hz")
PyPlot.ylabel("linear amplitude unit")
PyPlot.title("DTFT of refined candidates")

## test gradient.
f = bb->evalcLβcostfunc(bb, αs, λ, Ωs, U, S_U)
df = bb->evalcLβcostfuncgradient(bb, αs, λ, Ωs, U, S_U)

βs9, ∂𝓛_∂β_eval9, diff9 = setupcLβ𝓛(L, length(S_U), 1.0)
df9 = bb->evalcLβcostfuncgradient!(βs9, ∂𝓛_∂β_eval9, diff9,
                    bb,
                    αs,
                    λ,
                    Ωs,
                    U,
                    S_U)
df_AD = bb->ForwardDiff.gradient(f,bb)

b0 = randn(L)
df_b0 = df(b0)
df9_b0 = df9(b0)
df_AD_b0 = df_AD(b0)

#discrepancy = norm(df_b0-df_AD_b0)
discrepancy = norm(df9_b0-df_AD_b0)
println("discrepancy = ", discrepancy)
println()

##### solve.

# set up data.
N_𝓤 = 200
𝓤 = LinRange(u_lower, u_upper, N_𝓤)
S_𝓤 = S.(𝓤)


PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(U, abs.(S_U), label = "abs.(S_U)")
PyPlot.plot(𝓤, abs.(S_𝓤), "x")

PyPlot.legend()
PyPlot.xlabel("Hz")
PyPlot.ylabel("linear amplitude unit")
PyPlot.title("DTFT of optim test positions")


verbose_flag = false

# parameters.
max_iter = 500
β_initial = ones(L)
minimum_TR_radius = 1e-3

# solve.
@time p_star, f_p_array, norm_df_array,
        num_iters = solvecLβproblem( Ωs,
                        λ,
                        αs,
                        S_𝓤,
                        𝓤,
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
        num_iters = solvecLβproblem( Ωs,
                        λ,
                        αs,
                        S_𝓤,
                        𝓤,
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
        num_iters = solvecLβproblem( Ωs,
                        λ,
                        αs,
                        S_𝓤,
                        𝓤,
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
