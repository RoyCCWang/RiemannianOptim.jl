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

PyPlot.close("all")

Random.seed!(25)

fig_num = 1


fs = 800.0
N = 10000

## make oracle frequency positions.

# even example.
ν_array = [210.0; 245.0; 260.0; 340.0; 355.0; 390.0]

# odd example.
#ν_array = [210.0; 245.0; 300.0; 355.0; 390.0]

Ω_array = ν_array .* (2*π)
L = length(ν_array)

λ = 1.26
λ_array = λ .* ones(L)


N_pairs = 3
α_values = [2.4; 0.8; 0.3]
α_array = parseα(α_values, L)

β_array = projectcircle.( rand(length(ν_array)) .* (2*π) )

N_vars = length(α_values) + length(β_array)
p_oracle = copy(β_array)

## visualize.
t = gettimerange(N, fs)
#t = gettimerange(N, 3*fs)

s = tt->im*sum( exp(-λ_array[l]*tt)*α_array[l]*exp(im*(Ω_array[l]*tt+β_array[l])) for l = 1:L )
s_t = s.(t)

DTFT_s = vv->computeDTFTch3eq29AD(s_t, vv, t)

# eval.
N_viz = length(s_t)*2
u_range = LinRange(0, fs, N_viz)

DTFT_s_u = DTFT_s.(u_range)



PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(u_range, abs.(DTFT_s_u), label = "abs DTFT_s_u")

PyPlot.legend()
PyPlot.xlabel("Hz")
PyPlot.ylabel("linear amplitude unit")
PyPlot.title("DTFT of refined candidates")

#####
#### band-pass filter.

bp_a = 167.0
bp_b = 444.0
bp_δ = 20.0
N_h = 2001

h_func = xx->cosinetransitionbandpassimpulsefunc(xx, bp_a, bp_b, bp_δ)
t_h = gettimerangetunablefilter(N_h, fs)
h = h_func.(t_h)
DTFT_h = vv->computeDTFTch3eq29AD(h, vv, t_h)


##### old solve.

#𝓤 = LinRange(lower_bound_U, upper_bound_U, N_𝓤)
N_𝓤 = 200
𝓤 = LinRange(1.0, fs, N_𝓤)
𝓣 = t

DTFT_s_𝓤 = DTFT_s.(𝓤)
DTFT_h_𝓤 = DTFT_h.(𝓤)
DTFT_hs_𝓤 = DTFT_s_𝓤 .* DTFT_h_𝓤

β_initial = ones(L)

@time p_star, f_p_array, norm_df_array,
        num_iters = solveFIDDTFTβproblem(Ω_array,
                    λ_array,
                    α_array,
                    DTFT_s_𝓤,
                    DTFT_h_𝓤,
                    𝓣,
                    𝓤,
                    β_initial;
                    verbose_flag = true)

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
