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

N_pairs = 6
L = 11

ξs = sort(rand(L) .* 10.0)
λs = rand(L) .* 14.0
αs = rand(L) .* 21000.0
κs = rand(L) .* 3100.0

#ξs = -ξs ./ κs
λs = λs ./ κs
λsc = λs[1] .* ones(L)

αs = sort(αs ./ κs, rev = true)
α_values_oracle = αs[1:N_pairs]
αs_oracle = parseα(α_values_oracle, L)

β_oracle = projectcircle.( rand(L) .* (2*π) )
#β_oracle = zeros(L)
p_oracle = [α_values_oracle; β_oracle]

println("[αs_oracle β_oracle λs ξs] = ")
display([αs_oracle β_oracle λs ξs])
println()


## visualize.
S = pp->evalFIDFT(pp, αs_oracle, β_oracle, λs, ξs)
Sc = pp->evalFIDFT(pp, αs_oracle, β_oracle, λsc, ξs)

# eval.
N_viz = 1000
P = LinRange(0, 10.0, N_viz)

S_P = S.(P)
Sc_P = Sc.(P)

PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(P, abs.(S_P), label = "abs.(S_P)")
PyPlot.plot(P, abs.(Sc_P), "--", label = "abs.(Sc_P)")

PyPlot.legend()
PyPlot.xlabel("Hz")
PyPlot.ylabel("linear amplitude unit")
PyPlot.title("DTFT of refined candidates")




##### solve.

# set up data.
N_𝓟 = 200
𝓟 = LinRange(0, 10, N_𝓟)
S_𝓟 = S.(𝓟)


##### fast gradient.


f = pp->evalFIDFTαβcostfunc(pp, λs, ξs, 𝓟, S_𝓟)
fslow = pp->evalFIDFTαβcostfuncslow(pp, λs, ξs, 𝓟, S_𝓟)

df_AD = aa->ForwardDiff.gradient(f, aa)
df_AN = pp->evalFIDFTαβcostfuncgradient(pp, λs, ξs, 𝓟, S_𝓟)


p_test = rand(L + N_pairs)

println("Timing df_AN:")
@time df_AN(p_test)
@time df_AN(p_test)
@time df_AN(p_test)
println()

println("Timing AD:")
@time df_AD(p_test)
@time df_AD(p_test)
@time df_AD(p_test)
println()

df_AD_p0 = df_AD(p_test)
df_AN_p0 = df_AN(p_test)
q = df_AD_p0[end-L+1:end]
discrepancy = norm(q-df_AN_p0)
println("discrepancy = ", discrepancy)
display([q df_AN_p0])
println()

S = pp->evalFIDFT(pp, αs_oracle, β_oracle, λs, ξs)
Sr_AN = pp->realFTFID(pp, αs_oracle, β_oracle, λs, ξs)

discrepancy = norm(real(S(4.5)) - Sr_AN(4.5))
println("real S: discrepancy = ", discrepancy)
println()

Si_AN = pp->imagFTFID(pp, αs_oracle, β_oracle, λs, ξs)
discrepancy = norm(imag(S(4.5)) - Si_AN(4.5))
println("imag S: discrepancy = ", discrepancy)
println()


fslow(p_test)
f(p_test)

@assert 1==2

verbose_flag = true

# parameters.
#max_iter = 5000
max_iter = 200 # 6 seconds. try AN gradient.
α_values_initial = sort(rand(N_pairs), rev = true )
β_initial = rand(L)
α_max = 500.0
#α_max = 10.0
minimum_TR_radius = 1e-3
ϵ_retraction = 1e-9 # for intervalretraction.

# solve.
@time p_star, f_p_array, norm_df_array,
        num_iters = solveFIDFTαβproblem( ξs,
                        λs,
                        S_𝓟,
                        𝓟,
                        α_values_initial,
                        β_initial,
                        α_max;
                        verbose_flag = verbose_flag,
                        max_iter = max_iter,
                        minimum_TR_radius = minimum_TR_radius,
                        ϵ_retraction = ϵ_retraction)


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
