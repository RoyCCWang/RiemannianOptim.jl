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

# # stress test.
# N_pairs = 55
# L = 110

## problematic case if no PSO.
N_pairs = 6
L = 11

# # odd L.
# N_pairs = 4
# L = 7
#
# # even L.
# N_pairs = 4
# L = 8

ξs = sort(rand(L) .* 10.0)
λs = rand(L) .* 14.0
#αs = rand(L) .* 21000.0 # trouble case. Need to crank up α_max.
αs = rand(L) .* 2100.0
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


##### speed up gradient.

f = pp->evalFIDFTαβcostfunc(pp, λs, ξs, 𝓟, S_𝓟)
fslow = pp->evalFIDFTαβcostfuncslow(pp, λs, ξs, 𝓟, S_𝓟)

df_AD = aa->ForwardDiff.gradient(f, aa)

# #no pre-allocation.
# df_AN = pp->evalFIDFTαβcostfuncgradient(pp, λs, ξs, 𝓟, S_𝓟)

# with pre-allocation. Not much faster than the non-preallocated version.
αs_persist, βs_persist, ∂𝓛_∂β_eval_persist, ∂𝓛_∂α_eval_persist,
        ∂𝓛_∂a_eval_persist, diff_persist = setupFTFIDαβ𝓛(N_pairs, L, length(S_𝓟), 1.0)

#
df_AN = pp->evalFIDFTαβcostfuncgradient!(αs_persist,
                βs_persist, ∂𝓛_∂β_eval_persist,
                ∂𝓛_∂α_eval_persist, ∂𝓛_∂a_eval_persist,
                diff_persist,
                pp, λs, ξs, 𝓟, S_𝓟)


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
#q = df_AD_p0[end-L+1:end]
q = df_AD_p0
discrepancy = norm(q-df_AN_p0)
println("discrepancy = ", discrepancy)
display([q df_AN_p0])
println()

#@assert 1==2



### retraction settings.
#α_max = 500.0
α_max = 10.0
ϵ_retraction = 1e-9 # for intervalretraction.

# initial guess.
α_values_initial = sort(rand(N_pairs), rev = true )
β_initial = rand(L)

# ## good.
# α_values_initial = [     0.9553781538862007;
#                          0.8638975162587343;
#                          0.8007100409944219;
#                          0.3528806798699531;
#                          0.3099816588294799;
#                          0.20800886134956031]
# β_initial = [ 0.7914787386761699;
#                  0.6130379854906192;
#                  0.9149529050312983;
#                  0.09375968676995772;
#                  0.46501712469348533;
#                  0.1913590317184708;
#                  0.3470474144198672;
#                  0.9648255311678162;
#                  0.6128816366256244;
#                  0.04504530111973937;
#                  0.22437554345914945]


## bad.
# α_values_initial = [    0.9173825816743866;
#                          0.7617216102778348;
#                          0.44667816154094053;
#                          0.2833106512893173;
#                          0.26719550327432295;
#                          0.14710489126440351]
# β_initial = [ 0.03274246945152992;
#                  0.05330532505475882;
#                  0.36859160530704904;
#                  0.899194597154837;
#                  0.8142417430874107;
#                  0.005779082024976345;
#                  0.16917199094824653;
#                  0.5187446766594683;
#                  0.00886028360812885;
#                  0.8146622549131075;
#                  0.05096339644377901]

import Optim

p_initial = [α_values_initial; β_initial]

#p_cost = p_star
p_cost = p_initial

max_iters_PSO = 1000
N_particles = 3
println("Timing: PSO")
@time p_star_PSO = solveFIDFTαβproblemPSO(ξs, λs, S_𝓟, 𝓟, α_values_initial,
                        β_initial;
                        max_iters_PSO = max_iters_PSO,
                        N_particles = N_particles,
                        ϵ_retraction = ϵ_retraction)

α_star_PSO = p_star_PSO[1:length(α_values_initial)]
β_star_PSO = p_star_PSO[end-length(β_initial)+1:end]


discrepancy = norm(p_oracle-p_star_PSO)
println("discrepancy between oracle and the final solution: ", discrepancy)
println("[p_oracle p_star_PSO]:")
display([p_oracle p_star_PSO])
println()




# Riemmian optim parameters.
verbose_flag = true
minimum_TR_radius = 1e-3

#max_iters_RMO = 5000
max_iters_RMO = 100


# solve.
@time p_star, f_p_array, norm_df_array,
        num_iters, f, df = solveFIDFTαβproblemRMO( ξs,
                        λs,
                        S_𝓟,
                        𝓟,
                        α_star_PSO,
                        β_star_PSO,
                        α_max;
                        verbose_flag = verbose_flag,
                        max_iters_RMO = max_iters_RMO,
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

println("f(p_star_PSO) = ", f(p_star_PSO))
println("f(p_star) = ", f(p_star))
println()




# TODO play with the riemannian metric for α and β.



# hybrid between PSO and Riemannian manifold optimization.
