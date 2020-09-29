## test opt



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

include("FID_helpers.jl")

include("../src/declarations.jl")

include("../src/retractions/Rp.jl")

include("../src/manifold/vector_transport.jl")

include("../src/optimization/CG.jl")
include("../src/optimization/Rp/engine_Rp.jl")
include("../src/optimization/TRS/trustregion.jl")
include("../src/optimization/TRS/trhelpers.jl")

include("../src/problems/RKHS_positive_coefficients.jl")

PyPlot.close("all")

Random.seed!(25)

fig_num = 1


fs = 800.0
N = 10000

## make oracle frequency positions.

ν_array = [210.0; 245.0; 260.0; 340.0; 355.0; 390.0]
Ω_array = ν_array .* (2*π)
L = length(ν_array)

λ = 1.26
λ_array = λ .* ones(L)

N_pairs = 3
α_values = [2.4; 0.8; 0.3]
α_array = parseα(α_values)

β_array = rand(length(ν_array)) .* (2*π)

N_vars = length(α_values) + length(β_array)
p_oracle = [α_values; β_array]

## visualize.
t = gettimerange(N, fs)
#t = gettimerange(N, 3*fs)

s = tt->im*sum( exp(-λ_array[l]*tt)*α_array[l]*exp(im*(Ω_array[l]*tt+β_array[l])) for l = 1:L )
s_t = s.(t)

DTFT_s = vv->computeDTFTch3eq29(s_t, vv, t)

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
DTFT_h = vv->computeDTFTch3eq29(h, vv, t_h)


##### old solve.

#𝓤 = LinRange(lower_bound_U, upper_bound_U, N_𝓤)
N_𝓤 = 200
𝓤 = LinRange(1.0, fs, N_𝓤)
𝓣 = t

DTFT_s_𝓤 = DTFT_s.(𝓤)
DTFT_h_evals = DTFT_h.(𝓤)
DTFT_hs_𝓤 = DTFT_s_𝓤 .* DTFT_h_evals

# ## TODO load set up parameters and oracle solution.
# #data = BSON.load("../data/density_fit_data1.bson")
#
# y = data[:y]
# X = data[:X]
# K = data[:K]
# μ = data[:μ]
#
# α_SDP = data[:α_SDP] # this is the oracle solution.

p0 = rand(N_vars)

f = aa->FIDcostfunc(aa, Ω_array,
        λ_array, N_pairs, 𝓤, DTFT_hs_𝓤, DTFT_h)

df_Euc = aa->ForwardDiff.gradient(f, aa)
#H = ForwardDiff.hessian(f, p0)
#fill!(H, 0.0) # force H to not be posdef. This triggers a Hessain approximnation in the engine.

## Euclidean metric.
g = pp->1.0

f(p0) # computes.
f(p_oracle) # makes sense. is practically zero.
df_Euc(p0) # not computing.

@assert 1==2

# ## prop. metric.
# g = pp->dot(pp,pp)

## inv. prop. metric.
# g = pp->1.0/(dot(pp,pp)+1.0)

# ## fancy inv. prop. metric.
# g = pp->1.0/(dot(pp,pp)+norm(pp)+1.0)

## initial guess.
p_initial = ones(Float64, length(α_values)+length(β_array))

## optimization configuration.
max_iter = 1600 #400
verbose_flag = false #  false
max_iter_tCG = 100
ρ_lower_acceptance = 0.2 # recommended to be less than 0.25
ρ_upper_acceptance = 5.0
TR_config = TrustRegionConfigType(  1e-3,
                                    10.0,
                                    max_iter_tCG,
                                    verbose_flag,
                                    ρ_lower_acceptance,
                                    ρ_upper_acceptance)
#
norm_df_tol = 1e-5
objective_tol = 1e-5
avg_Δf_tol = 0.0 #1e-12 #1e-5
avg_Δf_window = 10
max_idle_update_count = 50
𝑟 = 1e-2
opt_config = OptimizationConfigType( max_iter,
                                        verbose_flag,
                                        norm_df_tol,
                                        objective_tol,
                                        avg_Δf_tol,
                                        avg_Δf_window,
                                        max_idle_update_count,
                                        𝑟 )

# TODO get this retraction lower bound sorted out.
#retraction_lower_bound = 1e-10
#ℜ =  xx->ℝ₊₊arrayexpquadraticretraction(xx...; lower_bound = retraction_lower_bound)
@time α_star, f_α_array, norm_df_array,
        num_iters = engineRp(f,
                                        df_Euc,
                                        α_initial,
                                        copy(α_initial),
                                        TR_config,
                                        opt_config,
                                        H;
                                        𝑔 = g)
                                        #ℜ = ℜ)
#
discrepancy = norm(α_SDP-α_star)
println("discrepancy between another solver's solution and the RiemannianOptim solution: ", discrepancy)

println("f(α_SDP)  = ", f(α_SDP))
println("f(α_star) = ", f(α_star))
println()

# Visualize regression result.
PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(collect(1:num_iters), log.(f_α_array))

title_string = "log f(α) history vs. iterations"
PyPlot.title(title_string)
