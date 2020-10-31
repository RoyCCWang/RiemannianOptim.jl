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

# stress test.
N_pairs = 55
L = 110

# ## problematic case if no PSO.
# N_pairs = 6
# L = 11


# N_pairs = 6
# L = 12

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




### retraction settings.
α_max = maximum(α_values_oracle) .* 1.2
ϵ_retraction = 1e-9 # for intervalretraction.

# initial guess.
α_values_initial = sort(rand(N_pairs), rev = true )
β_initial = rand(L)

p_initial = [α_values_initial; β_initial]

## epoch parameters.
# N_epochs = 3
#
# max_iters_PSO = [500; 300; 100]
# N_particles = [3; 3; 3]
#
# # Riemmian optim parameters.
# verbose_flag = false
# minimum_TR_radius = [1e-3; 1e-4; 1e-5]
#
# max_iters_RMO = [100; 100; 100]

N_epochs = 30 #2

max_iters_PSO = 300 .* ones(Int, N_epochs)
N_particles = 6 .* ones(Int, N_epochs)

# Riemmian optim parameters.
verbose_flag = false
minimum_TR_radius = 1e-4 .* ones(Float64, N_epochs)
max_iters_RMO = 200 .* ones(Int, N_epochs)

# solve.
println("Timing:")
@time p_star, p_star_PSOs, p_star_RMOs, f,
        df = solveFIDFTαβproblemhybrid( ξs,
                        λs,
                        S_𝓟,
                        𝓟,
                        α_values_initial,
                        β_initial,
                        α_max;
                        minimum_TR_radius = minimum_TR_radius,
                        iters_RMOs = max_iters_RMO,
                        N_particles = N_particles,
                        iters_PSOs = max_iters_PSO,
                        verbose_flag = verbose_flag,
                        ϵ_retraction = ϵ_retraction)


discrepancy = norm(p_oracle-p_star)
println("discrepancy between oracle and the solution: ", discrepancy)
println("[p_oracle p_star]:")
display([p_oracle p_star])
println()

f_PSO = f.(p_star_PSOs)
f_RMO = f.(p_star_RMOs)

# Visualize regression result.
PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(collect(1:N_epochs), log.(f_PSO), "x", label = "PSO")
PyPlot.plot(collect(1:N_epochs), log.(f_RMO), "d", label = "RMO")

title_string = "log f(p) vs. epoch index"
PyPlot.title(title_string)
PyPlot.legend()

println("f(p_star_PSO) = ", f_PSO)
println("f(p_star) = ", f_RMO)
println("f(p_initial) = ", f(p_initial))
println("f(p_star) = ", f(p_star))
println()




# TODO play with the riemannian metric for α and β.
