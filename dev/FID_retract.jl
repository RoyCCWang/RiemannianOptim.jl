import Calculus
import ForwardDiff

using LinearAlgebra

import PyPlot

import Random

include("../src/declarations.jl")

include("../src/retractions/Rp.jl")
include("../src/retractions/circular.jl")
include("../src/retractions/compositions.jl")
include("../src/retractions/interval.jl")


include("../src/manifold/vector_transport.jl")

PyPlot.close("all")

#Random.seed!(250)
Random.seed!(25)

fig_num = 1

# a should be positive.



### create curve.
println("demo: FID retraction.")

N_pairs = 3
α_max = 30.0
α_values = [2.4; 0.8; 0.3]
β_array = collect( projectcircle(rand()*(2*π)) for i = 1:(2*length(α_values)) )

p = [α_values; β_array]
X = randn(length(p))
t = abs(randn())


f = tt->FIDnDretraction(p,X,tt, N_pairs, α_max)

f_array = collect(tt->f(tt)[i] for i = 1:length(p))
df_array_ND = tt->collect(Calculus.derivative(f_array[i], tt) for i = 1:length(p))
df_array_AD = tt->collect(ForwardDiff.derivative(f_array[i], tt) for i = 1:length(p))

println("ND: df(0.0) = ", df_array_ND(0.0))
println("AD: df(0.0) = ", df_array_AD(0.0))
println("X = ", X)
println("This should be zero if retraction is first order: abs(X-df(0)) = ", norm(X-df_array_AD(0.0)))

# df2 = tt->ForwardDiff.derivative(df,tt)
# println("This should be zero is retraction is second order (using AD): df2(0.0) = ", df2(0.0))
# println("End of demo.")
# println()
