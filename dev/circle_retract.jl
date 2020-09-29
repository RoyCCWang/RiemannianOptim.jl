import Calculus
import ForwardDiff

using LinearAlgebra

import PyPlot

import Random

include("../src/declarations.jl")

include("../src/retractions/Rp.jl")
include("../src/retractions/circular.jl")

include("../src/manifold/vector_transport.jl")

PyPlot.close("all")

#Random.seed!(250)
Random.seed!(25)

fig_num = 1

# a should be positive.



### create curve.
println("demo: circle retraction.")
p = 5/6*pi #rand()
X = abs(randn()) # force positive.
#X = -abs(randn()) # force negative.
t = randn()


f = tt->circleretraction(p,X,tt)
df_num = tt->Calculus.derivative(f,tt)
df = tt->ForwardDiff.derivative(f,tt)

println("ND: df(0.0) = ", df_num(0.0))
println("AD: df(0.0) = ", df(0.0))
println("X = ", X)
println("This should be zero if retraction is first order: abs(X-df(0)) = ", abs(X-df(0.0)))

df2 = tt->ForwardDiff.derivative(df,tt)
println("This should be zero is retraction is second order (using AD): df2(0.0) = ", df2(0.0))
println("End of demo.")
println()



Nv = 1000
#t_range = LinRange(0, π/multiplier, Nv)
t_range = LinRange(0, 50.0, Nv)
#t_range = LinRange(0, 10.0, Nv)

f_t = f.(t_range)

PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(t_range, f_t)
#PyPlot.plot(t_range, f_t, "x")

title_string = "retraction vs. t"
PyPlot.title(title_string)
