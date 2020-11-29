import Calculus
import ForwardDiff

using LinearAlgebra

import PyPlot

import Random

include("../src/declarations.jl")

include("../src/retractions/interval.jl")

include("../src/manifold/vector_transport.jl")

PyPlot.close("all")

#Random.seed!(250)
Random.seed!(25)

fig_num = 1

# a should be positive.



### create curve.
println("demo: interval retraction.")

u = -20.5
v = 34.2

#p = randn()
p = v - 1e-2

X = abs(randn()) # force positive.
#X = -abs(randn()) # force negative.
t = abs(randn())


f = tt->intervalretraction(p,X,tt, u, v)
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
t_range = LinRange(0, 5.0, Nv)
#t_range = LinRange(-50.0, 0.0, Nv)

f_t = f.(t_range)

PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(t_range, f_t)
#PyPlot.plot(t_range, f_t, "x")

title_string = "retraction vs. t"
PyPlot.title(title_string)





###### array version.
# array of positive numbers.
println("demo: lowersimplex retraction.")
N = 10
p = abs.(randn(N))
sort!(p, rev=true)
v1 = p[1] + 1.0

X = randn(N)
t = abs(randn())

h = tt->lowersimplexretraction(p, X, tt, v1)
dh_num = tt->Calculus.derivative(h,tt)
dh = tt->ForwardDiff.derivative(h,tt)

println("ND: dh(0.0) = ", dh_num(0.0))
println("AD: dh(0.0) = ", dh(0.0))
println("X = ", X)
println("This should be zero if retraction is first order: norm(X-dh(0)) = ", norm(X-dh(0.0)))

dh2 = tt->ForwardDiff.derivative(dh,tt)
println("This should be zero is retraction is second order (using AD): norm(dh2(0.0)) = ", norm(dh2(0.0)))
println("End of demo.")
println()

println("[p h(1.0) X] = ")
display([p h(1.0) X])
println()
