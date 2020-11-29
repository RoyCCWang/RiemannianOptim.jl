import Calculus
import ForwardDiff

using LinearAlgebra

import PyPlot

import Random

include("../src/declarations.jl")

include("../src/retractions/Rp.jl")

include("../src/manifold/vector_transport.jl")

#include("../src/VGP/objective.jl")

PyPlot.close("all")

#Random.seed!(250)
Random.seed!(25)

fig_num = 1

### create curve.
println("demo: ℝ_+ retraction.")
p = rand()
#X = abs(randn()) # force positive.
X = -abs(randn()) # force negative.
t = randn()

f = tt->ℝ₊₊expquadraticretraction(p,X,tt)
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


g = tt->f(tt)/p

Nv = 1000
#t_range = LinRange(lower_bound, upper_bound, Nv)
τ = 80.0
t_range = LinRange(-τ+p, τ+p, Nv)

f_t = f.(t_range)

println("peak value:")
max_val, ind = findmax(f_t)
println("peak value should be at t_peak = ", p/X)
println("t_range[ind] = ", t_range[ind])
println("ind = ", ind)
println()

println("peak value should be f(t_peak) = ", f(p/X))
println("max_val = ", max_val)
println()
println()



PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(t_range, f_t)
#PyPlot.plot(t_range, f_t, "x")

title_string = "retraction vs. t"
PyPlot.title(title_string)

## No matter the p and X, we've.
# julia> g(p/X)
# 1.648721270700128
#
# julia> g(-p/X)
# 0.22313016014842987


###### array version.
# array of positive numbers.
println("demo: ℝ_+ array retraction.")
N = 3
p = abs.(randn(N))
X = randn(N)
t = abs(randn())

h = tt->ℝ₊₊arrayexpquadraticretraction(p,X,tt)
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

# try computing vector transport.
Y = randn(N)
v = VectorTransportType(p, ℝ₊₊arrayexpquadraticretraction,X,Y)
out = evalvectortransport(v,p,X,2 .* X)
out_self = evalvectortransport(v,p,X,X)
