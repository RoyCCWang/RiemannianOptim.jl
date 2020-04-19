# test retractions for Σ.


import Random
import PyPlot




import Calculus
import ForwardDiff

using LinearAlgebra

include("../src/declarations.jl")



include("../src/retractions/Rp.jl")

include("../src/manifold/vector_transport.jl")

include("../src/VGP/objective.jl")

PyPlot.close("all")

Random.seed!(25)

fig_num = 1


println("demo: MVN retraction.")
n = 3
S = randn(n,n)
S = S'*S
m = randn(n)

h = convertMVNtoarray(m,S)
μ, Σ = convertarraytoMVN(h,n)

p = MVNType(m,S)
X = randn(length(h))
zero_pt = zeros(length(h))

f = tt->MVNretraction(p,zero_pt,X,tt)
df = tt->ForwardDiff.derivative(f,tt)
#df = tt->Flux.Tracker.derivative(f,tt)

println("AD: df(0.0) = ", df(0.0))
println("X = ", X)
println("This should be zero if retraction is first order: norm(X-df(0)) = ", norm(X-df(0.0)))

df2 = tt->ForwardDiff.derivative(df,tt)
#df2 = tt->Flux.Tracker.derivative(df,tt)
println("This should be zero is retraction is second order (using AD): norm(df2(0.0)) = ", norm(df2(0.0)))
println("End of demo.")
println()



println("Demo AD on objective function.")
#import ReverseDiff
g = xx->simpleMVNobjectivefunc(xx,m,S)
dg = xx->ForwardDiff.gradient(g,xx)
#dg = xx->Flux.Tracker.gradient(g,xx)[1]
dg_num = xx->Calculus.gradient(g,xx)

x0 = randn(length(h))
@time ans_AD = dg(x0)
#@time ans_AD_R = dg_R(x0)
@time ans_ND = dg_num(x0)

#println("AD R gave: ", ans_AD_R)
println("AD gave: ", ans_AD)
println("ND gave: ", ans_ND)
println("discrepancy: ", norm(ans_AD-ans_ND))
println("ForwardDiff is very slow for large dimensions. Use ND.")
println("End of demo.")

p0 = MVNType(convertarraytoMVN(x0,n)...)
g(p0)
g(x0) # this works.

#@assert 1==2

### create curve.
println("demo: ℝ_+ retraction.")
p = randn()
X = randn()
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



# array of positive numbers.
println("demo: ℝ_+ array retraction.")
N = 3
p = randn(N)
X = randn(N)
t = randn()

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

@assert 1==2
#
#
# ### fixed rank retraction.
# println("demo: fixed rank retraction.")
#
#
# Nr = 3
# Nc = 2
# p = randn(Nr,Nc)
# X = randn(Nr,Nc)
# t = randn()
#
# #g = tt->fixedrankretraction(p,X,tt)
# g = tt->fixedrankretractiontall(p,X,tt)
# dg = tt->ForwardDiff.derivative(g,tt)
# dg_num = tt->Calculus.derivative(g,tt)
#
# g(2.0)
#
# println("ND: dg(0.0) = ", dg_num(0.0))
# println("AD: dg(0.0) = ", dg(0.0))
# println("X = ", X)
# println("This should be zero if retraction is first order: norm(X-dg(0)) = ", norm(X-dg(0.0)))
#
# dg2 = tt->ForwardDiff.derivative(dg,tt)
# println("This should be zero is retraction is second order (using AD): norm(dg2(0.0)) = ", norm(dg2(0.0)))
# println("End of demo.")
# println()
#
#
# # try computing vector transport.
# Y = randn(Nr,Nc)
# v_stiefel = VectorTransportType(p, fixedrankretractiontall, X, Y)
# out_stiefel = evalvectortransport(v_stiefel, p, X.*t, X)


### fixed rank retraction.
println("demo: composite retraction.")

Nr = 3
Nc = 2
p = Rank1PositiveDiagmType(rand(Nr),randn(Nr,Nc))
X = randn(Nr,Nc+1)
zero_vec = zeros(Nr,Nc+1)
t = randn()

g = tt->rank1positivediagmretraction(p,zero_vec,X,tt)
dg = tt->ForwardDiff.derivative(g,tt)
#dg_num = tt->Calculus.derivative(g,tt)

g(0.0)
#

#println("ND: dg(0) = ", dg_num(0.0))
println("AD: dg(0) = ", dg(0.0))
println("X = ", X)
println("This should be zero if retraction is first order: norm(X-dg(0)) = ", norm(X-dg(0.0)))

dg2 = tt->ForwardDiff.derivative(dg,tt)
println("This should be zero is retraction is second order (using AD): norm(dg2(0.0)) = ", norm(dg2(0.0)))
println("End of demo.")
println()


# try computing vector transport.
Y = randn(Nr,Nc+1)
v_comp = VectorTransportSimpleType(p, rank1positivediagmretraction, X, Y)
out_comp = evalvectortransport(v_comp, p, X.*t, X)


# paper idea: use a faster decomp instead of svd for the fixed-rank retraction.
#   see. https://arxiv.org/pdf/0909.4061.pdf

# paper idea: diagonal + low-rank updates: Google matrix approx. covmat matrix approx.
# paper idea: can we get Non-negative matrix from "chaining" or push-foward/pull-back
#   of the positive and svd retractions? would they retain 2nd order conditions?
