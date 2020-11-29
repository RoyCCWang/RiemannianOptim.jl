# p is the point.
function VectorTransportType(p, R::Function, X_in::Array{T,D}, Y_in::Array{T,D}) where {T<:Real, D}

    X = copy(X_in)
    Y = copy(Y_in)

    f = tt->R(p,X,Y,tt)
    v = zz->ForwardDiff.derivative(f,zz)
    return VectorTransportType(p,X,Y,v)
end

function evalvectortransport(θ::VectorTransportType{T,D,M},
                                p::M,
                                X::Array{T,D},
                                Y::Array{T,D})::Array{T,D} where {T,D,M}
    # update.
    copymanifoldpt!(θ.p, p)
    θ.X[:] = X
    θ.Y[:] = Y

    # evalute at t = 0.
    return θ.𝑉(zero(T))
end

function evalvectortransport(θ::VectorTransportType{T,D,M},
                                p::Array{T,D},
                                X::Array{T,D},
                                Y::Array{T,D})::Array{T,D} where {T,D,M}
    # update.
    θ.p[:] = p
    θ.X[:] = X
    θ.Y[:] = Y

    # evalute at t = 0.
    return θ.𝑉(zero(T))
end
