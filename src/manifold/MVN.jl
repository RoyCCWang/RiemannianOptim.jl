### copy, add/subtract operations on the parameters of the MVN.
# WIP.

function copymanifoldpt!(x::Rank1PositiveDiagmType{T},
                         c::Vector{T},
                         A::Matrix{T}) where T
    ####
    x.c[:] = c
    x.A[:] = A

    return nothing
end

# x is destination, y is source.
function copymanifoldpt!(x::Rank1PositiveDiagmType{T},
                         y::Rank1PositiveDiagmType{T}) where T
    ####
    x.c[:] = y.c
    x.A[:] = y.A

    return nothing
end

function copymanifoldpt!(x::MVNType{T},
                         y::MVNType{T}) where T
    ####
    x.μ[:] = y.μ
    x.L.c[:] = y.L.c
    x.L.a[:] = y.L.a

    return nothing
end

function copymanifoldpt(y::MVNType{T}) where T
    x = MVNType(similar(y.μ),CholeskyType(y.L.c,y.L.a))
    return x
end



function packCholeskyType(L)
    c = diag(L)

    n = length(c)
    a = Vector{Float64}(undef, round(Int, n*(n+1)/2-n))

    p = 1
    for j = 1:n
        for i = j+1:n
            a[p] = L[i,j]
            p += 1
        end
    end
    @assert p-1 == round(Int, n*(n+1)/2-n) # debug.

    return CholeskyType(c,a)
end

function parseCholeskyType(c,a)
    L = diagm(0=>c)

    n = length(c)
    p = 1
    for j = 1:n
        for i = j+1:n
            L[i,j] = a[p]
            p += 1
        end
    end
    @assert p-1 == round(Int, n*(n+1)/2-n) # debug.

    return L
end

function convertMVNtoarray( m::Vector{T},
                            S::Matrix{T})::Vector{T} where T
    n = length(m)
    @assert size(S) == (n,n)

    h = Vector{T}(undef, round(Int, n + n*(n+1)/2) )
    h[1:n] = m

    C = cholesky(S)
    L = C.L

    h[n+1:n+n] = diag(L)

    p = n+n+1
    for j = 1:n
        for i = j+1:n
            h[p] = L[i,j]
            p += 1
        end
    end
    @assert p-1 == length(h)

    return h
end

function convertarraytoMVN( h::Vector{T},
                            n::Int)::Tuple{Vector{T},Matrix{T}} where T

    m, L = convertarraytoMVNcholesky(h, n)
    return m, L*L'
end

function convertarraytoMVNcholesky( h::Vector{T},
                            n::Int)::Tuple{Vector{T},Matrix{T}} where T
    # check.
    @assert length(h) == round(Int, n + n*(n+1)/2)


    m = h[1:n]

    L = diagm( 0=> h[n+1:n+n])

    p = n+n+1
    for j = 1:n
        for i = j+1:n
            L[i,j] = h[p]
            p += 1
        end
    end
    @assert p-1 == length(h)

    return m, L
end


function convertMVNtoarray( p::MVNType{T} )::Vector{T} where T
    n = length(p.μ)

    h = [p.μ; p.L.c; p.L.a]
    @assert length(h) == round(Int, n + n*(n+1)/2)

    return h
end
