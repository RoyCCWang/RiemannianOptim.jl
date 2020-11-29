

# 𝓤 is indexed by n.
function FIDcomputerϕ(   Ω_array::Vector{T},
                        λ_array::Vector{T},
                        α_array::Vector{T},
                        β_array::Vector{T},
                        H::Vector{Complex{T}},
                        l::Int,
                        t::T,
                        n::Int,
                        u::T)::Tuple{T,T} where T <: Real
    #
    @assert 1 <= l <= length(Ω_array)
    @assert 1 <= n <= length(H)

    C_l_t = im*exp(im*Ω_array[l]*t-λ_array[l]*t)
    B_l_t_u = H[n]*C_l_t

    r = abs(B_l_t_u)
    ϕ = -2*π*u*t + angle(B_l_t_u)

    return r, ϕ
end

function FIDcomputeC(   Ω_array::Vector{T},
            λ_array::Vector{T})::Matrix{Complex{T}} where T <: Real
    #
    L = length(Ω_array)
    @assert length(λ_array) == L

    C = Matrix{Complex{T}}(undef, L, length(𝓣))
    for i = 1:length(𝓣)
        for l = 1:L
            t = 𝓣[i]
            C[l,i] = im*exp(im*Ω_array[l]*t-λ_array[l]*t)
        end
    end

    return C
end

function FIDcomputerϕ(  Hu::Complex{T},
                        C_l_t::Complex{T},
                        t::T,
                        u::T)::Tuple{T,T} where T <: Real

    B_l_t_u = Hu*C_l_t

    r = abs(B_l_t_u)
    ϕ = -2*π*u*t + angle(B_l_t_u)

    return r, ϕ
end

# function FIDcostfuncgradient(   p::Vector{T2},
#                         Ω_array::Vector{T},
#                         λ_array::Vector{T},
#                         N_pairs::Int,
#                         𝓣,
#                         𝓤,
#                         DTFT_hs_𝓤::Vector{Complex{T}},
#                         DTFT_h_𝓤::Vector{Complex{T}},
#                         C::Matrix{Complex{T}})::Vector{T} where {T <: Real, T2 <: Real}
#
#     # parse.
#     α_values = p[1:N_pairs]
#     α_array = parseα(α_values, L)
#     β_array = p[N_pairs+1:end]
#
#     L = length(β_array)
#     @assert length(α_array) == L
#
#     # model.
#     f = tt->sum( evalFIDcomponent(tt, α_array[l],
#                     β_array[l],
#                     λ_array[l],
#                     Ω_array[l]) for l = 1:L )
#     f_t = f.(𝓣)
#
#     DTFT_f = vv->computeDTFTch3eq29AD(f_t, vv, t)
#
#     # allocate.
#     ∂𝓛_∂α = zeros(T, L)
#     ∂𝓛_∂β = zeros(T, L)
#
#     # compute gradient.
#     for n = 1:length(𝓤)
#
#         diff = DTFT_f(𝓤[n])*DTFT_h_𝓤[n] - DTFT_hs_𝓤[n]
#
#         diff_real = real(diff)
#         diff_imag = imag(diff)
#
#         for l = 1:L
#
#             sum_real = zero(T)
#             sum_imag = zero(T)
#
#             for i = 1:length(𝓣)
#
#                 t = 𝓣[i]
#
#                 r, ϕ = FIDcomputerϕ( DTFT_h_𝓤[n],
#                             C[l,i], t, 𝓤[n])
#
#                 #
#                 sum_real += r*cos(β_array[l]+ϕ)
#                 sum_imag += r*sin(β_array[l]+ϕ)
#             end
#
#             α_contribution = 2*diff_real*sum_real +
#                                 2*diff_imag*sum_imag
#             ∂𝓛_∂α[l] += α_contribution
#         end
#
#     end
#
#     return ∂𝓛_∂α
# end

# H is DTFT_h_𝓤.
# S is DTFT_hs_𝓤.
# slow due to explicit sum form.
function FIDcostfuncsumform(   p::Vector{T2},
                        Ω_array::Vector{T},
                        λ_array::Vector{T},
                        N_pairs::Int,
                        𝓣,
                        𝓤,
                        S::Vector{Complex{T}},
                        H::Vector{Complex{T}},
                        C::Matrix{Complex{T}})::T2 where {T <: Real, T2 <: Real}

    # parse.
    α_values = p[1:N_pairs]
    α_array = parseα(α_values, L)
    β_array = p[N_pairs+1:end]

    L = length(β_array)
    @assert length(α_array) == L

    term_real = zero(T2)
    term_imag = zero(T2)

    for n = 1:length(𝓤)

        F_u_real = zero(T2)
        F_u_imag = zero(T2)

        for i = 1:length(𝓣)
            t = 𝓣[i]

            for l = 1:L
                # parse.
                α = α_array[l]
                β = β_array[l]

                # get contants.
                # r, ϕ = FIDcomputerϕ(   Ω_array,
                #             λ_array,
                #             α_array,
                #             β_array,
                #             H,
                #             l, t, n, 𝓤[n])
                r, ϕ = FIDcomputerϕ( H[n],
                            C[l,i], t, 𝓤[n])
                #
                F_u_real += r*α*cos(β+ϕ)
                F_u_imag += r*α*sin(β+ϕ)
            end
        end

        term_real += (F_u_real-real(S[n]))^2
        term_imag += (F_u_imag-imag(S[n]))^2
    end

    return term_real + term_imag
end



#### try persist.

# even peaks.
function updateαβeven!(α_array, β_array::Vector{T}, p, N_pairs) where T <: Real

    α_values = p[1:N_pairs]
    α_array[:] = parseα(α_values, L)
    β_array[:] = p[N_pairs+1:end]

    return nothing
end

function setupmodeleven(L::Int, N_pairs,
                        Ω_array::Vector{T},
                        λ_array::Vector{T}) where T <: Real

    # allocate persistant buffers.
    α_array = Vector{T}(undef, L)
    β_array = Vector{T}(undef, L)

    f = tt->sum( evalFIDcomponent(tt, α_array[l],
                    β_array[l],
                    λ_array[l],
                    Ω_array[l]) for l = 1:L )

    updatef! = pp->updateαβeven!(α_array, β_array, pp, N_pairs)

    return f, updatef!
end

# about same speed as FIDcostfunc.
function FIDcostfuncpersist!(   f_𝓣::Vector{Complex{T}},
                            p::Vector{T2},
                            f,
                            updatef!,
                            DTFT_f,
                            𝓣,
                            𝓤,
                            DTFT_hs_𝓤::Vector{Complex{T}},
                            DTFT_h_𝓤::Vector{Complex{T}})::T2 where {T <: Real, T2 <: Real}
    resize!(f_𝓣, length(𝓣))

    # update FID model.
    updatef!(p)
    for i = 1:length(𝓣)
        f_𝓣[i] = f(𝓣[i])
    end

    # set up DTFT.
    #DTFT_f = vv->computeDTFTch3eq29AD(f_𝓣, vv, 𝓣)

    # compute score.
    score = zero(T)
    for n = 1:length(𝓤)

        LHS = DTFT_hs_𝓤[n]

        #RHS = DTFT_hf(𝓤[n])
        RHS = DTFT_f(𝓤[n]) * DTFT_h_𝓤[n]

        # corresponds to least-squares objective.
        score += abs2( LHS - RHS )
    end

    return score
end
