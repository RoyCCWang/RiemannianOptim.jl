
function computeDTFTch3eq29AD(h::Vector{Complex{T}}, u, Λ)::Complex{T} where T <: Real

    running_sum = zero(T)
    for i = 1:length(Λ)
        x = Λ[i]

        running_sum += h[i]*exp(-im*2*π*u*x)
    end

    return running_sum
end

# TODO make separate versions for auto-diff wrt α and β and t.
function evalFIDcomponent(t, α, β::T, λ, Ω)::Complex{T} where T <: Real

    if t < zero(T)
        return zero(T)
    end

    term1 = im*Ω*t
    term2 = -λ*t

    return im*α*exp(term1+term2)*exp(im*β)
    #return α*exp(term1+term2)*exp(im*β)
end

function FIDcostfunc(   p::Vector{T2},
                        Ω_array::Vector{T},
                        λ_array::Vector{T},
                        N_pairs::Int,
                        𝓣,
                        𝓤,
                        DTFT_hs_𝓤::Vector{Complex{T}},
                        DTFT_h_𝓤::Vector{Complex{T}})::T2 where {T <: Real, T2 <: Real}

    L = length(Ω_array)

    # parse.
    α_values = p[1:N_pairs]
    α_array = parseα(α_values, L)
    β_array = p[N_pairs+1:end]

    @assert length(α_array) == L == length(β_array) == length(λ_array)

    # model.
    f = tt->sum( evalFIDcomponent(tt, α_array[l],
                    β_array[l],
                    λ_array[l],
                    Ω_array[l]) for l = 1:L )
    f_𝓣 = f.(𝓣)

    DTFT_f = vv->computeDTFTch3eq29AD(f_𝓣, vv, 𝓣)
    #DTFT_hf = vv->(DTFT_f(vv) * DTFT_h(vv))

    # data.
    #DTFT_hs = vv->(DTFT_s(vv) * DTFT_h(vv))

    score = zero(T2)
    for n = 1:length(𝓤)

        LHS = DTFT_hs_𝓤[n]

        #RHS = DTFT_hf(𝓤[n])
        RHS = DTFT_f(𝓤[n]) * DTFT_h_𝓤[n]

        # corresponds to least-squares objective.
        score += abs2( LHS - RHS )
    end

    return score
end

function parseα( α_values::Vector{T}, L::Int) where T <: Real


    α_array = Vector{T}(undef, length(α_values)*2)
    parseα!(α_array, α_values, L)

    return α_array
end

function parseα!( α_array::Vector{T},
                    α_values::Vector{T}, L::Int) where T <: Real

    N_pairs = length(α_values)
    @assert 2*N_pairs == L || 2*N_pairs-1 == L

    if 2*N_pairs > L
        # odd number of peaks.
        resize!(α_array, 2*N_pairs-1)
    else
        # even number of peaks.
        resize!(α_array, 2*N_pairs)
    end

    for i = 1:N_pairs
        α_array[i] = α_values[end-i+1]
        α_array[end-i+1] = α_values[end-i+1]
    end

    return nothing
end


#### the rest of this file can probably be
#       discarded after making s_t and h_t
#       data variables for the example.
function gettimerange(N::Int, fs::T) where T
   Ts::T = 1/fs

   return zero(T):Ts:(N-1)*Ts
end

# # case 1D.
# function computeDTFTch3eq29AD(h, u::T, Λ)::Complex{T} where T <: Real
#
#     running_sum = zero(T)
#     for i = 1:length(Λ)
#         x = Λ[i]
#
#         running_sum += h[i]*exp(-im*2*π*u*x)
#     end
#
#     return running_sum
# end

function cosinetransitionbandpassimpulsefunc(x::T, a, b, δ)::Complex{T} where T <: Real

    term1 = evalDTFTintegralrising(x, a-δ, a+δ, a-δ, a+δ)
    term2 = evalDTFTintegralrect(x, a+δ, b-δ)
    term3 = evalDTFTintegralfalling(x, b-δ, b+δ, b-δ, b+δ)

    return (term1 + term2 + term3)/fs
end

function evalDTFTintegralrising(x::T,
                stop_freq,
                pass_freq,
                integration_lower_limit,
                integration_upper_limit) where T <: Real
    # rename.
    a = integration_lower_limit
    b = integration_upper_limit
    p = pass_freq
    s = stop_freq

    # singularity locations.
    x_0_negative = -1/(2*(p-s))
    x_0_positive = 1/(2*(p-s))

    # common intermediate value.
    k2 = 2*π*x*im

    ### term with negative exponent.
    A_negative::Complex{T} = zero(Complex{T})

    if x_0_negative - eps(T)*2 < x < x_0_negative + eps(T)*2
        # singularity case: when x is x_0_negative.
        A_negative = exp(-p*π*im/(p-s))*(b-a)

    else
        k1 = -π*im
        B = p*k1/(p-s)
        c = -k1/(p-s)
        q = c+k2

        A_negative = exp(B)/q * ( exp(b*q) - exp(a*q))
    end

    ### term with positive exponent.
    A_positive::Complex{T} = zero(Complex{T})

    if x_0_positive - eps(T)*2 < x < x_0_positive + eps(T)*2
        # singularity case: when x is x_0_positive.
        A_positive = exp(p*π*im/(p-s))*(b-a)

    else
        k1 = π*im
        B = p*k1/(p-s)
        c = -k1/(p-s)
        q = c+k2

        A_positive = exp(B)/q * ( exp(b*q) - exp(a*q))
    end

    ### third term.
    term3 = evalDTFTintegralrect(x, a, b)

    return A_positive/4 + A_negative/4 + term3/2
end


function evalDTFTintegralfalling(x::T,
                pass_freq,
                stop_freq,
                integration_lower_limit,
                integration_upper_limit) where T <: Real
    # rename.
    a = integration_lower_limit
    b = integration_upper_limit
    p = pass_freq
    s = stop_freq

    # singularity locations.
    x_0_negative = -1/(2*(s-p))
    x_0_positive = 1/(2*(s-p))

    # common intermediate value.
    k2 = 2*π*x*im

    ### term with negative exponent.
    A_negative::Complex{T} = zero(Complex{T})

    if x_0_negative - eps(T)*2 < x < x_0_negative + eps(T)*2
        # singularity case: when x is x_0_negative.
        A_negative = exp(-p*π*im/(s-p))*(b-a)

    else
        k1 = -π*im
        B = p*k1/(s-p)
        c = -k1/(s-p)
        q = c+k2
        A_negative = exp(B)/q * ( exp(b*q) - exp(a*q))
    end


    ### term with positive exponent.
    A_positive::Complex{T} = zero(Complex{T})

    if x_0_positive - eps(T)*2 < x < x_0_positive + eps(T)*2
        # singularity case: when x is x_0_positive.
        A_positive = exp(p*π*im/(s-p))*(b-a)

    else
        k1 = π*im
        B = p*k1/(s-p)
        c = -k1/(s-p)
        q = c+k2
        A_positive = exp(B)/q * ( exp(b*q) - exp(a*q))
    end

    ### third term.
    term3 = evalDTFTintegralrect(x, a, b)

    return A_positive/4 + A_negative/4 + term3/2
end

function evalDTFTintegralrect(x::T, a, b)::Complex{T} where T <: Real

    q = 2*π*x*im

    if zero(T) - eps(T)*2 < x < zero(T) + eps(T)*2
        # singularity case: when x is 0.
        return exp(q*b)*b - exp(q*a)*a
    end

    return (exp(b*q)-exp(a*q))/q
end

function gettimerangetunablefilter(N::Int, fs::T) where T
    @assert isodd(N)
    M = N - 1

    Ts::T = 1/fs

    A = fld(M,2)

    return (-A:1:A) * Ts
end
