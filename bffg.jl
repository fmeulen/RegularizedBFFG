# Forward simulate (z ~ N(0,1))
forward(x, p, z) = p.ω + p.ψ * x + p.η * z

function forward(x0, S, p, Z)
  @assert length(Z)==S "length of innovations should equal S"
  x = Vector{typeof(x0)}(undef,S)# fill(x0, S)
  x[1] = forward(x0, p, Z[1])
  for i ∈ 2:S
    x[i] = forward(x[i-1], p, Z[i])
  end
  x
end

samplefromstationary(p) = rand(Normal(p.ω/(1.0 - p.ψ), sqrt(p.η^2 / (1 - p.ψ^2))))    



# Backward filtering
struct Message{Tc, TF, TH}
    c::Tc
    F::TF
    H::TH
end

fuse(m1::Message, m2::Message) = Message(m1.c+m2.c, m1.F+m2.F, m1.H+m2.H)

function pullback(m::Message, p)
    @unpack c, F, H = m
    @unpack ω, ψ, η, μ, σ = p
    C = η^2 + 1/H 
    Hnew = ψ^2 / C
    Fnew = ψ * (F/H - ω)/C
    cnew = 0.0#c - logpdf(NormalCanon(F,H),0) + logpdf(Normal(F/H,C),ω)
    Message(cnew, Fnew, Hnew)
end   

leaf_coefs(V, p) = [Message(logpdf(Normal(p.μ, p.σ), v), (v-p.μ)/p.σ^2, p.σ^(-2)) for v in V]
                                

function backwardfilter(V, p)
    leafmessages = leaf_coefs(V, p)
    m = leafmessages[end]
    ms = [m]
    S = length(V)
    for i in S-1:-1:1
        m = fuse(pullback(m, p), leafmessages[i])
        pushfirst!(ms, m)
    end
    ms
end



function guide(x, m::Message, p, z) 
    @unpack c, F, H = m
    @unpack ω, ψ, η, μ, σ = p
    μg = F/H
    σg2 = H^(-1)
    η2 = η^2
    μx = ω + ψ * x
    μᵒ = (μx*σg2 + μg*η2)/(η2 + σg2)
    σᵒ= sqrt((η2*σg2)/(η2+σg2))
    μᵒ + σᵒ * z
end

g(x, m::Message) = exp(m.c + m.F * x - 0.5*x*m.H*x)

#robustify(m::Message, ϵ) = Message(m.c + log(ϵ),m.F, m.H)

logchisq_density(x, ndf) = 
      (2^(ndf/2) * gamma(ndf/2))^(-1) * exp(0.5 * ndf * x - 0.5 * exp(x))


logpdf_logχ2(x) = x + logpdf(Chisq(1), exp(x))   

x= 0.5
logchisq_density(x, 1) 
logpdf_logχ2(x)

### Weight function for simple twist
# log_simple_twist_weights <- function(x_circ, r_input, v_input, 
#                                      g_coeffs, p_coeffs, epsilon){
  
function logweights(x0, Xᵒ, V, p, bf, ϵ)
    m = bf[1]
    W = [log((g(x0,pullback(m, p)) + ϵ)/(g(Xᵒ[1], m)+ ϵ)) + logpdf_logχ2(V[1]-Xᵒ[1])]
    S = length(V)
    for i ∈ 2:S
        m = bf[i]
        w = log((g(Xᵒ[i-1],pullback(m, p)) + ϵ)/(g(Xᵒ[i], m)+ ϵ)) + logpdf_logχ2(V[i]-Xᵒ[i])
        push!(W, w)
    end
    W
end

sumlogweights(x0, bf, p, Z, V) = (ϵ) -> sum(forwardguide(x0, bf, p, Z, V, ϵ)[3])

  

function forwardguide(x0, bf, p, Z, V, ϵ)
    #@assert ϵ>0 "ϵ should be strictly positive"
    S = length(bf)
    @assert S==length(Z) "length of innovations should equal S"
    x = x0
    xs = [x]
    λs = Float64[]
    for i in 1:S
         # Sampling from guided or unconditional?
        m = bf[i]
        κg = g(x,m)
        λ = κg/(κg + ϵ) # prob to sample from guided
        z = Z[i]
        x = rand()<λ ? guide(x, m, p, z) : forward(x, p, z)  
        push!(xs, x)
        push!(λs, λ)
    end
    lw = logweights(x0, xs, V, p, bf, ϵ)
    xs, λs, lw
end
