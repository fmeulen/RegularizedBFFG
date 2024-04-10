wd = @__DIR__ 
cd(wd)

using Random
using Distributions
using Plots
using UnPack

### Parameters
p = (ω =  0.0, ψ = 0.9, η=0.363,μ=-1.27, σ=√((π^2)/2))
# with larger variance
mult_var = 3.0
p_mv = (ω =  0.0, ψ = 0.9, η=0.363,μ=-1.27, σ = √(mult_var*(π^2)/2))

# Forward simulate (z ~ N(0,1))
forward(x, p, z) = p.ω + p.ψ *x + p.η * z

function xsim(x0, S, p, Z)
  @assert length(Z)==S "length of innovations should equal S"
  x = fill(x0, S+1)
  for i ∈ 2:S+1
    x[i] = forward(x[i-1], p, Z[i-1])
  end
  x[2:end]
end

Random.seed!(10)
x0 = rand(Normal(p.ω/(1.0 - p.ψ), sqrt(p.η^2 / (1 - p.ψ^2))))    # Sample x0 from the stationary distribution 
S = 500 # Correct for time 0 (called tot_steps)  (Time steps)

Z = randn(S)
X = xsim(x0, S, p, Z)
R = exp.(X/2) .* randn(S)
V = log.(R.^2)
U = V - X

# plotting
p1 = plot(X, label="X", title="latent")
p2 = plot(V, label="V", title="observed")
plot(p1, p2)


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
    cnew = c - logpdf(NormalCanon(F,H),0) + logpdf(Normal(F/H,C),ω)
    Message(cnew, Fnew, Hnew)
end   

leaf_coefs(V, p) = [Message(logpdf(Normal(p.μ, p.σ), v), 
                                (v-p.μ)/p.σ^2, p.σ^(-2)) for v in V]

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


bf = backwardfilter(V,p)
bf_mv = backwardfilter(V,p_mv)


function guide(x, m::Message, p, z) 
    @unpack c, F, H = m
    @unpack ω, ψ, η, μ, σ = p
    μg = F/H
    σg2 = H^(-1)
    η2 = η^2
    μx = ω + ψ * x
    μ̃ = (μx*σg2 + μg*η2)/(η2 + σg2)
    σ̃ = sqrt((η2*σg2)/(η2+σg2))
    μ̃ + σ̃ * z
end

logweight(x, m::Message) = m.c + m.F * x + 0.5*x*m.H*x

robustify(m::Message, ϵ) = Message(m.c + log(ϵ),m.F, m.H)

function forwardguide(x0, bf, p, Z; ϵ=0.1)
    @assert ϵ>0 "ϵ should be strictly positive"
    S = length(bf)
    @assert S==length(Z) "length of innovations should equal S"
    x = x0
    xs = [x]
    for i in 1:S
         # Sampling from guided or unconditional?
        kg = exp(logweight(x,bf[i]))
        λ = kg/(kg + ϵ)
        m = robustify(bf[i], ϵ)
        z = Z[i]
        x = rand()<λ ? guide(x, m, p, z) : forward(x, p, z)  
        push!(xs, x)
        # also compute logweight and save
    end
    xs
end

Zᵒ = randn(S)
Xᵒ = forwardguide(x0, bf, p, Zᵒ)
Xᵒmv = forwardguide(x0, bf, p_mv, Zᵒ)
plot!(p1, Xᵒ, color="red", label="Xᵒ")
plot!(p1, Xᵒmv, color="green", label="Xᵒmv")