wd = @__DIR__ 
cd(wd)

using Random
using Distributions
using Plots
using UnPack
using SpecialFunctions
using Test

include("bffg.jl")

# forward model
# observe Vᵢ = Xᵢ + Uᵢ', where 
# Uᵢ' ~ log Uᵢ², where Uᵢ ~ N(0,1)
# Xᵢ = ω + ψ X_{i-1} + η Wᵢ, where Wᵢ ~ N(0,1)

### Parameters
p = (ω =  0.0, ψ = 0.9, η=0.163, μ=-1.27, σ=√((π^2)/2))
# with larger variance
mv = 5.0
p_mv = (ω =  p.ω, ψ = p.ψ, η=p.η, μ=p.μ, σ = p.σ * √(mv))

#Random.seed!(10)
x0 = samplefromstationary(p) # Sample x0 from the stationary distribution 
S = 75 # Correct for time 0 (called tot_steps)  (Time steps)

Z = randn(S)
X = forward(x0, S, p, Z)
V =  X + log.(randn(S).^2)

# plotting
p1 = plot(X, label="X", title="latent")
p2 = plot(V, label="V", title="observed")
plot(p1, p2)


########################################################

# assess effect of ϵ

bf_ = backwardfilter(V, p)

bf = [Message(0.0, m.F, m.H) for m in bf_]

Zᵒ = randn(S)
ϵ = 0.1

Xᵒ, λs, lw, guids = forwardguide(x0, bf, p, Zᵒ, V, ϵ)
Xᵒ0, λs0, lw0, guids0 = forwardguide(x0, bf, p, Zᵒ,V, 0.0)
Xᵒmv, λs_mv, lw_mv, guids_mv = forwardguide(x0, bf, p_mv, Zᵒ,V, 0.0)

pX = plot(X,legend = :outertop, label="X")
plot!(pX, Xᵒ, color="red", label="Xᵒ")
plot!(pX, Xᵒ0, color="green", label="Xᵒ0")
# plot!(pX, Xᵒmv, color="orange", label="Xᵒ0_mv")

plw = plot(lw0, color="green", label="logweigth, ϵ=0",legend = :outertop)
plot!(plw, lw, color="red", label="logweigth")

plλ = plot(guids, color="blue", label="guids")
plot!(plλ, λs, color="red", label="λ",legend = :outertop)

l = @layout [a;b;c;d]
pall = plot(pX, plw, plλ, layout=l,size = (600, 1000))

savefig(pX, "paths.png")
savefig(plw, "weights.png")
savefig(plλ, "lambdas.png")
savefig(pall, "all.png")

# sum of logweights is decreasing in ϵ:
Zᵒ = randn(S)
ϵs = 0.0:0.01:10.0
sumlw = sumlogweights(x0, bf, p, Zᵒ, V).(ϵs) 
plot(ϵs, sumlw)

@show std(exp.(lw)), smc_ess(exp.(lw))
@show std(exp.(lw0)), smc_ess(exp.(lw0))

# Monte Carlo
ϵ = 0.1

B = 1000
lws = []
lws0 = []
for _ in 1:B
    Zᵒ = randn(S)
    Xᵒ, λs, lw, guids = forwardguide(x0, bf, p, Zᵒ, V, ϵ)
    Xᵒ0, λs0, lw0, guids0 = forwardguide(x0, bf, p, Zᵒ,V, 0.0)
    push!(lws, [sum(lw), smc_ess(exp.(lw))])
    push!(lws0, [sum(lw0), smc_ess(exp.(lw0))])
    
end

plot(first.(lws), color="red", title="total loglik over $B replications",
    label="ϵ=$ϵ", xlabel="replication id", ylabel="total loglikelihood")
plot!(first.(lws0), color="green",label="ϵ=0")
savefig("montecarlo_loglik.png")

plot(last.(lws), color="red", title="SMC-ESS over $B replications",
    label="ϵ=$ϵ", xlabel="replication id", ylabel="SMC-ESS")
plot!(last.(lws0), color="green",label="ϵ=0")
savefig("montecarlo_smc_ess.png")

# mcmc
#p = p_mv

iter = 10000
Zᵒ = randn(S)
Xᵒ, λs, lw, guids = forwardguide(x0, bf, p, Zᵒ, V, ϵ)
ℓ = sum(lw)
Xs = [Xᵒ]
ρ = 0.95
acc = 0

for _ in 1:iter
    W = randn(S)
    Znew = ρ * Zᵒ + √(1-ρ^2) * W
    Xnew, _, lwnew, _ = forwardguide(x0, bf, p, Znew, V, ϵ)
    ℓnew = sum(lwnew)
    if log(rand()) < ℓnew - ℓ
        ℓ = ℓnew
        Zᵒ .= Znew
        Xᵒ .= Xnew
        acc += 1
    end
    push!(Xs, copy(Xᵒ))
end
@show round(100*acc/iter;digits=2)

plot(X)
for i in 1:100:iter
    plot!(Xs[i], color="red", alpha=0.2, label="")
end
plot!(X, color="blue")
savefig("mcmc.png")
#plot!(mean(Xs), color="black")
