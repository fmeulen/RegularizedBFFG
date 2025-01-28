wd = @__DIR__ 
cd(wd)
using Pkg
Pkg.instantiate()

using Random
using Distributions
using Plots
using UnPack
using SpecialFunctions

using Test
using StatsFuns
using LinearAlgebra
using NNlib

include("bffg.jl")

# forward model
# observe Vᵢ = Xᵢ + Uᵢ', where 
# Uᵢ' ~ log Uᵢ², where Uᵢ ~ N(0,1)
# Xᵢ = ω + ψ X_{i-1} + η Wᵢ, where Wᵢ ~ N(0,1)

### Parameters
p = (ω =  0.0, ψ = 0.8, η=2.3, μ=-1.27, σ=√((π^2)/2))
# with larger variance
mv = 5.0
p_mv = (ω =  p.ω, ψ = p.ψ, η=p.η, μ=p.μ, σ = p.σ * √(mv))

#Random.seed!(10)
x0 = samplefromstationary(p) # Sample x0 from the stationary distribution 
S = 50 # Correct for time 0 (called tot_steps)  (Time steps)

Z = randn(S)
X = forward(x0, S, p, Z)
V =  X + log.(randn(S).^2)

# plotting
p1 = plot(X, label="X", title="latent")
p2 = plot(V, label="V", title="observed")
plot(p1, p2)


########################################################

# assess effect of ϵ

bf = backwardfilter(V, p)
Zᵒ = randn(S)
U = rand(S)
ϵ = 0.002 #18

Xᵒ, λs, ll, guids = forwardguide2(x0, bf, p, Zᵒ, V, ϵ, U)
Xᵒ0, λs0, ll0, guids0 = forwardguide2(x0, bf, p, Zᵒ,V, 0.0, U)
#Xᵒmv, λs_mv, lw_mv, guids_mv = forwardguide2(x0, bf, p_mv, Zᵒ,V, 0.0, U)

pX = plot(X,legend = :outertop, label="X")
plot!(pX, Xᵒ, color="red", label="Xᵒ")
plot!(pX, Xᵒ0, color="green", label="Xᵒ0")
# plot!(pX, Xᵒmv, color="orange", label="Xᵒ0_mv")

plw = plot(exp.(ll0), color="green", label="weigth, ϵ=0",legend = :outertop)
plot!(plw, exp.(ll), color="red", label="weigth")

plλ = plot(λs0, color="blue", label="λ with ϵ eq 0", title="λ is prob of guiding")
plot!(plλ, λs, color="red", label="λ with ϵ uneq 0",legend = :outertop)


@show std(exp.(ll)), smc_ess(exp.(ll))
@show std(exp.(ll0)), smc_ess(exp.(ll0))


l = @layout [a;b;c;d]
pall = plot(pX, plw, plλ, layout=l,size = (600, 1000))

savefig(pX, "paths.png")
savefig(plw, "weights.png")
savefig(plλ, "lambdas.png")
savefig(pall, "all.png")

# sum of logweights is decreasing in ϵ:
# Zᵒ = randn(S)
# ϵs = 0.0:0.01:10.0
# sumlw = sumlogweights(x0, bf, p, Zᵒ, V).(ϵs) 
# plot(ϵs, sumlw)


# Monte Carlo study, fix observations v
# Repeated simulation, each particle is a sampled path togeter with loglikelihood
# One would hope that with ϵ>0, the ess is larger than with ϵ=0

ϵ = 2.5

B = 1000
lls = zeros(B)
ll0s = zeros(B)
for i in 1:B
    Zᵒ = randn(S)
    Uᵒ = rand(S)    
    Xᵒ, λs, ll, guids = forwardguide2(x0, bf, p, Zᵒ, V, ϵ, Uᵒ)
    Xᵒ0, λs0, ll0, guids0 = forwardguide2(x0, bf, p, Zᵒ,V, 0.0, Uᵒ)
    lls[i] = sum(ll)
    ll0s[i] = sum(ll0)
end

plot(NNlib.softmax(lls), label="ϵ = $ϵ")
plot!(NNlib.softmax(ll0s), label="ϵ=0")


plot(first.(lls), color="red", title="total loglik over $B replications",
    label="ϵ=$ϵ", xlabel="replication id", ylabel="total loglikelihood")
plot!(first.(ll0s), color="green",label="ϵ=0")
savefig("montecarlo_loglik.png")

plot(last.(lls), color="red", title="SMC-ESS over $B replications",
    label="ϵ=$ϵ", xlabel="replication id", ylabel="SMC-ESS")
plot!(last.(ll0s), color="green",label="ϵ=0")
savefig("montecarlo_smc_ess.png")


# estimate some functional of the path
FF(x) = mean(x.>0.5) #sum(x.>0)  # path functional
ϵ = 0.0005

B = 10000

Fs = Float64[]
Fs0 = Float64[]
ℓ = Float64[]
ℓ0 = Float64[]
nrguids = Float64[]
nrguids0 = Float64[]
for _ in 1:B
    Zᵒ = randn(S)
    Uᵒ = rand(S)
    Xᵒ, λs, ll, guids = forwardguide2(x0, bf, p, Zᵒ, V, ϵ, Uᵒ)
    push!(nrguids, sum(guids))
    push!(ℓ, ll)    
    push!(Fs, FF(Xᵒ))

    Xᵒ0, λs0, ll0, guids0 = forwardguide2(x0, bf, p, Zᵒ,V, 0.0, Uᵒ)
    push!(nrguids0, sum(guids0))
    push!(ℓ0, ll0)    
    push!(Fs0, FF(Xᵒ0))
end

W = NNlib.softmax(ℓ) # weights
smc_ess(W) # ess of the weights 
dot(W, Fs)  # estimate

W0 = NNlib.softmax(ℓ0) # weights
smc_ess(W0) # ess of the weights 
dot(W0, Fs0)  # estimate

plot(W, label="ϵ=$ϵ")
plot!(W0, label="ϵ=0")





Random.seed!(5)

# generate trajectory
p = (ω =  0.0, ψ = 0.8, η=2.8, μ=-1.27, σ=√((π^2)/2))


x0 = samplefromstationary(p) # Sample x0 from the stationary distribution 
S = 200 # Correct for time 0 (called tot_steps)  (Time steps)

Z = randn(S)
X = forward(x0, S, p, Z)
V =  X + log.(randn(S).^2)

bf = backwardfilter(V, p)

iter = 15_000
bi = iter ÷ 2

ϵ = 180.5


# U = zeros(S)#rand(S)
# U = 0.5*ones(S)
U = rand(S)
Xs, Zs, lls, accperc = mcmc(x0, bf, p, V, ϵ, U; iter=iter, ρ_pcn = 0.999)
@show accperc


Xs0, Zs0, lls0, accperc0 = mcmc(x0, bf, p, V, 0.0, U; iter=iter, ρ_pcn = 0.999)
@show accperc0


c = 1.2*maximum(abs.(X))
p1 = plot(X, label="", ylims=(-c,c), title="ϵ=$ϵ", size=(700, 500))
for i in 1:100:bi
    plot!(Xs[i], color="magenta", alpha=0.05, label="")
end
for i in bi:100:iter
    plot!(Xs[i], color="blue", alpha=0.2, label="")
end
plot!(X, color="yellow", label="X", linewidth=2)

p2 = plot(X, label="", ylims=(-c,c), title="ϵ=0", size=(700, 500))
for i in 1:100:bi
    plot!(Xs0[i], color="magenta", alpha=0.05, label="")
end
for i in bi:100:iter
    plot!(Xs0[i], color="blue", alpha=0.2, label="")
end
plot!(X, color="yellow", label="X", linewidth=2)


plot(p1, p2, layout=@layout [a;b])

# savefig("mcmc.png")

# plot(last.(Zs))
# plot(lls)


# fg = forwardguide2(x0, bf, p, Z, V, ϵ, U)