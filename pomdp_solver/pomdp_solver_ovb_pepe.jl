# Kai Sandbrink, building off script by Johanni Brea
# 2023-01-31
# This script solves the Infobandit POMDP - OvB for the Pure Explore v. Pure Exploit Task

## LIBRARY IMPORT

using Pkg; Pkg.activate("."); Pkg.instantiate()
Pkg.add(["POMDPTools", "CSV"])
using POMDPs, POMDPTools, QuickPOMDPs, SARSOP, Statistics

using DelimitedFiles, Dates, CSV

## PARAMETERS

config = Dict([
    ("task", "OvB with Efficacy, No Sleep"),
    ("n_steps", 50),
    ("n_arms", 2),
    ("notes", ""),
    ("phi", 0.28),
    ("efficacy", 1), 
    ("volatility", 0),
])

println(config)

## GENERAL SETUP

n_steps = config["n_steps"]
n_arms = config["n_arms"]
phi = config["phi"]
efficacy = config["efficacy"]
volatility = config["volatility"]

datetime = Dates.format(now(), "yyyymmddHHMMSS")
config["datetime"] = datetime

n_repeats_stepthrough = 20

## SET UP POMDP
states = reshape([(0.5+x, 0.5-x, s, v) for x in (phi, -phi), s in 1:(n_steps+1), v in (false, true)], :)

## define initialstates
initialstates = Uniform(reshape([(0.5+x, 0.5-x, 1, false) for x in (phi, -phi)], :))

## define actions
actions = [[(:peek, 0)] ; [(:take, a) for a in (1,2)]]

transition = function (s, a)

    delta_steps = 0
    if s[n_arms+1] < (n_steps+1) ## i.e. if the final element in the state tuple is less than the total number of steps
        delta_steps = 1
    end

    @assert (1-volatility + volatility == 1) "probabilities don't add to 1 in transition function"

    SparseCat([(s[1], s[2], s[n_arms+1]+delta_steps, false), (s[2], s[1], s[n_arms+1]+delta_steps, true)], [1- volatility, volatility]) 
end

observation = function (a, sp)

    @assert (sp[1] + sp[2] == 1) "probabilities don't add to 1 in observation"

    if a[1] == :peek
        if !sp[4]
            return SparseCat([(1, 0, sp[3],), (0, 1, sp[3])], [sp[1], sp[2]]) ##first arm is successful with probability sp[1] and second arm with probability sp[2]
        else
            return SparseCat([(1, 0, sp[3],), (0, 1, sp[3])], [sp[2], sp[1]])
        end
    else
        return Deterministic((0, 0, 0))
    end
end

reward = function (s, a)
    if a[1] == :take

        p_int = 0.5 + efficacy/2
        p_other = 1-p_int    
        
        rew = (s[a[2]])*p_int + (s[3 - a[2]])*p_other
        
    else
        rew = 0
    end

    return rew
end

m = QuickPOMDP(
    states = states,
    actions = actions,
    transition = transition,
    discount = 1-eps(),
    isterminal = s -> s[3] == (n_steps+1),
    observations = [reshape([(x,1-x,s) for x in 0:1, s in 1:(n_steps+1)], :); [(0, 0, 0)]],
    observation = observation,
    reward = reward,
    initialstate = initialstates #needed so that alpha values are calculated correctly for all states
)

solver = SARSOPSolver(precision=0.2)

println("set up solver")

@time policy = solve(solver, m)

print("Expected Value: ",value(policy, initialstates))

## ACTIONS
println(policy.action_map) # which alpha vectors corresponds to which action.
println(actions)

## CREATE FOLDER AND SAVE OUTPUT
resultsfolder = "results/" * datetime
mkpath(resultsfolder)

open(resultsfolder * "/states.txt", "w") do file
    write(file, String(string(states)))
end

writedlm(resultsfolder * "/alphas.csv", policy.alphas, ", ")
writedlm(resultsfolder * "/action_map.csv", policy.action_map, ", ")
CSV.write(resultsfolder * "/config.csv", config)


n_observess = []
n_sleepss = []
rewss = []

rew = 0

for i in 1:(n_repeats_stepthrough)

    global rew = 0
    n_observes = 0
    n_sleeps = 0

    print("ROLLOUT ",i,"\n")
    
    for (b, s, a, o, r) in stepthrough(m, policy, "b, s, a, o, r")
        global rew += r
        #b.b shows belief state after step is taken but before information is integrated?

        if a == (:peek, 0)
            n_observes += 1
        elseif a == (:take, :sleep)
            n_sleeps += 1
        end
        
    end

    append!(rewss, [rew])
    append!(n_observess, [n_observes])
    append!(n_sleepss, [n_sleeps])

end 


println("Rollouts completed")
println(rewss)
println(n_observess)
println(n_sleepss)
