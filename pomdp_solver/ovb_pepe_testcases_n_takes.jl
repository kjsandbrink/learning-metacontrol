# Kai Sandbrink, building off script by Johanni Brea
# 2022-10-22
# This script solves the Infobandit POMDP - OvB Style, for a given bias phi, using exponential decay of efficacy [binned up to 5 states] of given speed k, probabilistic readout observation

using Pkg; Pkg.activate("."); Pkg.instantiate()
Pkg.add(["POMDPTools", "CSV"])
using POMDPs, POMDPTools, QuickPOMDPs, SARSOP, Statistics

using DelimitedFiles, Dates, CSV

include("ovb_pepe_pomdp_wrapper_compute_n_observes.jl")

## PARAMETERS

#phis = range(0.05, 0.5, step=0.05)
phi=0.4
#eff = 1
effs = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
vol = 0.05 ## volatility here means probability of transition, so is 1/2 python volatility
#precision = 0.1
precision = 1e-3
n_repeats_stepthrough = 20
#precision=4
initialsteps = 1

## FOR RECORD ONLY, BE SURE TO CHANGE IN WRAPPER FILE
config = Dict([
    ("task", "OvB with Efficacy, No Sleep"),
    ("n_steps", 50),
    ("n_arms", 2),
    ("notes", ""),
    ("vol", vol),
    ("precision", precision),
    ("phi", phi),
    ("effs", effs),
    ("n_repeats_stepthrough", 20),
    ("initialsteps", initialsteps)
])

#print("Using precision", precision, "\n")

## INITIALIZATIONS

datetime = Dates.format(now(), "yyyymmddHHMMSS")

## FOR EACH COMBINATION, CALCULATE OPTIMAL ARRAYS

#evs = zeros(length(phis),)
evs = zeros(length(effs),)
rewss = []
n_observess = []

#for (i, phi) in enumerate(phis)
for (i, eff) in enumerate(effs)
    eff_folder = "results/pepe/" * datetime * "/eff" * string(Int(eff*1000))
    mkpath(eff_folder)
    cd(eff_folder) do
        ev, rews, n_observes = calc_ovb_pepe_ev(phi, eff, vol, n_repeats_stepthrough, precision, config["n_steps"], config["n_arms"], config["initialsteps"])
        evs[i] = ev

        append!(rewss,[rews])
        append!(n_observess, [n_observes])    
    end
end

print("EVs for test cases: ", evs, "\n")
print("rewss for test cases: ", rewss, "\n")
print("n_observess for test cases: ", n_observess, "\n")

print("Saving to file\n")

#EVs = hcat(phis, evs, rewss, n_observess)
EVs = hcat(effs, evs, rewss, n_observess)
print("Concatenated successfully\n")

EVs = vcat(reshape(["efficacy", "EVs", "rewss", "n_observess"], 1, 4), EVs)
print("Added titles\n")

## once for safety (can be deleted if runs successfully)
resultsfolder = "results/pepe/" * datetime
mkpath(resultsfolder)

writedlm(resultsfolder * "/EVs_phi" * string(phi) * "_prec" * string(precision) * "_vol" * string(vol) * ".csv", EVs, ", ")
CSV.write(resultsfolder * "/config.csv", config)

print("Done")