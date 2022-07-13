module FairVFL

using LinearAlgebra
using Printf
using SparseArrays
using Random
using SharedArrays
using Distributed
using Combinatorics
using StatsBase
using Flux
using Zygote

export Client, AsynClient
export Server, AsynServer
export connect!
export get_protected_idx
export send_embedding!, update_embedding!
export update_model!, update_grads!, compute_gradient!, send_gradient!
export eval
export softmax, neg_log_loss
export load_data, split_data, generate_batches
export fair_vertical_lr_train!, evaluation
export read_libsvm

include("./utils.jl")
include("./client.jl")
include("./server.jl")
include("./training.jl")

end # module
