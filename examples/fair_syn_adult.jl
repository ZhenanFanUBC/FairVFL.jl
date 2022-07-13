########################################################################
# Synchronous Vertical Federated Logistic Regression for Adult dataset
########################################################################
using FairVFL
using Printf
using SparseArrays

# load data
filename = "adult"
Xtrain, Ytrain, Xtest, Ytest = load_data(filename)
idxDict, na, nb = get_protected_idx(Xtrain, Ytrain, 1)

# config
config = Dict{String, Union{Int64, Float64, String}}()
config["num_classes"] = 2
config["num_clients"] = 3
config["num_epoches"] = 2000
config["learning_rate_model"] = 1e-2
config["learning_rate_lambda"] = 1e-1
config["fairness_tolerance"] = 1e-2
config["local_model"] = "mlp"

# vertically split data
Xtrain_split, Xtest_split = split_data(Xtrain, Xtest, config["num_clients"])

# initialize server 
server = Server(Ytrain, Ytest, idxDict, na, nb, config)

# initialize clients
clients = Vector{Client}(undef, config["num_clients"])
for id = 1:config["num_clients"]
    c = Client(id, Xtrain_split[id], Xtest_split[id], config)
    clients[id] = c
    # connect with server
    connect!(server, c)
end

# training
startT = time()
fair_vertical_lr_train!(server, clients)
endT = time()
@printf "training time: %.2f secs \n" endT - startT

# evaluation
evaluation(server, clients)

