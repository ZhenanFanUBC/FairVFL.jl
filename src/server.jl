########################################################################
# Server for synchronous learning
########################################################################
mutable struct Server{T1<:Int64, T2<:Float64, T3<:Vector{T2}, T4<:Vector{T1}, T5<:Matrix{T2}, T6<:Vector{Client}, T7<:Dict{T1, String}} 
    Ytrain::T4                       # training label
    Ytest::T4                        # test label
    num_classes::T1                  # number of classes
    num_clients::T1                  # number of clients
    num_epoches::T1                  # number of epoches
    clients::T6                      # set of clients
    train_embeddings::T5             # embeddings for all training data 
    test_embeddings::T5              # embeddings for all test data 
    idxDict::T7                      # Dictionary for protected classes
    grads::T5                        # gradient information
    λ::T3                            # Lagrangian multipliers
    ϵ::T2                            # fairness tolerance
    β::T2                            # learning rate for λ
    na::T1                           # number of training points in protected class a
    nb::T1                           # number of training points in protected class b
    function Server(Ytrain::Vector{Int64}, 
                    Ytest::Vector{Int64}, 
                    idxDict::Dict{Int64, String},
                    na::Int64,
                    nb::Int64,
                    config::Dict{String, Union{Int64, Float64, String}})
        num_classes = config["num_classes"]
        num_clients = config["num_clients"]
        num_epoches = config["num_epoches"]
        ϵ = config["fairness_tolerance"]
        β = config["learning_rate_lambda"]
        λ = zeros(Float64, 2)
        clients = Vector{Client}(undef, num_clients)
        train_embeddings = zeros(Float64, num_classes, length(Ytrain))
        test_embeddings = zeros(Float64, num_classes, length(Ytest))
        grads = zeros(Float64, num_classes, length(Ytrain))
        new{Int64, Float64, Vector{Float64}, Vector{Int64}, Matrix{Float64}, Vector{Client}, Dict{Int64, String}}(Ytrain, Ytest, num_classes, num_clients, num_epoches, clients, train_embeddings, test_embeddings, idxDict, grads, λ, ϵ, β, na, nb)
    end
end

# connect with client
function connect!(s::Server, c::Client)
    s.clients[c.id] = c
end


# send embeddings to server
function send_embedding!(c::Client, s::Server; tag = "training")
    if tag == "training"
        if typeof(c.W) <: Matrix{Float64}
            embedding = c.W * c.Xtrain
        else
            embedding, c.back = Zygote.pullback(()->c.W(c.Xtrain), params(c.W))
        end
    else
        if typeof(c.W) <: Matrix{Float64}
            embedding = c.W * c.Xtest
        else
            embedding = c.W(c.Xtest)
        end
    end
    update_embedding!(s, embedding, tag=tag)
end

# update embedding
function update_embedding!(s::Server, embedding::Matrix{Float64}; tag = "training")
    if tag == "training"
        s.train_embeddings .+= embedding
    else
        s.test_embeddings .+= embedding
    end
end

# compute gradient
function compute_gradient!(s::Server)
    num_classes = s.num_classes
    num_data = length(s.Ytrain)
    loss = 0.0
    loss_a = 0.0
    loss_b = 0.0
    grads = zeros( num_classes, num_data )
    # compute gradient
    for i = 1:num_data
        y = s.Ytrain[i]
        emb = s.train_embeddings[:,i]
        pred = softmax(emb)
        Δloss = neg_log_loss(pred, y)
        loss += Δloss
        grads[:, i] .= pred
        grads[y, i] -= 1.0
        # scaling of the gradients
        if s.idxDict[i] == "a"
            v = 1/num_data + (s.λ[1] - s.λ[2])/s.na
            loss_a += Δloss
        elseif s.idxDict[i] == "b"
            v = 1/num_data - (s.λ[1] - s.λ[2])/s.nb
            loss_b += Δloss
        else
            v = 1/num_data
        end
        grads[:, i] .*= v
    end
    # update local gradient information 
    s.grads .= grads
    # send gradient information to clients
    for c in s.clients
        update_grads!(c, grads)
    end
    # set embeddings to be zero
    s.train_embeddings .= 0.0
    # return mini-batch loss
    return loss/num_data, loss_a/s.na, loss_b/s.nb
end

# update Lagrangian multipliers
function update_λ!(s::Server, loss_a::Float64, loss_b::Float64)
    D = loss_a - loss_b
    g = s.β*[D - s.ϵ; -D - s.ϵ]
    s.λ .+= g
    s.λ[1] = max(s.λ[1], 0.0)
    s.λ[2] = max(s.λ[2], 0.0)
end


########################################################################
# Server for asynchronous learning
########################################################################
mutable struct AsynServer{T1<:Int64, T2<:Float64, T3<:Vector{T2}, T4<:Vector{T1}, T5<:SharedArray{T2, 3}, T6<:Vector{AsynClient}} 
    Ytrain::T4                       # training label
    Ytest::T4                        # test label
    num_classes::T1                  # number of classes
    num_clients::T1                  # number of clients
    learning_rate::T2                # learning rate
    clients::T6                      # set of clients
    b::T3                            # server model
    embeddings::T5                   # latest embeddings
    train_embeddings::T5             # embeddings for all training data (used for final evaluation)
    test_embeddings::T5              # embeddings for all test data (used for final evaluation)
    function AsynServer(Ytrain::Vector{Int64}, Ytest::Vector{Int64}, config::Dict{String, Union{Int64, Float64, String}})
        num_classes = config["num_classes"]
        num_clients = config["num_clients"]
        learning_rate = config["learning_rate"]
        clients = Vector{AsynClient}(undef, num_clients)
        b = zeros(Float64, num_classes)
        embeddings = SharedArray{Float64}(num_clients, num_classes, length(Ytrain))
        train_embeddings = SharedArray{Float64}(num_clients, num_classes, length(Ytrain))
        test_embeddings = SharedArray{Float64}(num_clients, num_classes, length(Ytest))
        new{Int64, Float64, Vector{Float64}, Vector{Int64}, SharedArray{Float64, 3}, Vector{AsynClient}}(Ytrain, Ytest, num_classes, num_clients, learning_rate, clients, b, embeddings, train_embeddings, test_embeddings)
    end
end

# connect with client
function connect!(s::AsynServer, c::AsynClient)
    s.clients[c.id] = c
end

# send embeddings to server
function send_embedding!(c::AsynClient, s::AsynServer; tag = "batch")
    if tag == "batch"
        num_data = length(s.Ytrain)
        if c.num_commu == 0
            batch = collect(1:num_data)
        else
            batch = sample(collect(1:num_data), c.batch_size, replace=false)
        end
        Xbatch = c.Xtrain[:, batch]
        if typeof(c.W) <: Matrix{Float64}
            embedding = c.W * Xbatch
        else
            embedding, c.back = Zygote.pullback(()->c.W(Xbatch), params(c.W))
        end
        update_embedding(s, c.id, embedding, batch)
        @printf "Client %i finish uploading embedding \n" c.id
        return batch
    elseif tag == "training"
        if typeof(c.W) <: Matrix{Float64}
            embedding = c.W * c.Xtrain
        else
            embedding = c.W(c.Xtrain)
        end
        s.train_embeddings[c.id,:,:] .= embedding
    else
        if typeof(c.W) <: Matrix{Float64}
            embedding = c.W * c.Xtest
        else
            embedding = c.W(c.Xtest)
        end
        s.test_embeddings[c.id,:,:] .= embedding
    end
end

# update embedding
function update_embedding!(s::AsynServer, id::Int64, embedding::Matrix{Float64}, batch::Vector{Int64})
    s.embeddings[id,:,batch] .= embedding
end

# compute gradient
function send_gradient!(s::AsynServer, id::Int64, batch::Vector{Int64})
    batch_size = length(batch)
    num_classes = s.num_classes
    sum_embeddings = reshape( sum( s.embeddings[:,:,batch], dims=1), num_classes, batch_size )
    loss = 0.0
    grads = zeros( num_classes, batch_size )
    # compute mini-batch gradient
    for i = 1:batch_size
        y = s.Ytrain[ batch[i] ]
        emb = sum_embeddings[:, i] + s.b
        pred = softmax(emb)
        loss += neg_log_loss(pred, y)
        grads[:, i] .= pred
        grads[y, i] -= 1.0
    end
    
    # send gradient information to client
    update_grads(s.clients[id], grads)

    # return mini-batch loss
    return loss / batch_size
end

# Compute training and test accuracy
function eval(s::Union{Server, AsynServer})
    train_size = length(s.Ytrain)
    test_size = length(s.Ytest)
    train_loss = 0.0
    train_acc = 0.0
    test_loss = 0.0
    test_acc = 0.0
    for i = 1:train_size
        y = s.Ytrain[i]
        emb = s.train_embeddings[:, i] 
        pred = softmax(emb)
        train_loss += neg_log_loss(pred, y)
        if argmax(pred) == y
            train_acc += 1.0
        end
    end
    for i = 1:test_size
        y = s.Ytest[i]
        emb = s.test_embeddings[:, i]
        pred = softmax(emb)
        test_loss += neg_log_loss(pred, y)
        if argmax(pred) == y
            test_acc += 1.0
        end
    end
    return train_loss/train_size, train_acc/train_size, test_loss/test_size, test_acc/test_size
end

