########################################################################
# Synchronous Vertical Federated Logistic Regression
########################################################################

function fair_vertical_lr_train!(server::Server, clients::Vector{Client})
    # number of epoches
    num_epoches = server.num_epoches 
    # start training
    @inbounds for epoch = 1:num_epoches
        for c in clients
            # client compute and upload embeddings
            send_embedding!(c, server)
        end
        # server compute the loss and the gradient
        loss, loss_a, loss_b = compute_gradient!(server)
        @printf "Epoch %d, Loss %.2f, Loss(a) %.2f, Loss(b) %.2f\n" epoch loss loss_a loss_b
        # server and clients update model
        update_λ!(server, loss_a, loss_b)
        for c in clients
            update_model!(c)
        end
    end
end

function evaluation(server::Server, clients::Vector{Client})
    server.train_embeddings .= 0.0
    server.test_embeddings .= 0.0
    # test and train accuracy
    for c in clients
        # client compute and upload training embeddings
        send_embedding!(c, server, tag = "training")
        # client compute and upload test embeddings
        send_embedding!(c, server, tag = "test")
    end
    train_loss, train_acc, test_loss, test_acc = eval(server)
    @printf "Train Loss %.2f, Train Accuracy %.2f, Test Loss %.2f, Test Accuracy %.2f\n" train_loss train_acc test_loss test_acc
end


########################################################################
# Asynchronous Vertical Federated Logistic Regression
########################################################################

function vertical_lr_train!(server::AsynServer, clients::Vector{AsynClient}, time_limit::Float64)
    tag = true
    # set time limit
    @async begin
        sleep(time_limit)
        tag = false
    end
    # start training
    Threads.@threads for c in clients
        while tag
            # client compute and send embedding to server
            batch = send_embedding!(c, server)
            # server compute and send back the gradient
            send_gradient!(server, c.id, batch)
            # client update model
            update_model!(c, batch)
            # time break
            sleep(c.ts)
        end
    end
    @printf "Finish training after %.2f seconds\n" time_limit
    # print number of communication rounds
    for c in clients
        @printf "Client %i communicate %i times with server \n" c.id c.num_commu
    end
end

function evaluation(server::AsynServer, clients::Vector{AsynClient})
    # test and train accuracy
    for c in clients
        # client compute and upload training embeddings
        send_embedding!(c, server, tag = "training")
        # client compute and upload test embeddings
        send_embedding!(c, server, tag = "test")
    end
    train_loss, train_acc, test_loss, test_acc = eval(server)
    @printf "Train Loss %.2f, Train Accuracy %.2f, Test Loss %.2f, Test Accuracy %.2f\n" train_loss train_acc test_loss test_acc
end