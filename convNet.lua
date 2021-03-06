require 'gnuplot'
require 'torch'
require 'optim'
require 'paths'
require 'image'
require 'nn'

require 'cutorch'
require 'cunn'


-- needed in order to make all the layers work!
torch.setdefaulttensortype('torch.FloatTensor')


-- training conf
coefL1 = 0
coefL2 = 0
batch_size = 200
sgd_config = {
    learningRate = 1,
    learningRateDecay = 5.0e-3,
    momentum = 0.9,
}
cuda_enabled = true


function create_cnn_model()
    -- building the network:
    local model = nn.Sequential()
    model:add(nn.Reshape(1,28,28))
    -- layer 1:
    model:add(nn.SpatialConvolution(1,16,5,5))          -- 16 x 24 x 24
    model:add(nn.Tanh())
    model:add(nn.SpatialAveragePooling(2,2,2,2))        -- 16 x 12x 12
    -- layer 2:
    model:add(nn.SpatialConvolution(16,256,5,5))        -- 256 x 8 x 8
    model:add(nn.Tanh())
    model:add(nn.SpatialAveragePooling(2,2,2,2))        -- 256 x 4 x 4       
    -- layer 3 (2 fully connected neural nets):
    model:add(nn.Reshape(256*4*4))
    model:add(nn.Linear(256*4*4,200))
    model:add(nn.Tanh())
    model:add(nn.Linear(200,10))
    model:add(nn.SoftMax())
    
    return model
end


function oneHotEncoding(n)
    local v = torch.zeros(10)
    v[n+1] = 1
    return v
end

function oneHotDecoding(v)
    local indices = torch.linspace(0,9,10)
    local n = indices[v:eq(torch.max(v))]
    return n[1]
end


function getData()
    -- only for debugging porposes
    local mnist = require 'mnist'
    local trainset = mnist.traindataset()
    local testset = mnist.testdataset()
    return trainset, testset
end



function preProcessData(trainset, testset)

    -- transofrm this to Tensor (x + oneHot(y)) x trainset.size and compare performace on gpus vs cpu and optimal batch size.

    print(string.format('Training instances: %i',trainset.size))
    print(string.format('Testing instances: %i',testset.size))

    local tx_trainset = torch.Tensor(trainset.size,28*28)
    local ty_trainset = torch.Tensor(trainset.size,10)
    local tx_testset = torch.Tensor(testset.size,28*28)
    local ty_testset = torch.Tensor(testset.size,10)

    for i=1,trainset.size do
        tx_trainset[{i,{}}] = (trainset[i].x:float()-torch.Tensor(28,28):fill(128))/255
        ty_trainset[{i,{}}] = oneHotEncoding(trainset[i].y)
    end

    for i=1,testset.size do
        tx_testset[{i,{}}] = (testset[i].x:float()-torch.Tensor(28,28):fill(128))/255
        ty_testset[{i,{}}] = oneHotEncoding(testset[i].y)
    end
    return tx_trainset,ty_trainset,tx_testset,ty_testset
end



function preProcessData_old(trainset, testset)

    -- transofrm this to Tensor (x + oneHot(y)) x trainset.size and compare performace on gpus vs cpu and optimal batch size.

    print(string.format('Training instances: %i',trainset.size))
    print(string.format('Testing instances: %i',testset.size))

    -- train input normalization and label oneHotEncoding
    local train = {}
    train.size = trainset.size
    for i=1,trainset.size do
        train[i] = {}
        train[i].x = (trainset[i].x:float()-torch.Tensor(28,28):fill(128))/255
        train[i].y = oneHotEncoding(trainset[i].y)
    end
    -- test input normalization and label oneHotEncoding
    local test = {}
    test.size = testset.size
    for i=1,testset.size do
        test[i] = {}
        test[i].x = (testset[i].x:float()-torch.Tensor(28,28):fill(128))/255
        test[i].y = oneHotEncoding(testset[i].y)
    end
    return train, test
end



function train(model, criterion, dataset_x, dataset_y)


    print('-----------------TRAIN-----------------')

    -- train for the specified number of batches
    total_batches = dataset_x:size()[1]/batch_size
    errors = torch.Tensor(total_batches-1)
    print('Batch size:',batch_size)
    print('Total_batches:',total_batches)


    local train_start = sys.clock()

    -- retrieve model parameters and gradients
    parameters,gradParameters  = model:getParameters()

    local batch
    for batch = 1,total_batches-1 do

        -- display progress (only works if using console)
        xlua.progress(batch, total_batches)

        -- time measurement
        local start = sys.clock()

        -- create closure to evaluate f(x) and df/dx
        local feval = function(x)

            -- get new parameters
            if x ~= parameters then
                parameters:copy(x)
            end

            -- reset params
            local f = 0
            gradParameters:zero()

            for b=1,batch_size do

                -- get minibatch
                local inputs = dataset_x[batch*batch_size + b]
                local targets = dataset_y[batch*batch_size + b]


                if cuda_enabled then
                    inputs = inputs:cuda()
                    targets = targets:cuda()
                end

                -- evaluate function for complete minibatch
                local outputs = model:forward(inputs)
                local err = criterion:forward(outputs, targets)
                f = f + err

                -- appliying regularization penalties (L1 and L2)
                -- NOTE: important no to perform this operation even if coefs == 0 (very time consuming)
                if coefL1 ~= 0 or coefL2 ~= 0 then
                    -- Loss:
                    f = f + coefL1 * torch.norm(parameters,1)
                    f = f + coefL2 * (torch.norm(parameters,2)^2)/2

                    -- params:
                    gradParameters:add( torch.sign(parameters)*coefL1 + parameters*coefL2 )
                end

                -- estimate dl/dw
                dl_do = criterion:backward(outputs, targets)

                -- GPU computation
                if cuda_enabled then
                    model:backward(inputs, dl_do:cuda())
                else
                    model:backward(inputs, dl_do)
                end

            end

            -- normalize by the batch size
            gradParameters:div(batch_size)
            f = f/batch_size

            -- print('MSE for epoch ' .. tostring(epoch+1) .. ':')
            -- print(f)

            return f,gradParameters
        end

        -- optimization step SGD
        x,fx = optim.sgd(feval, parameters, sgd_config)
        errors[batch] = fx[1]

        -- local elapsed = sys.clock() - start
        -- print(string.format('Elapsed time: %f', elapsed))

    end

    local train_elapsed = sys.clock() - train_start
    -- print(string.format('Total elapsed training time for %i batch: %f', batch, train_elapsed))

    -- actual plot
    gnuplot.figure(1)
    gnuplot.title('Training error')
    gnuplot.xlabel('batch number')
    gnuplot.ylabel('Mean Square Error')
    gnuplot.plot(errors)

    return fx[1]
end



function test(model, dataset_x, dataset_y)

    print('-----------------TEST-----------------')

    -- time measurement
    local start = sys.clock()

    -- confusion matrix to hold test accuracy
    local classes = torch.linspace(0,9,10):totable() -- {0,1,2,3,4,5,6,7,8,9}
    local num_classes = #classes
    local confusion = torch.zeros(num_classes, num_classes)
    -- local confusion = optim.ConfusionMatrix(classes)


    for t = 1,dataset_x:size()[1] do

        -- get data
        local input = dataset_x[t]
        local target = oneHotDecoding(dataset_y[t])

        -- forward pass
        local pred
        if cuda_enabled then
            pred = model:forward(input:cuda())
            pred = oneHotDecoding(pred:float())
        else
            pred = model:forward(input)
            pred = oneHotDecoding(pred)
        end

        -- update confusion matrix
        confusion[pred+1][target+1] = confusion[pred+1][target+1] + 1
        -- confusion:add(pred+1, target+1) -- because of the lua 1-indexing scheme
        
    end

    -- accuracy info
    diag = confusion[torch.eye(num_classes):byte()]
    acc = torch.sum(diag)/torch.sum(confusion)

    precission = torch.Tensor(num_classes)
    recall = torch.Tensor(num_classes)
    f_score = torch.Tensor(num_classes)

    for i=1,num_classes do
        precission[i] = confusion[i][i] / torch.sum(confusion[i])       -- sum over rows (other --> clas i)
        recall[i] = confusion[i][i] / torch.sum(confusion[{{},i}])      -- sum over columns ()
        f_score[i] = 2*(precission[i] * recall[i]) / (precission[i] + recall[i])
    end

    print('Classification accuracy:', acc)
    print('Mean Precission:', torch.mean(precission))
    print('Mean Recall', torch.mean(recall))
    print('Mean Fscore', torch.mean(f_score))

    --monitorimage.display(confusion:render())

    -- time measurement
    elapsed = sys.clock() - start
    print(string.format('Total elapsed testing time: %f', elapsed))

    return 1-acc
end



function main()

    -- logging variables
    save_path = '/users/josemarcos/Desktop/convNet/models'
    trainLogger = optim.Logger(paths.concat(save_path, 'train.log'))
    testLogger = optim.Logger(paths.concat(save_path, 'test.log'))

    -- datasets
    train_x,train_y, test_x,test_y = preProcessData(getData())
    train_err = 1/0
    test_err = 1/0


    -- model
    model = create_cnn_model()
    criterion = nn.MSECriterion()

    -- ship model to the GPU
    if cuda_enabled then
        model:cuda()
        criterion:cuda()
    end

    epoch = 1

    while train_err > 0.01 or epoch < 10 do
        -- train stage
        train_err = train(model, criterion, train_x, train_y)
        print('Training error=',train_err)

        -- saving the model
        local filename = paths.concat(save_path, 'convNet_r' .. tostring(epoch) .. '.net')
        os.execute('mkdir -p ' .. sys.dirname(filename))
        torch.save(filename, model)

        -- test stage
        test_err = test(model, test_x,test_y)
        print('Test error=', test_err)


        -- plot evolution
        trainLogger:style{['% mean class accuracy (train set)'] = '-'}
        testLogger:style{['% mean class accuracy (test set)'] = '-'}
        trainLogger:plot()
        testLogger:plot()

        epoch = epoch + 1
    end
end


-- main()


model = torch.load('models/convNet_r1.net')
_,_, test_x,test_y = preProcessData(getData())
test(model, test_x, test_y)



