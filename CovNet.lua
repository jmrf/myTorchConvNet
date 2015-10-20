require 'torch'
require 'optim'
require 'paths'
require 'image'
require 'nn'

-- require 'util.LogsLoader'


-- needed in order to make all the layers work!
torch.setdefaulttensortype('torch.FloatTensor')


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
    -- train input normalization and label oneHotEncoding
    local test = {}
    test.size = testset.size
    for i=1,testset.size do
        test[i] = {}
        test[i].x = (testset[i].x:float()-torch.Tensor(28,28):fill(128))/255
        test[i].y = oneHotEncoding(testset[i].y)
    end
    return train, test
end



function train(model, criterion, dataset)

    -- training conf
    local MAX_EPOCH = 200
    local batch_size = dataset.size/MAX_EPOCH     -- by now it is ok
    local sgd_config = {
        learningRate = 1,
        learningRateDecay = 5.0e-6,
        momentum = 0.9,
    }


    local train_start = sys.clock()

    -- retrieve model parameters and gradients
    parameters,gradParameters  = model:getParameters()

    -- train for the specified number of epochs
    batch_num = 0

    for epoch = 1,MAX_EPOCH do
        -- print('\n--------------------')
        -- print('#epoch: ' .. epoch)

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

            for b = 1,batch_size do

                -- get minibatch
                local inputs = dataset[batch_num*batch_size + b].x
                local targets = dataset[batch_num*batch_size + b].y

                -- evaluate function for complete minibatch
                local outputs = model:forward(inputs)
                local err = criterion:forward(outputs, targets)
                f = f + err

                -- estimate dl/dw
                dl_do = criterion:backward(outputs, targets)
                model:backward(inputs, dl_do)

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
        batch_num = batch_num + 1

        -- display progress (only works if using console)
        xlua.progress(epoch, MAX_EPOCH)

        local elapsed = sys.clock() - start
        -- print(string.format('Elapsed time: %f', elapsed))

        -- saving the model
        local filename = paths.concat(save_path, 'convNet_e' .. tostring(epoch) .. '.net')
        os.execute('mkdir -p ' .. sys.dirname(filename))
        torch.save(filename, model)
    end

    local train_elapsed = sys.clock() - train_start
    print(string.format('Total elapsed training time for %i epochs: %f', MAX_EPOCH, train_elapsed))

    return fx[1]
end



function test(model, dataset)

    -- time measurement
    local start = sys.clock()

    -- confusion matrix to hold test accuracy
    local classes = torch.linspace(0,9,10):totable() -- {0,1,2,3,4,5,6,7,8,9}
    local confusion = optim.ConfusionMatrix(classes)

    for t = 1,dataset.size do

        -- get data
        local input = dataset[t].x
        local target = oneHotDecoding(dataset[t].y)

        -- forward pass
        local pred = model:forward(input)
        pred = oneHotDecoding(pred)

        -- update confusion matrix
        confusion:add(pred+1, target+1) -- because of the lua 1-indexing scheme
        
    end

    -- accuracy info
    acc = confusion.totalValid * 100
    image.display(confusion:render())
    print(string.format('Classification accuracy: %f', acc))

    -- time measurement
    elapsed = sys.clock() - start
    print(string.format('Total elapsed testing time: %f', elapsed))

    return 1-acc
end



function main()

    -- coded like this by now. TODO: parametrize and generalize
    word_embedding_size = 300
    sentence_size = 6
    
    -- logging variables
    save_path = '/users/josemarcos/Desktop/convNet/models'
    trainLogger = optim.Logger(paths.concat(save_path, 'train.log'))
    testLogger = optim.Logger(paths.concat(save_path, 'test.log'))

    -- model
    model = create_cnn_model()
    criterion = nn.MSECriterion()

    -- datasets
    train_data, test_data = preProcessData(getData())
    train_err = 9999
    test_err = 9999


    while train_err > 0.01 do
        -- train/test
        train_err = train(model, criterion, train_data)
        test_err = test(model, test_data)

        print(string.format('Training error=%f',train_err))
        print(string.format('Test error=%f', test_err))

        -- plot evolution
        trainLogger:style{['% mean class accuracy (train set)'] = '-'}
        testLogger:style{['% mean class accuracy (test set)'] = '-'}
        trainLogger:plot()
        testLogger:plot()
    end
end


-- main()

model = torch.load('models/convNet_e200.net')
train_data, test_data = preProcessData(getData())

test(model, test_data)

