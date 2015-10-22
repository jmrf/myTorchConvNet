require 'cutorch'
require 'optim'
require 'paths'
require 'image'
require 'cunn'

-- needed in order to make all the layers work!
torch.setdefaulttensortype('torch.FloatTensor')

-- some global vars to make things easier
classes = torch.linspace(0,9,10):totable() -- {0,1,2,3,4,5,6,7,8,9}
num_classes = #classes
data_format = 'table'
cuda_enabled = false


--[[ 
    ----------------------------------------------------------------------------------------------- 
    -------------------------------------- DATA FUNCTIONS -----------------------------------------
    -----------------------------------------------------------------------------------------------
-- ]]

function getData()
    -- only for debugging porposes
    local mnist = require 'mnist'
    local trainset = mnist.traindataset()
    local testset = mnist.testdataset()
    return trainset, testset
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


function preProcessData_table(trainset, testset)

    -- transofrm this to Tensor (x + oneHot(y)) x trainset.size and compare performace on gpus vs cpu and optimal batch size.
    print('Training instances:',trainset.size)
    print('Testing instances:',testset.size)

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



function preProcessData_tensor(trainset, testset)

    print('Training instances:',trainset.size)
    print('Testing instances:',testset.size)

    -- preprocess all the data in a Tensor format, so sending chunks forward through the network is easier.
    local img_size = trainset[1].x:size()
    local trainset_t = torch.FloatTensor(trainset.size,img_size[1]*img_size[2] + num_classes)
    local testset_t = torch.FloatTensor(testset.size,img_size[1]*img_size[2] + num_classes)
    -- train set: Tensorification + data standarization
    for i=1,trainset.size do
        local label_tensor = oneHotEncoding(trainset[i].y):reshape(1,num_classes)
        local norm_img = (trainset[i].x:float() - torch.Tensor(img_size[1],img_size[2]):fill(128))/255
        trainset_t[{i,{}}] = norm_img:reshape(1,img_size[1]*img_size[2]):cat(label_tensor)
    end
    -- test set: Tensorification + data standarization
    for i=1,testset.size do
        local label_tensor = oneHotEncoding(testset[i].y):reshape(1,num_classes)
        local norm_img = (testset[i].x:float() - torch.Tensor(img_size[1],img_size[2]):fill(128))/255
        testset_t[{i,{}}] = norm_img:reshape(1,img_size[1]*img_size[2]):cat(label_tensor)
    end
    return trainset_t, testset_t
end


function getMiniBatch(dataset,batch_size, batch_num, b)
    -- dirty way of doing, just for testing purposes....
    -- notice the 3rd returnning value, actually acts the control of the loop:
    -- if we do it in the tensorized way we only perform 1 pass of the loop (all data forwarded to the network at once)
    local inputs
    local targets
    if data_format == 'table' then
        -- provide an input at a time
        inputs = dataset[batch_num*batch_size + b].x
        targets = dataset[batch_num*batch_size + b].y
        return inputs, targets, b+1

    elseif data_format == 'tensor' then
        -- provide a tensor'd form to provide many inputs at a time
        local ini = batch_num*batch_size + 1
        local fin = ini + batch_size - 1
        inputs = dataset[{{ini,fin},{1,-num_classes-1}}]
        targets = dataset[{{ini,fin},{-num_classes,-1}}]
        return inputs, targets, batch_size+1
    else
        print('Invalid data format detected...')
    end

end





--[[ 
    ----------------------------------------------------------------------------------------------- 
    ----------------------------- MODEL & TRAINING/TESTING FUNCTIONS ------------------------------
    -----------------------------------------------------------------------------------------------
-- ]]


function create_cnn_model()
    -- building the network:
    local model = nn.Sequential()
    model:add(nn.Reshape(1,28,28))
    -- layer 1:
    model:add(nn.SpatialConvolution(1,4,5,5))          -- 4 x 24 x 24
    model:add(nn.Tanh())
    model:add(nn.SpatialAveragePooling(2,2,2,2))        -- 4 x 12x 12
    -- layer 2:
    model:add(nn.SpatialConvolution(4,16,5,5))        -- 16 x 8 x 8
    model:add(nn.Tanh())
    model:add(nn.SpatialAveragePooling(2,2,2,2))        -- 16 x 4 x 4       
    -- layer 3 (2 fully connected neural nets):
    model:add(nn.Reshape(16*4*4))
    model:add(nn.Linear(16*4*4,100))
    model:add(nn.Tanh())
    model:add(nn.Linear(100,10))
    model:add(nn.SoftMax())
    
    return model
end


function train(model, criterion, dataset)

    print('\n---------------- TRAIN ----------------\n')

    -- training conf
    local MAX_EPOCH = 200
    local coefL1 = 0
    local coefL2 = 0.01
    local batch_size

    if data_format == 'table' then
        batch_size = dataset.size/MAX_EPOCH     -- 1 epoch is a complete pass through all the data...
    elseif data_format == 'tensor' then
        batch_size = dataset:size()[1]/MAX_EPOCH
    else
        print('Data format not recognized....')
    end

    -- printing some info before starting the training
    print('Total number of epochs: ', MAX_EPOCH)
    print('Batch size:', batch_size)
    print('Data format:', data_format)
    print('Using GPU:', cuda_enabled)

    -- Stochastic Gradient Descent configuration
    local sgd_config = {
        learningRate = 0.1,
        learningRateDecay = 5.0e-6,
        momentum = 0.9,
    }


    local train_start = sys.clock()

    -- retrieve model parameters and gradients (init with random weights)
    parameters,gradParameters  = model:getParameters()
    parameters:uniform(-0.1,0.1)

    -- train for the specified number of epochs
    batch_num = 0

    for epoch = 1,MAX_EPOCH do

        -- display progress (only works if using console)
        xlua.progress(epoch, MAX_EPOCH)

        -- time measurement
        local start = sys.clock()

        -- create closure to evaluate f(x) and df/dx
        local feval = function(x)

            -- get new parameters
            if x ~= parameters then
                parameters:copy(x)
            end

            -- reset params
            local b = 1
            local f = 0
            gradParameters:zero()

            
            while b <= batch_size do

                -- get minibatch (remove sending to GPU from training function --> encapsulate elsewhere)
                inputs, targets,b = getMiniBatch(dataset, batch_size, batch_num, b)
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
                if cuda_enabled then
                    dl_do = dl_do:cuda()
                end
                model:backward(inputs, dl_do)
            end

            -- normalize by the batch size
            gradParameters:div(batch_size)
            f = f/batch_size
            return f,gradParameters

        end

        -- optimization step SGD
        x,fx = optim.sgd(feval, parameters, sgd_config)
        batch_num = batch_num + 1

        -- local elapsed = sys.clock() - start
        -- print(string.format('Elapsed time: %f', elapsed))

    end

    local train_elapsed = sys.clock() - train_start
    print(string.format('Total elapsed training time for %i epochs: %f', MAX_EPOCH, train_elapsed))

    return fx[1]
end



function test(model, dataset)

    print('\n---------------- TEST ----------------\n')

    -- time measurement
    local start = sys.clock()

    -- confusion matrix to hold test accuracy
    local confusion = torch.zeros(num_classes, num_classes)
    -- local confusion = optim.ConfusionMatrix(classes)

    local dataset_size = 0
    if data_format == 'table' then
        dataset_size = dataset.size
    elseif data_format == 'tensor' then 
        dataset_size = dataset:size()[1]
    else
        print('Data format not recognized...')
    end


    for t = 1,dataset_size do

        -- get datasets
        local inputs
        local targets
        if data_format == 'table' then
            inputs = dataset[t].x:float()
            targets = oneHotDecoding(dataset[t].y)
        elseif data_format == 'tensor' then
            inputs = dataset[{t,{1,-num_classes-1}}]
            targets = oneHotDecoding(dataset[{t,{-num_classes,-1}}])
        else
            print("Data format not recognized")
        end

        if cuda_enabled then
            inputs = inputs:cuda()
        end

        -- forward pass
        local pred = model:forward(inputs)
        if cuda_enabled then pred = pred:float() end
        pred = oneHotDecoding(pred)

        -- update confusion matrix
        confusion[pred+1][targets+1] = confusion[pred+1][targets+1] + 1
        -- confusion:add(pred+1, targets+1) -- because of the lua 1-indexing scheme
        
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
    print('Precission:')
    print(precission:reshape(1,10))
    print('Recall:')
    print(recall:reshape(1,10))
    print('Fscore:')
    print(f_score:reshape(1,10))

    --monitorimage.display(confusion:render())

    -- time measurement
    elapsed = sys.clock() - start
    print(string.format('Total elapsed testing time: %f', elapsed))

    return 1-acc
end





--[[ 
    ----------------------------------------------------------------------------------------------- 
    ---------------------------------------- MAIN FUNCTIONS ---------------------------------------
    -----------------------------------------------------------------------------------------------
-- ]]

function main()
    
    -- logging variables
    save_path = '/users/josemarcos/Desktop/convNet/models'
    trainLogger = optim.Logger(paths.concat(save_path, 'train.log'))
    testLogger = optim.Logger(paths.concat(save_path, 'test.log'))

    -- model
    model = create_cnn_model()
    criterion = nn.MSECriterion()

    -- datasets
    if data_format == 'table' then
        train_data, test_data = preProcessData_table(getData())
    elseif data_format == 'tensor' then
        train_data, test_data = preProcessData_tensor(getData())
    else
        print('Data format not recognized....')
    end

    train_err = 1/0
    test_err = 1/0


    -- move everuthing to GPU to speed up
    if cuda_enabled then
        model:cuda()
        criterion:cuda()
    end

    round = 1
    while train_err > 0.01 do
        -- train stage
        train_err = train(model, criterion, train_data)
        print('Training error=',train_err)

        -- saving the model
        local ext = '.net'
        if cuda_enabled then 
            ext = '_cuda' .. ext
        end
        local filename = paths.concat(save_path, 'convNet_r' .. tostring(round) .. '_' .. data_format .. ext)
        os.execute('mkdir -p ' .. sys.dirname(filename))
        torch.save(filename, model)

        -- test stage
        test_err = test(model, test_data)
        print('Test error=', test_err)


        -- plot evolution
        trainLogger:style{['% mean class accuracy (train set)'] = '-'}
        testLogger:style{['% mean class accuracy (test set)'] = '-'}
        trainLogger:plot()
        testLogger:plot()

        round = round + 1
    end
end


main()

-- model = torch.load('models/convNet_r1_tensor.net')
-- train_data, test_data = preProcessData_tensor(getData())
-- test(model, test_data)
