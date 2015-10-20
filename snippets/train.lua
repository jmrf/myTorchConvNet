
--[[

This file trains a character-level multi-layer RNN on text data

Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6
but modified to have multi-layer support, GPU support, as well as
many other common model/optimization bells and whistles.
The practical6 code is in turn based on 
https://github.com/wojciechz/learning_to_execute
which is turn based on other stuff in Torch, etc... (long lineage)

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'
local CharSplitLMMinibatchLoader = require 'util.CharSplitLMMinibatchLoader' -- getting the Character Splitter Mini Batcher from Utils file
local model_utils = require 'util.model_utils'
local LSTM = require 'model.LSTM'
local GRU = require 'model.GRU'
local RNN = require 'model.RNN'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/tinyshakespeare','data directory. Should contain the file input.txt with input data')
-- model params
cmd:option('-rnn_size', 128, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-model', 'lstm', 'lstm,gru or rnn')
-- optimization
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-seq_length',50,'number of timesteps to unroll for')
cmd:option('-batch_size',30,'number of sequences to train on in parallel')
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
            -- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',1000,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
-- train / val / test split for data, in fractions
local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac)) -- test_frac is the percentage of data allocated for testing.
local split_sizes = {opt.train_frac, opt.val_frac, test_frac} -- split_sizes contains the percentages split between train, validation and test set.



-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- initialize clnn/cltorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 1 then
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if not ok then print('package clnn not found!') end
    if not ok2 then print('package cltorch not found!') end
    if ok and ok2 then
        print('using OpenCL on GPU ' .. opt.gpuid .. '...')
        cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        torch.manualSeed(opt.seed)
    else
        print('If cltorch and clnn are installed, your OpenCL driver may be improperly configured.')
        print('Check your OpenCL driver installation, check output of clinfo command, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end



-- create the data loader class
local loader = CharSplitLMMinibatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes)
local vocab_size = loader.vocab_size  -- the number of distinct characters
local vocab = loader.vocab_mapping -- vocab is a table with characters being keys and values being indeces of characters to be used for one hot encodings.

print('vocab size: ' .. vocab_size)
-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end



-- define the model: prototypes for one timestep, then clone them in time TODO: cloning is done over fixed periods of time. we would need it over lengt of input sequence.
local do_random_init = true
if string.len(opt.init_from) > 0 then -- checking if we are loading an existing pre-trained model.
    print('loading an LSTM from checkpoint ' .. opt.init_from)  -- opt.init_from is the model name
    local checkpoint = torch.load(opt.init_from) -- loaded the checkpoit model
    protos = checkpoint.protos -- getting the prototype of the loaded model


    -- make sure the vocabs are the same
    local vocab_compatible = true
    for c,i in pairs(checkpoint.vocab) do -- vocab consists of characters and their indices.
    -- let's see what that vocab looks like

   -- print ("c is: " .. c) -- TODO remove this part: used now for debugging only
   -- print ("i is: " .. i) -- TODO remove this part: used now for debugging
        if not vocab[c] == i then -- vocab is the table with vocabularly for current dataset. i is index of of character c in the saved model
            vocab_compatible = false
        end
    end
    assert(vocab_compatible, 'error, the character vocabulary for this dataset and the one in the saved checkpoint are not the same. This is trouble.')
    -- overwrite model settings based on checkpoint to ensure compatibility
    print('overwriting rnn_size=' .. checkpoint.opt.rnn_size .. ', num_layers=' .. checkpoint.opt.num_layers .. ' based on the checkpoint.')
    opt.rnn_size = checkpoint.opt.rnn_size --  the size of the LSTM internal state. can be varied using arg or opt table at the start of this file
    opt.num_layers = checkpoint.opt.num_layers -- number of layers in the LSTM
    do_random_init = false
else
    print('creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')
    protos = {} -- protos is just a table containing modules of our model.
    if opt.model == 'lstm' then
        protos.rnn = LSTM.lstm(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
    elseif opt.model == 'gru' then
        protos.rnn = GRU.gru(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
    elseif opt.model == 'rnn' then
        protos.rnn = RNN.rnn(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
    end
    protos.criterion = nn.ClassNLLCriterion() -- TODO: we would not need this criterion to be cloned. as we only want output and criterion at the last h or in a separate decoder.
    -- Also does the criterion include softmax or is softmax part of rnn?
end

-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do -- so we do it for each layer. most cases we will have only 2.
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size) -- creating a matrix filled with zeros TODO: in our case, batch_size varies, but can be fixed for every sequence length. So we might want to set the iniial state in for loop when propagate the batch, so batch_size can be varied for each batch.
    -- of size batch_size rows and rnn_size(dimension of hidden state) columns. so each row represents a hidden state
    -- and the number of these hidden states in h_init is the number of examples in a batch.
    -- this is same dimension as x and y.
    if opt.gpuid >=0 and opt.opencl == 0 then h_init = h_init:cuda() end
    if opt.gpuid >=0 and opt.opencl == 1 then h_init = h_init:cl() end
    table.insert(init_state, h_init:clone())
    if opt.model == 'lstm' then
        table.insert(init_state, h_init:clone()) -- TODO: so if it's lstm we insert h_init twice into the init_state table, but if it was rnn or gru we dont?!
    end
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 and opt.opencl == 0 then
    for k,v in pairs(protos) do v:cuda() end
end
if opt.gpuid >= 0 and opt.opencl == 1 then
    for k,v in pairs(protos) do v:cl() end
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn) -- params are all the parameters of the lstm.
-- grad_params will probably contain the gradients of all these params.

-- initialization
if do_random_init then
params:uniform(-0.08, 0.08) -- small numbers uniform
end

print('number of parameters in the model: ' .. params:nElement())
-- make a bunch of clones after flattening, as that reallocates memory
clones = {} -- clones is just a table with a few elements: rnn, criterion. but rnn has value of the sequence of lstms. so does criterion.
for name,proto in pairs(protos) do -- in this loop we clone the prototype in time. so we unroll the lstm.
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters) -- TODO: this is where we need to make sure that we only
    -- clone for the length of the input (length of sentence, or number of words/tokens in the sentence).
    -- clones contains the model. TODO: we might want to have the final softmax not clones but only for the last sequence.
end

-- evaluate the loss over an entire split
function eval_split(split_index, max_batches)
    print('evaluating loss over split index ' .. split_index)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end -- n is the total number of batches in the split.

    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    local rnn_state = {[0] = init_state} -- rnn_state is the table having the hidden states (NOT parameters, but values of h).
    -- So we start by initializing them to zero. Since init_state is a matrix filled with zeros of size batch_size by hidden_dimension.
    


    for i = 1,n do -- iterate over batches in the split


        -- fetch a batch
        local x, y = loader:next_batch(split_index) -- get the next batch within the same split index. split_index is 1, 2, 3 for train, val, and test.
        -- x should be inputs, y  - outputs. x and y are matrices of size batch_size by sequence_size.
        -- So every row is one of the examples in a batch and every column is a time step.

        print("printing x: ")
        print (x)
        print("Size of x: ")
        print(x:size())
        print("printing y: ")
        print(y) -- same as above for x.
        print("Size of y:")
        print(y:size())
        if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
            -- have to convert to float because integers can't be cuda()'d
            x = x:float():cuda()
            y = y:float():cuda()
        end
        if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
            x = x:cl()
            y = y:cl()
        end


        -- forward pass
        for t=1,opt.seq_length do
            clones.rnn[t]:evaluate() -- for dropout proper functioning. TODO: (Evaluates the output of rnn at each time step t.)?
            local lst = clones.rnn[t]:forward{x[{{}, t}], unpack(rnn_state[t-1])} -- for each time step t make a forward pass with that character.
            -- x[k] would be the character we need but because x is a tensor that has number_of_batches * number_of_time_steps elements.
            -- so to get the characters from all the batches at time step t, we need: x[{{}, t}] this gives us a vector of size number_of_batches
            -- to do a step forward we also need to know the outputs(hidden states, memory cells) from previous step which we get from rnn_state[t-1]
            -- lst contains:
            print("at time step " .. t)
            print("x is: ")
            print(x[{{}, t}])

            print("y is: ")
            print(y[{{}, t}])
            rnn_state[t] = {}
            for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
            prediction = lst[#lst] -- TODO: we want to do it only for the last time step since we are not having softmax at
            -- each time step t
            loss = loss + clones.criterion[t]:forward(prediction, y[{{}, t}]) -- TODO: we want to do it only for the last time step since we are not having softmax at
            -- each time step t
        end
        -- carry over lstm state
        rnn_state[0] = rnn_state[#rnn_state] -- so for the next example in the batch the initial state of the lstm will be the last state of the lstm from
        -- previous example. TODO: we probably dont need it for our purposes. Since questions are kinda independant, but might work for branches though? Lets try both.
        print(i .. '/' .. n .. '...')
    end

    loss = loss / opt.seq_length / n -- TODO: need to change in light of changes above where we have only 1 softmax at the end of the lstm and so loss is
    -- not computed over sum of timesteps

    return loss
end

-- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y = loader:next_batch(1)
    if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
    end
    if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
        x = x:cl()
        y = y:cl()
    end
    ------------------- forward pass -------------------

    -- TODO: need to change the forward pass in the same way as above when we compute the predictions loss.
    local rnn_state = {[0] = init_state_global}
    local predictions = {}           -- softmax outputs
    local loss = 0
    for t=1,opt.seq_length do -- for every time step
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.rnn[t]:forward{x[{{}, t}], unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
        predictions[t] = lst[#lst] -- last element is the prediction
        loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])

        -- TODO: delete the below, used for debugging now.
        --[[print("prediction is")
        print(predictions[t])
        print ("y is ")
        print(y[{{}, t}])]]--
    end
    loss = loss / opt.seq_length -- dividing loss by the number of time steps

    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
    for t=opt.seq_length,1,-1 do -- from time step t to 1
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t}]) -- TODO: we only need it for the softmax at the end of the lstm.

        print ("\n\ny passed to backward() is: ")
        print(y[{{}, t}])
        table.insert(drnn_state[t], doutput_t) -- NOTE: here we insert doutput_t in addition to the table of deltas propagated by the layer t+1
        -- (see the for k, v pairs(dlst) loop below.
        -- TODO: we need to remove this from the loop because in our case we only need deltas coming from layer t+1 which is already performed in the loop below.
        local dlst = clones.rnn[t]:backward({x[{{}, t}], unpack(rnn_state[t-1])}, drnn_state[t]) -- TODO: since we won't have softmax at the top of each rnn at time
        -- step t, what would be our gradient inputs (drnn_state[t]) ? The gradient will be coming from the softmax classifier which is at the top of last lstm time step.
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then -- k == 1 is gradient on x, which we dont need
                -- note we do k-1 because first item is dembeddings, and then follow the 
                -- derivatives of the state, starting at index 2. I know...
                drnn_state[t-1][k-1] = v
            end
        end
    end
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params
end

-- start optimization here
train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * loader.ntrain
local iterations_per_epoch = loader.ntrain
local loss0 = nil
for i = 1, iterations do
    local epoch = i / loader.ntrain

    local timer = torch.Timer()
    local _, loss = optim.rmsprop(feval, params, optim_state)
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss

    -- exponential learning rate decay
    if i % loader.ntrain == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    -- every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_loss = eval_split(2) -- 2 = validation
        val_losses[i] = val_loss

        local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.vocab_mapping
        torch.save(savefile, checkpoint)
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
    end
   
    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    if loss0 == nil then loss0 = loss[1] end
    if loss[1] > loss0 * 3 then
        print('loss is exploding, aborting.')
        break -- halt
    end
end


