--[[command line arguments]]--
require 'dp'


cmd = torch.CmdLine()

cmd:text()
cmd:text('Image Classification using MLP Training/Optimization')
cmd:text('Example:')
cmd:text('$> th neuralnetwork.lua --batchSize 128 --momentum 0.5')

cmd:text('Options:')

cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--schedule', '{[200]=0.01, [400] = 0.001}', 'learning rate schedule')
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--momentum', 0, 'momentum')
cmd:option('--hiddenSize', '{200,200}', 'number of hidden units per layer')
cmd:option('--batchSize', 32, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find out a better local minima for early-stopping')
cmd:option('--dropout', false, 'apply dropout on hidden neurons')
cmd:option('--batchNorm', false, 'use batch normalization. dropout is mostly redundant with this')
cmd:option('--dataset', 'Mnist', 'which dataset to use : Mnist | NotMnist | Cifar10 | Cifar100')
cmd:option('--standardize', false, 'apply Standardize preprocessing')
cmd:option('--zca', false, 'apply Zero-Component Analysis whitening')
cmd:option('--progress', false, 'display progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')
cmd:text()

opt = cmd:parse(arg or {})

opt.schedule = dp.returnString(opt.schedule)
opt.hiddenSize = dp.returnString(opt.hiddenSize)

if not opt.silent then
table.print(opt)
end

local input_preprocess = {} 

if opt.standardize then 
 table.insert(input_preprocess, dp.Standardize())
end

if opt.zca then 
 table.insert(input_preprocess, dp.ZCA())
end

if opt.lecunlcn then
 table.insert(input_preprocess, dp.GCN())
 table.insert(input_preprocess, dp.LeCunLCN{progress=true})
end

if opt.dataset == 'Mnist' then
 ds = dp.Mnist{input_preprocess = input_preprocess}

elseif opt.dataset == 'NotMnist' then
 ds = dp.NotMnist{input_preprocess = input_preprocess}

elseif opt.dataset == 'Cifar10' then
 ds = dp.Cifar10{input_preprocess = input_preprocess}

elseif opt.dataset == 'Cifar100' then 
 ds = dp.Cifar100{input_preprocess = input_preprocess}

else
 error("Unknown Dataset")
end


--[[Model]]--

model = nn.Sequential()
model:add(nn.Convert(ds:ioShapes(), 'bf') )

--hidden Layers

inputSize = ds:featureSize()
for i,hiddenSize in ipairs(opt.hiddenSize) do

 model:add(nn.Linear(inputSize, hiddenSize)) -- weights/parameters
 
 if opt.batchNorm then
  model:add(nn.BatchNormalization(hiddenSize))
 end
 
 model:add(nn.Tanh())
 
 if opt.dropout then
  model:add(nn.Dropout())
 end

 inputSize = hiddenSize

end


-- output Layer

model:add(nn.Linear(inputSize, #(ds:classes())))
model:add(nn.LogSoftMax())


--[[Propagators]]--


train = dp.Optimizer{
 acc_update = opt.accUpdate, 
 loss = nn.ModuleCriterion(nn.ClassNLLCriterion(), nil, nn.Convert()),

 callback = function(model, report)
   opt.learningRate = opt.schedule[report.epoch] or opt.learningRate

   if opt.accUpdate then

      model:accUpdateGradParameters(model.dpnn_input, model.output, opt.learningRate)
   else 
      model:updateGradParameters(opt.momentum) -- affects gradParams
      model:udpateParameters(opt.learningRate) -- affects params
   end
   
   model:maxParamNorm(opt.maxOutNorm) -- affects params
   model:zeroGradParameters() -- affects gradParams
  end, 
  
  feedback = dp.Confusion(), 

  sampler = dp.ShuffleSampler{batch_size = opt.batchSize}, 

  progress = opt.progress

}

