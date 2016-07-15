function pt (a)
  for k, v in pairs( a ) do
    print(k, v)
  end
end

require 'rnn'
require 'gnuplot'

gpu=1
if gpu>0 then
  print("CUDA ON")
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(gpu)
end

nIters = 2000
batchSize = 80
rho = 10
hiddenSize = 300
nIndex = 1
lr = 0.0001
nPredict=200

rnn = nn.Sequential()
   :add(nn.Linear(nIndex, hiddenSize))
   :add(nn.FastLSTM(hiddenSize, hiddenSize))
   :add(nn.NormStabilizer())
   :add(nn.Linear(hiddenSize, nIndex))
   :add(nn.HardTanh())
rnn = nn.Sequencer(rnn)
rnn:training()
print(rnn)

if gpu>0 then
  rnn=rnn:cuda()
end

criterion = nn.MSECriterion()
if gpu>0 then
  criterion=criterion:cuda()
end

ii=torch.linspace(0,200, 2000)
sequence=torch.cos(ii)
if gpu>0 then
  sequence=sequence:cuda()
end

offsets = {}
for i=1,batchSize do
   table.insert(offsets, math.ceil(math.random()* (sequence:size(1)-rho) ))
end
offsets = torch.LongTensor(offsets)
if gpu>0 then
  offsets=offsets:cuda()
end

local gradOutputsZeroed = {}
for step=1,rho do
  gradOutputsZeroed[step] = torch.zeros(batchSize,1)
  if gpu>0 then
    gradOutputsZeroed[step] = gradOutputsZeroed[step]:cuda()
  end
end

local iteration = 1
while iteration < nIters do
   local inputs, targets = {}, {}
   for step=1,rho do
      inputs[step] = sequence:index(1, offsets):view(batchSize,1)
      offsets:add(1)
      for j=1,batchSize do
         if offsets[j] > sequence:size(1) then
            offsets[j] = 1
         end
      end
      targets[step] = sequence:index(1, offsets)
   end
   rnn:zeroGradParameters()
   local outputs = rnn:forward(inputs)
   local err = criterion:forward(outputs[rho], targets[rho])
   print(string.format("Iteration %d ; NLL err = %f ", iteration, err))
   local gradOutputs = criterion:backward(outputs[rho], targets[rho])
   gradOutputsZeroed[rho] = gradOutputs
   local gradInputs = rnn:backward(inputs, gradOutputsZeroed)
   rnn:updateParameters(lr)
   iteration = iteration + 1
end


rnn:evaluate()
predict=torch.FloatTensor(nPredict)
if gpu>0 then
  predict=predict:cuda()
end
for step=1,rho do
  predict[step]= sequence[step]
end

start = {}
iteration=0
while rho + iteration < nPredict do
  for step=1,rho do
    start[step] = predict:index(1,torch.LongTensor({step+iteration})):view(1,1)
  end

  print("start")
  pt(start)

  output = rnn:forward(start)

  print("result")
  print((output[rho]:float())[1][1] )
  predict[iteration+rho+1] = (output[rho]:float())[1][1]

  iteration = iteration + 1
end

print("result")
print(predict)

gnuplot.plot({'f(x)',predict,'+-'})
