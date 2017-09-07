require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'

local t = require '../datasets/transforms_dense'
local imagenetLabel = require './imagenet'


--data_info = torch.load('../gen/imagenet.t7')
--print(data_info)

cutorch.setDevice(1)

function scandir(directory)
    local i, t, popen = 0, {}, io.popen
    local pfile = popen('ls -a "'..directory..'"')
    for filename in pfile:lines() do
        i = i + 1
        t[i] = filename
    end
    pfile:close()
    return t
end

function TableConcat(t1,t2)
    for i=1,#t2 do
        t1[#t1+1] = t2[i]
    end
    return t1
end

function convertLinear2Conv1x1(linmodule,in_size)
   local s_in = linmodule.weight:size(2)/(in_size[1]*in_size[2])
   local s_out = linmodule.weight:size(1)
   print(s_in)
   print(s_out)
   local convmodule = cudnn.SpatialConvolution(s_in,s_out,1,1)
   convmodule.weight:copy(linmodule.weight)
   convmodule.bias:copy(linmodule.bias)
   return convmodule
end

 function largest(t)
   local maxcount = 0
   local maxindex
   for index, value in pairs(t) do
    if value > maxcount then
       maxcount = value
       maxindex = index
     end
   end
   return maxcount, maxindex
 end

-- Load the model
local model = torch.load(arg[1]):cuda()
local softMaxLayer = cudnn.SoftMax():cuda()
local scale_num = arg[2]

-- add Softmax layer
conv_layer = convertLinear2Conv1x1(model:get(model:size()), {1,1})
print(model:get(model:size()))
print(model:get(model:size()-1))
print(model:get(model:size()-2))
print('add conv')
print(model)
model:remove(model:size())
model:remove(model:size())

model:add(conv_layer:cuda())
print('added conv')
-- model:add(softMaxLayer)

-- Evaluate mode
model:evaluate()

-- The model was trained with this input normalization
local meanstd = {
   mean = { 0.5, 0.5, 0.5 },
   std = { 0.5, 0.5, 0.5 },
}

local transform = {
t.Compose{
   t.Scale(224),
   t.ColorNormalize(meanstd)
},
t.Compose{
   t.Scale(256),
   t.ColorNormalize(meanstd)
},
t.Compose{
   t.Scale(384),
   t.ColorNormalize(meanstd)
},
t.Compose{
   t.Scale(480),
   t.ColorNormalize(meanstd)
},
t.Compose{
   t.Scale(320),
   t.ColorNormalize(meanstd)
}
}

local N = 5
local f_img = io.popen('find -L /home/user/hs_db/ -iname "*.jpg"')
--local f_img = io.popen('find -L /disk1/2200 -iname "*.jpg"')
print(f_img)

result_filename = 'result_s' .. scale_num .. '_' .. arg[1] .. '.csv'
prob_filename = 'prob_s' .. scale_num .. '_' .. arg[1] .. '.csv'

print(result_filename)
local f = io.open(result_filename, "w")
cnt = 0
local f2 = io.open(prob_filename, "w")

while true do
   collectgarbage('collect')
   cnt = cnt + 1
   print(cnt)
   -- load the image as a RGB float tensor with values 0..1
   local line = f_img:read('*line')
   if not line then break end
   filename = '/home/user/hs_db/' .. paths.basename(line)

   print(filename)
   local img = image.load(filename, 3, 'float')

   local result = {}
   for j=1,1000 do
     result[j] = 0
   end
   org_img = img
	max_val = 0
--   print(org_img:size())
   for k=scale_num,scale_num do
	   -- Scale, normalize, and crop the image
	   img = transform[k](org_img)

	   -- View as mini-batch of size 1
	   local batch = img:view(1, table.unpack(img:size():totable()))

    --print(batch:size())
	   -- Get the output of the softmax
	   local output = model:forward(batch:cuda())

      -- print(output:size())
	   -- Get the top 5 class indexes and probabilities
   	sum = 0

   total_size = output:size(3) * output:size(4)

   for m=1,output:size(3) do
      for n=1,output:size(4) do
		    sum = 0
		cal_val = {}
		for j=1,output:size(2) do
-- math.exp(output[1][j][m][n])
			cal_val[j] = math.exp(output[1][j][m][n])
			sum = sum + cal_val[j]
		end

	        for j=1,output:size(2) do
        	        result[j] = result[j] + cal_val[j] / sum / total_size
	        end
	 end
	end
   end

   val, idx = largest(result)
   --print(best_row)

--   for n=1,1 do
--     print(string.gsub(paths.basename(line), ".jpg", ""), data_info.classList[idx])
--     f:write(string.gsub(paths.basename(line), ".jpg", "") .. ',' .. data_info.classList[idx] .. '\r')
--   end
   f:write("\n")

--   print(data_info.classList[idx])

   for i=1,1000 do
        if i == 1 then
	   f2:write(string.gsub(paths.basename(line), ".jpg", ""))
           f2:write(" ")
        end

        f2:write(tostring(result[i]))
        f2:write(" ")
   end
   f2:write("\n")


--   if cnt == 60000 then
--      break;
--   end
end

f:close()
f2:close()
