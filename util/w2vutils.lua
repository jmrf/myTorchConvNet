torch.setdefaulttensortype('torch.FloatTensor')

--[[
cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-binfilename','GoogleNews-vectors-negative300.bin','Name of the bin file.')
cmd:option('-outfilename','word2vec.t7','Name of the output t7 file.')

opt = cmd:parse(arg)
]]--

binfilename = 'GoogleNews-vectors-negative300.bin'

local web = web or false 
if web == true then 
  outfilename = '../util/word2vec.t7'
else 
  outfilename = 'util/word2vec.t7'
end
w2vutils = {}

--[[
if not paths.filep(outfilename) then
	w2vutils = require('util.bintot7') -- changed from  -- w2vutils = require('bintot7.lua') -- because with .lua would
else
	w2vutils = torch.load(outfilename)
	print('Done reading word2vec data.')
end
]]

w2vutils = torch.load(outfilename)

w2vutils.findNearest = function (self, word, k)
	local k = k or 1
	local vec = self:findVector(word)

	local returnWords = {}
	local returnDistances = {}
	returnDistances, returnWords = self:distance(vec, k)

	return returnDistances, returnWords

end




w2vutils.findVector = function (self, word)
	local word = word or 'hello'
	local i = self.w2vvocab[word] -- index of the word
	local vec = self.M[{{i}, {}}] -- the vector of the word.
	vec = vec:transpose(1, 2) -- thranspose the vector so it's n*1 dimension.
	vec = vec:select(2, 1) -- reduce the dimensionality from n*1 to just n. 1D size is needed for torch.addmv() later.
	return vec;

end


w2vutils.relationship = function (self, word1, word2, word3, k)
	local k = k or 1
	local vec1 = self:findVector(word1)
	local vec2 = self:findVector(word2)
	local vec3 = self:findVector(word3)
	local vec4 = vec1 - vec2 + vec3
	local returnDistances = {}
	local returnWords = {}
	returnDistances, returnWords = self:distance(vec4, k)
	return returnDistances, returnWords

end

w2vutils.distance = function (self,vec,k)

	-- if vec:dimensions TODO: add validation of vec's dimensions.
	local k = k or 1 -- k is the number of similar words to be found.
	self.zeros = self.zeros or torch.zeros(self.M:size(1));
	local norm = vec:norm(2)
	vec:div(norm)
	local distances = torch.addmv(self.zeros,self.M ,vec)
	distances , oldindex = torch.sort(distances,1,true)
	local returnwords = {}
	local returndistances = {}
	for i = 1,k do
		table.insert(returnwords, w2vutils.v2wvocab[oldindex[i]])
		-- print(w2vutils.v2wvocab[oldindex[i]])
		table.insert(returndistances, distances[i])
	end
	return returndistances, returnwords
end


return w2vutils
