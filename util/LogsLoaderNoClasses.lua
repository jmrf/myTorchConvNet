--
-- Created by IntelliJ IDEA.
-- User: bohdanmaksak
-- Date: 05/09/2015
-- Time: 09:43
-- To change this template use File | Settings | File Templates.
--

-- here we load raw logs when we don't have any classification of answers.

-- frist, lets make the model work when questions come in bacthes of 1. Then we will make it work with larger bacthes of questions to speed up the training.

-- MAJOR TODO: some very important words present in the answers or questions might be completely new to the word2vec embedding model and so will be assigned UNK, but they are still very important.

local LogsLoader = {}

LogsLoader.__index = LogsLoader

function LogsLoader.create(split_fractions, batch_size, limit)

    print("Creating LogsLoader")

    local self = {}
    setmetatable(self, LogsLoader)

    local w2vutils = require 'util.w2vutils'

    -- DB credentials and hostname

    local db_name = 'comcastlogs'
    local username = 'bogdan'
    local password = '42genius'
    local host_name = 'comcastlogs.csh4igez7mgq.eu-west-1.rds.amazonaws.com'
    --local logs_table_name = 'logs'
    local logs_table_name = 'bmwlogs'
    local split_fractions = split_fractions or  {0.8, 0.2, 0 } -- train, validation and test splits.
    if (batch_size ~= 1) then
        print("ERROR: the batch_size is not 1. The current version of the code is only ready for batch_size = 1 :( ")
        os.exit()
    end

    math.randomseed(os.time())

    -- Connect to the DB.
    local driver = require 'luasql.mysql'
    local env = driver.mysql()
    local conn = env:connect(db_name, username, password, host_name)

    conn:execute("SET sql_mode = 'STRICT_ALL_TABLES'") -- make sure the strict mode is on.

    local cursor_logs = conn:execute("SELECT * FROM " .. logs_table_name)

    self.num_of_examples = limit or cursor_logs:numrows()
    self.word_dim = 300

    -- TODO: we might not need to store all_raw_logs field in the object, might be enouph to have a local variable.
    self.all_raw_logs = {}

    self.all_tensor_logs = {}

    self.all_answers = {}

    local unique_answers = {}

    for i=1, self.num_of_examples do

        local row_table = {}
        cursor_logs:fetch(row_table, 'a') -- use 'a' option to save the values with keys being names of columns or alternatively use 'n' for numerical indices.

        --[[ Print the questions and answers - useul for debugging with small datasets.
        print("Question is: " .. row_table.questions)
        print("Answer is: ".. row_table.answers)
        ]]--

        self.all_raw_logs[i] = row_table

    end

    for k, v in pairs(self.all_raw_logs) do

        -- get the question and answer pair.
        local question = v.questions
        local answer = v.answers

        -- construct tensors of sentences based on word vectors

        -- count the number of words to then initialize sentence tensors.

        local i, ia

        -- count all the words in the question
        i = 0
        for w in string.gmatch(question, "%a+") do
            i = i + 1
        end

        -- count all the words in the answer
        ia = 0
        for wa in string.gmatch(answer, "%a+") do
            ia = ia + 1
        end


        -- initialize the tensors
        local questionTensor = torch.Tensor(self.word_dim*i):fill(0)
        local answerTensor = torch.Tensor(self.word_dim*ia):fill(0)

        -- bring counters back to zero
        i = 0
        ia = 0


        -- fill in the question tensor  TODO: re-use the results in for loop above.

        for w in string.gmatch(question, "%a+") do

            i = i + 1
            -- get the word vector for the word.
            -- TODO: throw a notification when a word is not found in word2vec vocab. Plus we cant afford to ignore the words not found in vocab. They are often very important.
            local vec = w2vutils:findVector(w)
            -- select a sub-tensor of the question tensor. the sub tensor will contain embedding of current word.
            local vecCopy = questionTensor:sub((i-1)*300 + 1, i*300)
            -- copy the word embedding into the sub-tensor. Because the sub-tensor shares same underlying storage with the question tensor, the sub part of question tensor will be filled too.
            vecCopy:copy(vec)

        end


        -- fill in the answer tensor

        for wa in string.gmatch(answer, "%a+") do
            ia = ia +1
            -- get the word vector for the word.

            -- TODO: throw a notification when a word is not found in word2vec vocab. Plus we cant afford to ignore the words not found in vocab. They are often very important.
            local veca = w2vutils:findVector(wa)
            -- select a sub-tensor of the answer tensor. the sub tensor will contain embedding of current word.
            local vecCopya = answerTensor:sub((ia-1)*300 + 1, ia*300)
            -- copy the word embedding into the sub-tensor. Because the sub-tensor shares same underlying storage with the answer tensor, the sub part of answer tensor will be filled too.
            vecCopya:copy(veca)

        end

        -- self.all_answers is table containing as keys length of answers and values: arrays of answer tensors.
        -- this table will be useful to sample answer sentences for negative matching and having them organised by their length will help
        -- to construct the batches.

        local answere = {}
        answere["text"] = answer
        answere["tensor"] = answerTensor

        if not self.all_answers[ia] then
            self.all_answers[ia] = {}
            table.insert(self.all_answers[ia], answere)
            unique_answers[answer] = true
        else
            if unique_answers[answer] == nil then
                table.insert(self.all_answers[ia], answere)
                unique_answers[answer] = true
            end
        end

        -- self.all_tensor_logs table is just an array of correct question-answer pairs and their tensor representations.
        -- the questions are not grouped by their legth here.
        local tensor_entry = {}
        tensor_entry.question = question
        tensor_entry.questionTensor = questionTensor
        -- tensor_entry.answer = answer
        -- tensor_entry.answerTensor = answerTensor
        tensor_entry.answerInd = #self.all_answers[ia]
        tensor_entry.answerIa = ia
        table.insert(self.all_tensor_logs, tensor_entry)
    end

    local function shuffleTable (t)

        local rand = math.random
        assert(t, "expected a table, got nil")
        local j

        for i = #t, 2, -1 do
            j = rand(i)
            t[i], t[j] = t[j], t[i]
        end
    end

    -- shuffle the all logs table.
    shuffleTable(self.all_tensor_logs)



    -- TODO: the below assignemnt is only true when batch_size is 1. Change to accomodate various batch sizes for questions.
    local n_batches = self.num_of_examples

    if n_batches < 50 then
        print('WARNING: less than 50 batches in the data in total? Looks like very small dataset. You probably want to use smaller batch_size and/or seq_length.')
    end


    -- perform safety checks on split_fractions, in other words check if all the fractoins are between 0 and 1.
    assert(split_fractions[1] >= 0 and split_fractions[1] <= 1, 'bad split fraction ' .. split_fractions[1] .. ' for train, not between 0 and 1')
    assert(split_fractions[2] >= 0 and split_fractions[2] <= 1, 'bad split fraction ' .. split_fractions[2] .. ' for val, not between 0 and 1')
    assert(split_fractions[3] >= 0 and split_fractions[3] <= 1, 'bad split fraction ' .. split_fractions[3] .. ' for test, not between 0 and 1')



    -- batch_ix contains indeces for train, val, test sets
    self.batch_ix = {0, 0, 0}

    -- TODO: need to add code for test_set too.
    self.train_set = {}
    self.val_set = {}


    local total_n  = #(self.all_tensor_logs)

    local train_n = math.floor(total_n*split_fractions[1])

    for i, v in ipairs(self.all_tensor_logs) do
        if i <=train_n then
            table.insert(self.train_set, v)
        else
            table.insert(self.val_set, v)
        end
    end

    self.ntrain  = #(self.train_set)
    self.nval = #(self.val_set)

    self.ntest = 0 -- TODO: allow for test.

    self.split_sizes = {self.ntrain, self.nval, self.ntest}

    print(string.format(' Number of data batches in train: %d, val: %d, test: %d', self.ntrain, self.nval, self.ntest))



    -- resets the index to beginning or supplied value.
    function self:reset_batch_pointer(split_index, batch_index)
        batch_index = batch_index or 0
        self.batch_ix[split_index] =  batch_index
    end



    -- returns the next batch in a given split (train, val or test)
    function self:next_batch(split_index)


        local function getNegativeAnswers(tensor_entry)

            local answerIa = tensor_entry.answerIa
            local answerInd = tensor_entry.answerInd

            local neg_answers = {}
            for k, v in pairs(self.all_answers) do
                neg_answers[k] = {}
                if k ~= answerIa then
                    for i, v2 in ipairs(v) do
                        table.insert(neg_answers[k], v2)
                        -- TODO: check that some answers who will have different string representation, but the same vector representation, e.g. extra space in the second answer, while the actual words are the same, are not duplicated. So check answer uniqueness based on vector representation rather than string.
                    end
                else
                    for i, v2 in ipairs(v) do
                        if i ~= answerInd then
                            table.insert(neg_answers[k], v2)
                        end
                    end
                end
            end
            return neg_answers
        end

        if self.split_sizes[split_index] == 0 then
            local split_names = {'train', 'val', 'test' }
            print('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
            os.exit() -- crash violently
        end

        -- increment the index for the given split
        self.batch_ix[split_index] = self.batch_ix[split_index] + 1

        -- check if we reached the end of the split set.
        if self.batch_ix[split_index] > self.split_sizes[split_index] then
            self.batch_ix[split_index] = 1 -- cycle around to beginning when we reach the end of the set.
        end

        local ix = self.batch_ix[split_index]

        -- if it's the train split
        if split_index == 1 then

            -- return a true table; answers, which are not the same as the true answer, grouped by their length.
            local tensor_e = self.train_set[ix]
            return tensor_e

        -- if it's the val split
        elseif split_index == 2 then

            -- return a true table; and answers, which are not the same as the true answer, grouped by their length.
            local tensor_e = self.val_set[ix]
            return tensor_e
        end

    end

    collectgarbage()
    return self

end

return LogsLoader


