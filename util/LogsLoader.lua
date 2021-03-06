--
-- Created by IntelliJ IDEA.
-- User: bohdanmaksak
-- Date: 07/08/2015
-- Time: 15:48
-- To change this template use File | Settings | File Templates.
-- MAJOR TODO: Need to train the word vectors on a data set which is OK to use for commercial purposes Currently using pre-trained word vectors from Google: https://code.google.com/p/word2vec/
-- TODO: Also check the entity vectors on Freebase names: https://code.google.com/p/word2vec/#Pre-trained_entity_vectors_with_Freebase_naming
-- TODO: Make a foreign key consraint in logs table on classes table.


local LogsLoader = {}
local print_some_batches = false


LogsLoader.__index = LogsLoader


function LogsLoader.create(split_fractions, batch_size)

    print("Creating LogsLoader")

    local w2vutils = require 'util.w2vutils'

    local self = {}
    setmetatable(self, LogsLoader)

    
    -- DB credentials and hostname
    local db_name = 'comcastlogs'
    local username = 'bogdan'
    local password = '42genius'
    local host_name = 'comcastlogs.csh4igez7mgq.eu-west-1.rds.amazonaws.com'
    local logs_table_name = 'logs'
    local classes_table_name = 'classes'

    -- Connect to the DB.
    local driver = require 'luasql.mysql'
    local env = driver.mysql()
    local conn = env:connect(db_name, username, password, host_name)

    -- data splitting fractions
    local split_fractions = split_fractions or  {0.8, 0.2, 0 } -- train, validation and test splits.
    if (batch_size ~= 1) then
        print("ERROR: the batch_size is not 1. The current version of the code is only ready for batch_size = 1 :( ")
        os.exit()
    end

    math.randomseed(os.time())

    -- logging variables variables
    local print_logs_and_word_vectors = false
    local print_all_data_tensors = false
    local print_train_val_sets = false
    local print_all_data_table  = false


    --[[
    -- If we need to create a table use the below:
    --
    -- conn:execute("CREATE TABLE logs (id INT NOT NULL AUTO_INCREMENT, questions TEXT NOT NULL, answers TEXT, classes INT NOT NULL,
    -- PRIMARY KEY(id), FULLTEXT(answers), FULLTEXT(questions)) ENGINE=MyISAM ")
    --
    -- use MyISAM engine for FULLTEXT indexing. Might be needed for search.
    --
    -- An insert would look like this:
    --
    -- conn:execute("INSERT INTO logs(questions, answers, classes) VALUES ('whats the top speed?', 'The top speed of i3 is 100 miles per hour', 12)")
    --
    -- ]]--

    

    -- NOTE: be carefull, all the results returned by the mysql driver are of string type! So don't forget to convert explicitly when appropriate.
    conn:execute("SET sql_mode = 'STRICT_ALL_TABLES'") -- make sure the strict mode is on.
    local cursor_logs = conn:execute("SELECT * FROM " .. logs_table_name)
    local cursor_classes = conn:execute("SELECT * FROM " .. classes_table_name)


    local all_logs = {}     -- will contain all the rows from the logs table.
    self.all_classes = {}   -- will contain all the rows from classes table.

    -- some data sizes
    self.num_of_classes = cursor_classes:numrows()
    self.num_of_examples = cursor_logs:numrows()

    self.word_dim = 300
    local highest_num_of_words = 20 -- the highest number of words in a sentence. TODO: dont think it's used, check if can be removed.

    -- so we create orig2rep_class_map that contains original class ids as keys and representation classes as values
    self.orig2rep_class_map = {} -- in case some classes have been ommited, e.g. if we have classes like: 1, 2, 10, 11, 12, etc.
    for i=1, self.num_of_classes do
        local row_table = {}
        cursor_classes:fetch(row_table, 'a') -- use 'a' option to save the values with keys being names of columns or alternatively use 'n' for numerical indices.
        table.insert(self.all_classes, row_table)
        self.orig2rep_class_map[tonumber(row_table.id)] = i
    end

    print("\n\nAll classes table:")
    for i, v in ipairs(self.all_classes) do
        print(i, v)
    end

    print("\n\norig2rep Class Map:")
    for i, v in pairs(self.orig2rep_class_map) do
        print(i, v)
    end


    local function getClassTensor(c)
        local t = torch.Tensor(self.num_of_classes):fill(0)
        t[self.orig2rep_class_map[c]] = 1
        return t
    end

    -- get all the logs(questions and their classes/answers)
    for i=1, self.num_of_examples do
        local row_table = {}
        cursor_logs:fetch(row_table, 'a')
        all_logs[i] = row_table
    end

    if (print_logs_and_word_vectors) then
        print("\nPrinting questions")
    end

        -- local all_data = torch.Tensor(num_of_examples, num_of_classes + self.word_dim * highest_num_of_words):fill(0)

    self.all_data = {}       -- this table will contain all the matrices of different length.
    local classes_freq = {} -- will contain classes as keys and number of occurences of a class as values.


    -- iterate through all the logs
    for k, v in ipairs(all_logs) do

        -- print questions
        if print_logs_and_word_vectors then
            print(k)
        end

        -- classes frequency count
        local question = v.questions
        local q_class = tonumber(v.classes)

        if not classes_freq[q_class] then 
            classes_freq[q_class] = 1
        else 
            classes_freq[q_class] = classes_freq[q_class] + 1
        end

        if print_logs_and_word_vectors then
            print("\nQuestion is: " .. question .. "\n")
        end


        local i = 0
        for w in string.gmatch(question, "%a+") do
            -- word count
            i = i + 1
        end

        local logTensor = torch.Tensor(1 + self.word_dim*i):fill(0) -- we need the first element to indicate which class the question belongs to.
        logTensor[1] = self.orig2rep_class_map[q_class]             -- set the first element to the correct class.

        local sentenceTensor  = torch.Tensor(i, self.word_dim):fill(0)
        
        i = 0
        for w in string.gmatch(question, "%a+") do
            i = i + 1

            -- word to vector transform
            local vec = w2vutils:findVector(w)

            -- copy the word vectorized representation to the sentece sub-tensor
            local vecCopy = logTensor:sub((i-1)*300 + 2, i*300 + 1)
            vecCopy:copy(vec)

            if print_logs_and_word_vectors then
                print ("\n" .. w)
                print("\nThe vector of the word is: ")
                print (vec)
            end
            sentenceTensor[i] = vec
        end

        if not self.all_data[i] then
            self.all_data[i] = {}
            table.insert(self.all_data[i], logTensor)
        else

            table.insert(self.all_data[i], logTensor)
        end


        -- print("\n\n\nSentence vector: \n")
        -- print(tostring(sentenceTensor)) TODO: remove sentenceTensor, no longer used.
        if (print_logs_and_word_vectors) then
            print("\n\n\nLog vector: \n")
            print(tostring(logTensor))
        end
    end


    local function shuffleTable (t)
        assert(t, "expected a table, got nil")

        -- swap every row with any other random one
        for i = #t, 2, -1 do
            local j = math.random(i)
            t[i], t[j] = t[j], t[i]
        end
    end



    -- Now let's shuffle the order of tensors inside each sequence array
    for k, v in pairs(self.all_data) do
        shuffleTable(v)
    end

    if print_all_data_table then
        print("\n\nPrinting self.all_data table: \n")
        for k, v in pairs(self.all_data) do
            print(k, v)
            --  print(#v, v[1]:size(1))
        end
    end




    -- NOTE: at the moment we are not using the below.
    -- Here we construct tensors(matrices) for each i in the self.all_data table. So that all questions of the same legnth are in 1 tensor.

    local all_data_tensors = {}

    for k, v in pairs(self.all_data) do

        local t = torch.Tensor(#v, v[1]:size(1))
        for m, n in ipairs(v) do
            t[m] = n
        end

        all_data_tensors[k] = t

    end

    if (print_all_data_tensors) then
        for k, v in pairs(all_data_tensors) do
            print(k, v)
        end
    end


    -- At this point we would need to split the data-set into batches. Let's aim to keep the batch_size fixed within every sequence legnth. For now,
    -- let's see what happens if we keep batch_size at 1 because we dont have much data to split it into many batches.

    -- when batch_size is 1.
    local n_batches = self.num_of_examples

    if n_batches < 50 then
        print('WARNING: less than 50 batches in the data in total? Looks like very small dataset. You probably want to use smaller batch_size and/or seq_length.')
    end

    -- Now let's split the data (self.all_data table) into train, val, test sets.
    -- perform safety checks on split_fractions, in other words check if all the fractoins are between 0 and 1.
    local sum = 0
    for i=1,3 do
        assert(split_fractions[i] >= 0 and split_fractions[i <= 1, 'bad split fraction ' .. split_fractions[i] .. ' for train, not between 0 and 1')
        sum = sum + split_fractions[i]
    end
    assert(sum > 1, 'bad split fractions, should at most sum up to 1...')



    self.batch_ix = {0, 0, 0}
    self.train_set = {}
    self.val_set = {}

    -- esto puede petar si split_fraction[1] (train) == 0 por culpa del repeat
    for k, v in pairs(self.all_data) do

        local total_n  = #v
        local train_n = math.floor(total_n*split_fractions[1])

        local i = 1
        repeat
            table.insert(self.train_set, v[i])
            i = i + 1
        until i > train_n

        if i <= total_n then
            repeat
                table.insert(self.val_set, v[i])
                i = i + 1
            until i > total_n
        end
    end

    self.ntrain  = #(self.train_set)
    self.nval = #(self.val_set)

    self.ntest = 0 -- TODO: allow for test.

    self.split_sizes = {self.ntrain, self.nval, self.ntest}

    print(string.format(' Number of data batches in train: %d, val: %d, test: %d', self.ntrain, self.nval, self.ntest))

    if print_train_val_sets then
        print("\n\nPrinting train set: ")
        for i, v in ipairs(self.train_set) do
            print(i, v)
        end

        print("\n\nPrinting validation set: ")
        for i, v in ipairs(self.val_set) do
            print(i, v)
        end
    end




    ------------------------------------------ Showing data distribution ------------------------------------------
    -- Let's sort the all_classes table by number of occurences, so we print it and see how questions are ditributed among classes.

    print("\nClasses un-sorted: ")
    for i, k in pairs(classes_freq) do
        print(i .. ': ' .. k)
    end

    -- creating an iterator function to be used for sorting.
    local function spairs(t, order) -- takes in arguments t - a table to sort and order - function for sorting.

    -- put all the keys into an array
    local keys = {}
    for k in pairs(t) do 
        keys[#keys + 1] = k 
    end 

    if order then
        table.sort(keys, function(a, b) return order(t, a, b) end)
    else
        table.sort(keys)
    end

    -- return the iterator function
    local i = 0
    return function ()
        i = i + 1
        if keys[i] then
            return keys[i], t[keys[i]]
        end
    end

    end

    print("\nSorted classes: ")

    for k, v in spairs(classes_freq, function (t, a, b) return t[b] < t[a] end) do
        print(k, v)
    end
    ---------------------------------------- Done showing data distribution


    function self:reset_batch_pointer(split_index, batch_index)
        batch_index = batch_index or 0
        self.batch_ix[split_index] =  batch_index
    end

    function self:next_batch(split_index)

        if self.split_sizes[split_index] == 0 then
            local split_names = {'train', 'val', 'test' }
            print('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
            os.exit() -- crash violently
        end

        self.batch_ix[split_index] = self.batch_ix[split_index] + 1
        if self.batch_ix[split_index] > self.split_sizes[split_index] then
            self.batch_ix[split_index] = 1 -- cycle around to beginning when we reach the end of the set.
        end

        local ix = self.batch_ix[split_index]

        if split_index == 1 then
            return self.train_set[ix]:sub(2, self.train_set[ix]:size(1)), self.train_set[ix][1]
        end


        if split_index == 2 then
            return self.val_set[ix]:sub(2, self.val_set[ix]:size(1)), self.val_set[ix][1]
        end

    end

    collectgarbage()
    return self
end



if print_some_batches then

    local logsLoader = LogsLoader.create({0.8, 0.2, 0}, 1)

    for i = 1,20 do

        print("\n\n\nTranining batch: ")
        local a, b = logsLoader:next_batch(1)

        print("Class: " .. tostring(b) .. "\n")
        print("Question tensor: \n" .. tostring(a) .. "\n" )
        print("\n\n\nValidation batch: ")

        a, b = logsLoader:next_batch(2)
        print("Class: " .. tostring(b) .. "\n")
        print("Question tensor: " .. tostring(a) .. "\n")
    end

end

return LogsLoader


































