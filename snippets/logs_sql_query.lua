require 'torch'


local driver = require 'luasql.mysql'


-- DB credentials
local db_name = 'comcastlogs'
local username = 'bogdan'
local password = '42genius'
local host_name = 'comcastlogs.csh4igez7mgq.eu-west-1.rds.amazonaws.com'
local logs_table_name = 'customnet_dialog'
local classes_table_name = 'customnet_classes'

-- Connect to the DB.
local env = driver.mysql()
local conn = env:connect(db_name, username, password, host_name)

-- init table cursors
conn:execute("SET sql_mode = 'STRICT_ALL_TABLES'") 
local cursor_logs = conn:execute("SELECT * FROM " .. logs_table_name .. " where answer1 is not null and class1 is not null")
local cursor_classes = conn:execute("SELECT * FROM " .. classes_table_name)

all_classes = {} -- will contain all the rows from classes table.
all_logs = {} -- will contain all the rows from the logs table.

num_of_classes = cursor_classes:numrows()
num_of_examples = cursor_logs:numrows()

for i =1,5 do
    local row_table = {}
    cursor_classes:fetch(row_table, 'a') -- use 'a' option to save the values with keys being names of columns or alternatively use 'n' for numerical indices.
    table.insert(all_classes, row_table)

    print(row_table)
end