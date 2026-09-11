# files to read
old_train_path = "__train.txt"
old_test_path = "__test.txt"
old_val_path = "__val.txt"

# file to generate
all_item_path = "__all_item.txt"

all_item = []

with open(old_train_path, 'r') as f:
    for line in f:
        all_item.append(line.strip())

with open(old_test_path, 'r') as f:
    for line in f:
        all_item.append(line.strip())

with open(old_val_path, 'r') as f:
    for line in f:
        all_item.append(line.strip())

all_item.sort()

with open(all_item_path, 'w') as f:
    for line in all_item:
        f.write(line + '\n')

