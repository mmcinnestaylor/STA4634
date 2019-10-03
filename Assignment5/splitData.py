data_file = "../data/Covtype/covtype.data"
test_file = "../data/Covtype/covtype.test"
train_file = "../data/Covtype/covtype.train"

with open(data_file, 'r') as d_file, open(test_file, 'w') as te_file, open(train_file, 'w') as tr_file:
    row = ['X' + str(i) for i in range(1, 55)] + ['Y\n']
    row = ','.join(row)

    te_file.write(row)
    tr_file.write(row)

    for i in range(15120):
        line = d_file.readline()
        line = line[:-2] + 'C' + line[-2:]
        tr_file.write(line)

    for i in range(565892):
        line = d_file.readline()
        line = line[:-2] + 'C' + line[-2:]
        te_file.write(line)
