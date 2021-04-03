# This file runs the EM scheme and saves outputs of each round in a new directory
# Along the way table/maps for each round can be generated to compare the process

import id_gen
import M_step
import table_map

output_dir = './EM_rounds/'

train_slides=[4,5,11,12,22,24]
test_list_internal = id_gen._test(train_slides, 'patches2x')

test_slides=list(set(list(range(27)))-set(train_slides))
test_list_external = id_gen._test(test_slides, 'patches2x')

for it_n in range(6):
    if it_n==0:
        print(f'iteration = {it_n}')
        round_out_dir = output_dir+'round_'+ str(it_n) + '/'
        train_list, val_list = id_gen._random(150, train_slides, 'patches2x')
        M_step.CNN_train_round_0( round_out_dir, it_n, 'patches2x', train_list, val_list, test_list_internal, 3, 1)
        M_step.CNN_external_test( round_out_dir+'external/','patches2x', test_list_external, round_out_dir + 'model_round_' + str(it_n) + '.pth')
        table_map._table_prob(round_out_dir+'external/')

    if it_n>=1:
        print(f'iteration = {it_n}')
        round_in_dir = output_dir+'round_'+ str(it_n-1) + '/'
        round_out_dir = output_dir+'round_'+ str(it_n) + '/'
        top_train_ids, top_val_ids = id_gen._top(250,50, round_in_dir + 'imgs_round_' + str(it_n-1) + '.npy', round_in_dir + 'prbs_round_' + str(it_n-1) + '.npy', 'patches2x')
        M_step.CNN_train_round_n(round_out_dir, it_n, 'patches2x', top_train_ids, top_val_ids, test_list_internal, round_in_dir + 'model_round_' + str(it_n-1) + '.pth', 1)
        M_step.CNN_external_test( round_out_dir+'external/','patches2x', test_list_external, round_out_dir + 'model_round_' + str(it_n) + '.pth')
        table_map._table_prob(round_out_dir+'external/')

