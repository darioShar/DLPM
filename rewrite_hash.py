from bem.utils_exp import FileHandler, print_dict
import dlpm.dlpm_experiment as dlpm_exp
import pathlib

dataset='cifar10_lt'
method='dlpm'
folder='./models/paper_ddpm/{}/'.format(dataset)

FH = FileHandler(exp_hash=dlpm_exp.exp_hash)

L = FH.get_params_from_folder(folder)

# check that all elements in L have the same parameters appart from alpha
print_dict(L[0][1])
for i in range(1, len(L)):
    for x in L[i][1].keys():
        if (x == 'run') or (x == 'eval') or (x == 'training') or (x == 'optim'):
            continue
        if type(L[i][1][x]) == dict:
            for y in L[i][1][x].keys():
                if y != 'alpha':
                    assert L[i][1][x][y] == L[0][1][x][y], 'Parameters are not the same for all files in the folder: at 1st key = {}, 2nd key = {}: {} != {}'.format(x, y, L[i][1][x][y], L[0][1][x][y])
        else:
            assert L[i][1][x] == L[0][1][x], 'Parameters are not the same for all files in the folder: at key = {}: {} != {}'.format(x, L[i][1][x], L[0][1][x])

print('parameters are the same for all files in the folder, except for alpha')


# Consider the config file matches the one used in the training. Just join with alpha 

p = FH.get_param_from_config('dlpm/configs/', '{}.yml'.format(dataset))
p['method'] = method

for file_name, parameter in L:
    # update parameter before computing corresponding new hash
    p[method]['alpha'] = parameter['diffusion']['alpha']
    
    
    new_hash = FH.get_exp_hash(p)
    old_hash = str(file_name).split('_')[-1][:-3]
    print('Old hash: {}, New hash: {}'.format(old_hash, new_hash))
    # use pathlib to rename files in the folder, replacing old_hash with new_hash. use glob to match names
    for file in pathlib.Path(folder).glob('*{}*'.format(old_hash)):
        new_name = pathlib.Path(str(file).replace(old_hash, new_hash))
        file.rename(new_name)
        print('Renamed {} to {}'.format(file, new_name))
    
