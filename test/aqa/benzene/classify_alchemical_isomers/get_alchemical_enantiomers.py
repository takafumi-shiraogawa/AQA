from aqa.alch_calc_utils import get_alch_isomers_indexes

# Set parameters
path_xyz_file = "./benzene.xyz"


alchemical_isomers = get_alch_isomers_indexes(path_xyz_file)
print(len(alchemical_isomers))
for i in alchemical_isomers:
    print("size", len(i))
    for j in i:
        print(j)

alchemical_isomers = get_alch_isomers_indexes(path_xyz_file, mutation_sites=list(range(6)))
print(len(alchemical_isomers))
for i in alchemical_isomers:
    print("size", len(i))
    for j in i:
        print(j)

alchemical_isomers = get_alch_isomers_indexes(path_xyz_file, mutation_sites=list(range(6)), distinguish_isomers=False)
print(len(alchemical_isomers))
for i in alchemical_isomers:
    print("size", len(i))
    for j in i:
        print(j)
