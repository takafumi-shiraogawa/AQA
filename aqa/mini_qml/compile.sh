# Please check possible compiler options for your system by using
# f2py -c --help-fcompiler
# TODO: check my ifort environment
# ifort
# f2py -m frepresentations -c frepresentations.f90 --fcompiler=intelem --opt="-O3" --f90flags="-qopenmp"
# gfortran
f2py -m frepresentations -c frepresentations.f90 --fcompiler=gfortran --opt="-O3" --f90flags="-fopenmp"