from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
from rdkit.Chem.Draw import rdMolDraw2D
from aqa.utils import read_target_molecules

Image.MAX_IMAGE_PIXELS = 1000000000

mols = []

ori_target_mols = []
target_mols = []

ori_target_mols += read_target_molecules('target_molecules.inp').tolist()
target_mols += ori_target_mols

highlight = []
for i in range(len(ori_target_mols)):
  selected_target_mol = ori_target_mols[i]

  sub = []
  for j in range(6):
    if selected_target_mol[j] != 6 and selected_target_mol[j] != 1 and selected_target_mol[j] != 'C':
      sub.append(j)

    if selected_target_mol[j] == 5:
      selected_target_mol[j] = 'B'
    elif selected_target_mol[j] == 6:
      selected_target_mol[j] = 'C'
    elif selected_target_mol[j] == 7:
      selected_target_mol[j] = 'N'

  highlight.append(sub)

  print(selected_target_mol)

  target = Chem.MolFromSmiles(
        '%s1(%s(%s(%s(%s(%s1[H])[H])[H])[H])[H])C([H])([H])[H]' % (selected_target_mol[0],selected_target_mol[1],
                                                                      selected_target_mol[2],selected_target_mol[3],
                                                                      selected_target_mol[4],selected_target_mol[5]))

  print(target)
  target_h = Chem.AddHs(target, explicitOnly=True)

  mols.append(target_h)

dopts = rdMolDraw2D.MolDrawOptions()
dopts.bondLineWidth = 5
dopts.maxFontSize = 1000
dopts.fixedFontSize = 1000
dopts.highlightBondWidthMultiplier = 6
dopts.highlightRadius = 0.6
dopts.legendFontSize=1000
img = Draw.MolsToGridImage(mols[:77], molsPerRow=8, useSVG=True, subImgSize=(200,200), highlightAtomLists=highlight, drawOptions=dopts,
                           legends=[str(x + 1) for x in range(77)])

with open('all_chem_structures_toluene.svg', 'w') as f:
  f.write(img)
