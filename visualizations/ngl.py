import nglview
import pandas as pd

def inpaint(struct, predictions,mini=0,maxi=0.3,field='Binding site probability'):
    level =  struct.get_level()
    if isinstance(predictions,pd.DataFrame):
        L = len(predictions)
        for l in range(L):
            line = predictions.iloc[l]
            model = line['Model']
            chain = line['Chain']
            resid = int(line['Residue Index'])
            proba = line[field]
            color =  int( max( min(100, (1 - (proba-mini)/(maxi-mini) ) * 100 ), 0 ) )
            if level == 'S':
                residue = struct[model][chain][resid]
            elif level == 'M':
                residue = struct[chain][resid]
            elif level == 'C':
                residue = struct[resid]
            else:
                raise ValueError
            for atom in residue.get_list():
                atom.set_bfactor(color)
    elif isinstance(predictions,dict):
        for key,item in predictions.items():
            model,chain,resid = key
            proba = item
            color =  int( max( min(100, (1 - (proba-mini)/(maxi-mini) ) * 100 ), 0 ) )
            if level == 'S':
                residue = struct[model][chain][resid]
            elif level == 'M':
                residue = struct[chain][resid]
            elif level == 'C':
                residue = struct[resid]
            else:
                raise ValueError
            for atom in residue.get_list():
                atom.set_bfactor(color)
    return struct

def display(structure,predictions,
            representation = 'cartoon',
            mini=0,maxi=0.3,field='Binding site probability'):
    inpaint(structure, predictions, mini=mini, maxi=maxi, field=field)
    view = nglview.show_biopython(structure)
    view.clear_representations()
    view.add_representation(representation, selection='protein', color_scheme='bfactor')
    return view

