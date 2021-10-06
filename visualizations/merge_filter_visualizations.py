import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import os
import matplotlib.image as mpimg
import pandas as pd

if __name__ == '__main__':
#    folder = '/Users/jerometubiana/Downloads/ResidueAtomGAT3_51/AtomicFilters/'
    folder = sys.argv[1]
    try:
        model_name = sys.argv[2]
    except:
        model_name = 'ScanNet'
    is_atom = True #'Atom' in folder
    nfilters = 128
    if is_atom:
        positions = ['1','2','3']
    else:
        positions = ['4']
    nfilters_per_page = 4
    npages = nfilters//nfilters_per_page
    npositions = 1
    figsize = (6,6)
    dpi = 150

    if is_atom:
        prefix = 'atom'
    else:
        prefix = 'aa'

    filter_screenshots = [['filter_%s%s_view%s.png'%(prefix,index,position) for index in range(nfilters)] for position in positions]
    neighborhood_screenshots = [['filter_%s_topneighbor_%s_view%s.png'%(prefix,index,position) for index in range(nfilters)] for position in positions]
    filter_activity = ['filter_%s_activity_distribution_%s.png' % (prefix,index) for index in range(nfilters)]
#    filter_activity = ['filter_activity_distribution_%s.png'%index for index in range(nfilters)]

    if is_atom:
        table = pd.read_csv(folder+'Table_AtomicFilters_%s.csv'%model_name)
    else:
        table = pd.read_csv(folder + 'Table_AminoAcidFilters_%s.csv' % model_name)

    for page in range(npages):
        fig = plt.figure(figsize=(figsize[0] * (2*npositions + 1) ,  figsize[1] * nfilters_per_page))
        gs = gridspec.GridSpec(nfilters_per_page, 2*npositions+2,
                               width_ratios=[figsize[0]]*2*npositions + [figsize[0]/2] *2 )
        for x,index in enumerate(range(
                page * nfilters_per_page,
                min( (page +1)* nfilters_per_page, nfilters)  ) ):
            for y,position in enumerate(positions[:npositions] ):
                ax = fig.add_subplot(gs[x,y])
                try:
                    img = mpimg.imread(folder + filter_screenshots[y][index] )
                    ax.imshow(img)
                except:
                    print('Cound not find %s'%(filter_screenshots[y][index]))
                ax.axis('off')

                ax = fig.add_subplot(gs[x,npositions+y])
                try:
                    img = mpimg.imread(folder + neighborhood_screenshots[y][index] )
                    ax.imshow(img)
                except:
                    print('Cound not find %s'%(neighborhood_screenshots[y][index]))
                ax.axis('off')
            ax = fig.add_subplot(gs[x,2*npositions])
            try:
                img = mpimg.imread(folder + filter_activity[index])
                ax.imshow(img)
            except:
                print('Cound not find %s'%(filter_activity[index]) )
            ax.axis('off')

            ax = fig.add_subplot(gs[x,2*npositions+1])

            # try:
            line = table.iloc[index]
            if is_atom:
                messages = ['Filter %s' % index, '$R_\\mathrm{pred}^\\mathrm{max} = %.2f$' % line['Corr with prediction (MaxAAPool)'],
                            '$R_\\mathrm{pred}^\\mathrm{mean} = %.2f$' % line['Corr with prediction (MeanAAPool)'],
                            'Percent Active = %.f' % (line['Fraction of atoms with nonzero activity'] * 100),
                            'Pooling attention %.2f' % line['Contribution to pooling attention'],
                            'Pooling feature %.2f' % line['Contribution to pooling features']]

            else:
                messages = ['Filter %s' % index, '$R_\\mathrm{pred} = %.2f$' % line['Corr with prediction'],
                            'Percent Active = %.f' % (line['Fraction of aa with nonzero activity'] * 100),
                            'Pooling attention %.2f' % line['Contribution to pooling attention'],
                            'Pooling feature %.2f' % line['Contribution to pooling features']]

            big_message = '\n'.join(messages)
            big_message2 = '\n'.join(['Top activators'] + [line['Top %s activator' % l] for l in range(1, 11)] )



            ax.text(0.2, 0.8, big_message+'\n'+big_message2, fontsize=12,va='top',ha='left')
            ax.axis('off')
            # except:
            #     print('Could not add additional information')
            ax.axis('off')


        plt.tight_layout()
        fig.savefig(folder+'tmp_#%s.png'%page,dpi=dpi)
    command = 'pdfjoin ' + ' '.join(['%stmp_#%s.png'%(folder,page) for page in range(npages)]) +' -o all_filters_%s_%s.pdf'%( ('atom' if is_atom else 'aa'),  model_name)
    print(command)
    os.system(command)
    command = 'rm '+folder+'tmp_#*.png'
    os.system(command)





