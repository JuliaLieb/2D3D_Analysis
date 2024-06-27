import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

def plt_compar_on_off(path, subj, ses, run, rois):
    prefix_on_l = subj + '_ses' + str(ses) + '_run' + str(run) + '_online_ERDS_left'
    prefix_on_r = subj + '_ses' + str(ses) + '_run' + str(run) + '_online_ERDS_right'
    prefix_off_l = subj + '_ses' + str(ses) + '_run' + str(run) + '_offline_ERDS_left'
    prefix_off_r = subj + '_ses' + str(ses) + '_run' + str(run) + '_offline_ERDS_right'


    # calculated with main - plot online erds - ERDS_caculation.erds_values_plot_preparation
    list_online_r = []
    list_online_l = []
    for file in os.listdir(path):
        if file.startswith(prefix_on_l):
            list_online_l.append(path + file)
        if file.startswith(prefix_on_r):
            list_online_r.append(path + file)

    # calculated from eeg with main - calc_offline_online_ERDS - ERDS_calculation.erds_per_roi
    list_offline_r = []
    list_offline_l = []
    for file in os.listdir(path):
        if file.startswith(prefix_off_l):
            list_offline_l.append(path + file)
        if file.startswith(prefix_off_r):
            list_offline_r.append(path + file)

    erds_on_l = np.loadtxt(list_online_l[-1], delimiter=',')
    erds_on_r =np.loadtxt(list_online_r[-1], delimiter=',')
    erds_off_l = np.loadtxt(list_offline_l[-1], delimiter=',').T
    erds_off_r = np.loadtxt(list_offline_r[-1], delimiter=',').T
    erds_off_l2 = np.loadtxt(list_offline_l[-2], delimiter=',').T
    erds_off_r2 = np.loadtxt(list_offline_r[-2], delimiter=',').T

    mean_erds_on_l = np.mean(erds_on_l, axis=0)
    mean_erds_on_r = np.mean(erds_on_r, axis=0)
    mean_erds_off_l = np.mean(erds_off_l, axis=0)
    mean_erds_off_r = np.mean(erds_off_r, axis=0)
    mean_erds_off_l2 = np.mean(erds_off_l2, axis=0)
    mean_erds_off_r2 = np.mean(erds_off_r2, axis=0)

    #print(mean_erds_on_l)
    print(mean_erds_on_r)
    #print(mean_erds_off_l)
    print(mean_erds_off_r)
    #print(mean_erds_off_l2)
    #print(mean_erds_off_r2)

    for roi in rois:
        plt.plot(erds_on_l[:,roi], label=f'online left ROI {roi+1} calculated with {list_online_l[-1]}')
        #plt.plot(erds_on_r[:,roi], label=f'online right ROI {roi+1} calculated with {list_online_r[-1]}')
        plt.plot(erds_off_l[:, roi], label=f'offline left ROI {roi + 1} calculated with {list_offline_l[-1]}')
        #plt.plot(erds_off_r[:,roi], label=f'offline right ROI {roi+1} calculated with {list_offline_r[-1]}')
        plt.plot(erds_off_l2[:, roi], label=f'offline left ROI {roi + 1} calculated with {list_offline_l[-2]}')
        # plt.plot(erds_off_r2[:,roi], label=f'offline right ROI {roi+1} calculated with {list_offline_r[-2]}')
        plt.xlabel('Task No.')
        plt.ylabel('ERDS Value')
        plt.grid(True)
        plt.xticks(np.arange(0,len(erds_on_l),1))
        plt.title('ERDS comparison')
        plt.legend()
        plt.show()

    '''
    avg = []
    for roi in range(erds_on_l.shape[1]):
        avg.append(np.mean(erds_on_l[:,roi]))
    #for i in range (erds_on_l.shape[1]):
    #    plt.plot(erds_on_l[:,i], label=f'ROI {i+1}')
    plt.plot(erds_on_l[:,2], label=f'ROI {2+1}')
    plt.plot(erds_on_l[:,3], label=f'ROI {3+1}')
    plt.xlabel('Row Index')
    plt.ylabel('Value')
    plt.title(f'ERDS online left {avg:.4f}')
    plt.legend()
    plt.show()
    
    avg = []
    for roi in range(erds_on_r.shape[1]):
        avg.append(np.mean(erds_on_r[:,roi]))
    #for i in range (erds_on_r.shape[1]):
    #    plt.plot(erds_on_r[:,i], label=f'ROI {i+1}')
    plt.plot(erds_on_r[:,2], label=f'ROI {2+1}')
    plt.plot(erds_on_r[:,3], label=f'ROI {3+1}')
    plt.xlabel('Row Index')
    plt.ylabel('Value')
    plt.title(f'ERDS online right {avg:.4f}')
    plt.legend()
    plt.show()
    
    avg = []
    for roi in range(erds_off_l.shape[1]):
        avg.append(np.mean(erds_off_l[:,roi]))
    #for i in range (erds_off_l.shape[1]):
    #    plt.plot(erds_off_l[:,i], label=f'ROI {i+1}')
    plt.plot(erds_off_l[:,2], label=f'ROI {2+1}')
    plt.plot(erds_off_l[:,3], label=f'ROI {3+1}')
    plt.xlabel('Row Index')
    plt.ylabel('Value')
    plt.title(f'ERDS offline left {avg:.4f}')
    plt.legend()
    plt.show()
    
    avg = []
    for roi in range(erds_off_r.shape[1]):
        avg.append(np.mean(erds_off_r[:,roi]))
    #for i in range (erds_off_r.shape[1]):
    #    plt.plot(erds_off_r[:,i], label=f'ROI {i+1}')
    plt.plot(erds_off_r[:,2], label=f'ROI {2+1}')
    plt.plot(erds_off_r[:,3], label=f'ROI {3+1}')
    plt.xlabel('Row Index')
    plt.ylabel('Value')
    plt.title(f'ERDS offline right {avg:.4f}')
    plt.legend()
    plt.show()
    
    avg = []
    for roi in range(erds_off_l_nf.shape[1]):
        avg.append(np.mean(erds_off_l_nf[:,roi]))
    #for i in range (erds_off_l_nf.shape[1]):
    #    plt.plot(erds_off_l_nf[:,i], label=f'ROI {i+1}')
    plt.plot(erds_off_l_nf[:,2], label=f'ROI {2+1}')
    plt.plot(erds_off_l_nf[:,3], label=f'ROI {3+1}')
    plt.xlabel('Row Index')
    plt.ylabel('Value')
    plt.title(f'ERDS offline left V2 {avg:.4f}')
    plt.legend()
    plt.show()
    
    avg = []
    for roi in range(erds_off_r_nf.shape[1]):
        avg.append(np.mean(erds_off_r_nf[:,roi]))
    #for i in range (erds_off_r_nf.shape[1]):
    #    plt.plot(erds_off_r_nf[:,i], label=f'ROI {i+1}')
    plt.plot(erds_off_r_nf[:,2], label=f'ROI {2+1}')
    plt.plot(erds_off_r_nf[:,3], label=f'ROI {3+1}')
    plt.xlabel('Row Index')
    plt.ylabel('Value')
    plt.title(f'ERDS offline right V2 {avg:.4f}')
    plt.legend()
    plt.show()
    '''