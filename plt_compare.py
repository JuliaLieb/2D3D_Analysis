import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')


filename_on_l = 'C:\\2D3D_Analysis\\Results\\S20_ses1_run2_mean_erds_l.csv'
filename_on_r = 'C:\\2D3D_Analysis\\Results\\S20_ses1_run2_mean_erds_r.csv'
filename_off_l = 'C:\\2D3D_Analysis\\Results\\S20_ses1_run2mean_erds_calc_l_.csv'
filename_off_r = 'C:\\2D3D_Analysis\\Results\\S20_ses1_run2mean_erds_calc_r_.csv'
filename_off_l_nf = 'C:\\2D3D_Analysis\\Results\\S20_ses1_run2mean_erds_calc_l_2.csv'
filename_off_r_nf = 'C:\\2D3D_Analysis\\Results\\S20_ses1_run2mean_erds_calc_r_2.csv'
filename_other_l = "C:\\2D3D_Analysis\\Results\\S20-ses1-run2-left-ERDS-calculation"

erds_on_l = np.loadtxt(filename_on_l, delimiter=',')
erds_on_r =np.loadtxt(filename_on_r, delimiter=',')
erds_off_l = np.loadtxt(filename_off_l, delimiter=',')
erds_off_r = np.loadtxt(filename_off_r, delimiter=',')
erds_off_l_nf = np.loadtxt(filename_off_l_nf, delimiter=',')
erds_off_r_nf = np.loadtxt(filename_off_r_nf, delimiter=',')
erds_other_l = np.loadtxt(filename_other_l, delimiter=',').T

plt.plot(erds_on_l[:,2], label=f'online ROI {2+1}')
plt.plot(erds_off_l[:,2], label=f'offline ROI {2+1}')
#plt.plot(erds_off_l_nf[:,2], label=f'offline ROI nf {2+1}')
#plt.plot(erds_other_l[:,2], label=f'other ROI {2+1}')
plt.xlabel('Row Index')
plt.ylabel('Value')
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